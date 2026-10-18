"""把带角色和证据范围的消息转换为真实 token 轨迹。"""

from typing import Any, Protocol

from .contracts import ExperimentContract
from .data import TrajectoryRecord
from .validation import array, boolean, choice, digest, fail, integer, obj, string, strings, text_digest


class ChatTokenizer(Protocol):
    is_fast: bool
    chat_template: str

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...
    def __call__(self, text: str, **kwargs: Any) -> Any: ...
    def decode(self, ids: list[int], **kwargs: Any) -> str: ...


def render_with_ranges(
    tokenizer: ChatTokenizer, messages: list[dict[str, Any]],
) -> tuple[str, list[tuple[int, int]]]:
    """通过唯一占位符定位内容。模板依赖内容而改变结构时，拒绝导入。"""
    chat = [{"role": m["role"], "content": m["content"]} for m in messages]
    options = {"tokenize": False, "add_generation_prompt": False, "enable_thinking": False}
    text = tokenizer.apply_chat_template(chat, **options)
    markers = [f"__RSI_CONTENT_{digest(chat)}_{i}__" for i in range(len(chat))]
    marked = [{"role": m["role"], "content": marker} for m, marker in zip(chat, markers)]
    template = tokenizer.apply_chat_template(marked, **options)
    ranges = []
    cursor = 0
    for message, marker in zip(chat, markers):
        if template.count(marker) != 1 or marker in text:
            fail("chat_template", "模板不能提供唯一的消息内容位置。")
        start = template.find(marker, cursor)
        if start < 0:
            fail("chat_template", "模板改变了消息顺序。")
        template = template[:start] + message["content"] + template[start + len(marker):]
        cursor = start + len(message["content"])
        ranges.append((start, cursor))
    if template != text:
        fail("chat_template", "模板会按内容改变文本。需要专用位置适配器。")
    return text, ranges


def compile_trajectory(
    source: dict[str, Any], tokenizer: ChatTokenizer, contract: ExperimentContract, max_length: int,
) -> dict[str, Any]:
    """完整编码且不截断。模板 token 和跨内容边界 token 不进入监督。"""
    value = obj(source, "source", {"schema_version", "data_id", "role", "split_group", "duplicate_group",
                "skills", "quality", "messages"})
    if value["schema_version"] != "source-1":
        fail("source.schema_version", "仅支持 source-1。")
    if not tokenizer.is_fast:
        fail("tokenizer", "需要支持字符位置的 fast tokenizer。")
    skills = strings(value["skills"], "source.skills")
    messages = array(value["messages"], "source.messages", True)
    for i, message in enumerate(messages):
        path = f"source.messages[{i}]"
        obj(message, path, {"role", "content", "correctness", "supervise", "evidence_id", "diagnostics"})
        role = choice(message["role"], path + ".role", {"system", "user", "assistant", "tool"})
        string(message["content"], path + ".content")
        correct = choice(message["correctness"], path + ".correctness", {"valid", "incorrect", "unknown", "context"})
        if boolean(message["supervise"], path + ".supervise"):
            if role != "assistant" or correct != "valid":
                fail(path, "只有有效 assistant 消息可以请求监督。")
            string(message["evidence_id"], path + ".evidence_id")
        for span in array(message["diagnostics"], path + ".diagnostics"):
            obj(span, path, {"skill", "start", "end"})
            if string(span["skill"], path) not in skills:
                fail(path, "诊断技能未列入轨迹标签。")
            start = integer(span["start"], path)
            end = integer(span["end"], path, 1)
            if not start < end <= len(message["content"]):
                fail(path, "诊断字符区间越界。")
    text, ranges = render_with_ranges(tokenizer, messages)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids = list(encoded["input_ids"])
    offsets = [list(pair) for pair in encoded["offset_mapping"]]
    if len(ids) > max_length:
        fail(value["data_id"], "轨迹超过最大长度。程序不会自动截断。")
    if tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
        fail(value["data_id"], "分词往返文本不一致。")
    roles, correctness, mask = ["system"] * len(ids), ["context"] * len(ids), [False] * len(ids)
    spans = []
    for message, (left, right) in zip(messages, ranges):
        for index, (start, end) in enumerate(offsets):
            if left <= start < end <= right:
                roles[index], correctness[index] = message["role"], message["correctness"]
                mask[index] = bool(index > 0 and message["supervise"])
        for span in message["diagnostics"]:
            covered = [i for i, (start, end) in enumerate(offsets)
                       if left + span["start"] <= start < end <= left + span["end"] and mask[i]]
            if not covered:
                fail(value["data_id"], "诊断片段没有完整的有效监督 token。")
            start, end = min(covered), max(covered) + 1
            spans.append({"skill": span["skill"], "token_start": start, "token_end": end,
                          "char_start": offsets[start][0], "char_end": offsets[end - 1][1],
                          "evidence_id": message["evidence_id"], "mask_version": "content_spans_v1"})
    record = {
        "schema_version": "1.0", "data_id": value["data_id"], "version": digest(source),
        "role": value["role"], "split_group": value["split_group"], "duplicate_group": value["duplicate_group"],
        "text": text, "text_sha256": text_digest(text), "skills": list(skills), "quality": value["quality"],
        "tokenization": {"status": "verified", "tokenizer_revision": contract.model.tokenizer_revision,
                         "template_revision": contract.model.template_revision, "token_ids": ids,
                         "offsets": offsets, "roles": roles, "correctness": correctness},
        "supervision": {"mask_version": "assistant_content_v1", "loss_mask": mask},
        "diagnostic_spans": spans,
        "usage": {"ever_used_in_training_attempt": False, "in_accepted_lineage": False,
                  "training_attempt_ids": [], "cumulative_supervised_tokens": 0}, "scores": None,
    }
    TrajectoryRecord.parse(record, contract, value["data_id"])
    return record
