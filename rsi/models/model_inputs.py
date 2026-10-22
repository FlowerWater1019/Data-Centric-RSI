"""根据本地模型内容冻结模型、分词器和模板引用。"""

from typing import Any

from rsi.configuration.validation import digest


def bound_model(identity: dict[str, Any], tokenizer: Any) -> dict[str, str]:
    tokenizer_files = [entry for entry in identity["files"] if entry["name"] in {
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt", "tokenizer.model",
    }]
    return {"checkpoint_id": "base_" + identity["content_id"], "base_revision": identity["content_id"],
            "adapter_revision": "none", "tokenizer_revision": digest(tokenizer_files),
            "template_revision": digest({"template": tokenizer.chat_template, "enable_thinking": False})}
