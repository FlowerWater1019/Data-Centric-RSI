"""导入真实数据并保留来源、划分和参考解验证证据。"""

from collections import Counter
from pathlib import Path
import re
import shutil
from typing import Any

from .artifacts import file_sha256, write_json
from .mbpp import MbppRecord, duplicate_groups, engineering_role
from .python_tasks import PythonTask, PythonTaskSuite
from .sandbox import BubblewrapExecutor, SandboxLimits
from .validation import ValidationError, canonical, digest, fail, integer, obj, read_json, read_jsonl, string, to_json_value


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def import_mbpp(source_path: str | Path, origin_path: str | Path, sandbox_path: str | Path,
                output_path: str | Path, limit_per_role: int | None = None) -> dict[str, Any]:
    """先冻结子集，再检查参考解。失败条目进入隔离清单，不补选替代条目。"""
    source, output = Path(source_path).resolve(), Path(output_path).resolve()
    if output.exists():
        fail("output", "导入目录必须尚不存在。")
    origin = obj(read_json(origin_path), "origin", {"dataset", "revision", "url", "license", "sha256"})
    for key in origin:
        string(origin[key], "origin." + key)
    if (origin["dataset"] != "google-research/mbpp" or origin["license"] != "CC-BY-4.0"
            or not re.fullmatch(r"[0-9a-f]{40}", origin["revision"])):
        fail("origin", "需要官方 MBPP 标识、完整提交号和 CC-BY-4.0 许可声明。")
    if origin["url"] != f"https://raw.githubusercontent.com/google-research/google-research/{origin['revision']}/mbpp/mbpp.jsonl":
        fail("origin.url", "来源 URL 必须对应已记录的官方提交。")
    if file_sha256(source) != origin["sha256"]:
        fail("source", "原始文件哈希与来源记录不一致。")
    records = tuple(MbppRecord.parse(raw) for raw in read_jsonl(source))
    if len({record.task_id for record in records}) != len(records):
        fail("mbpp", "任务编号重复。")
    if {record.task_id for record in records} != set(range(1, 975)):
        fail("mbpp", "导入要求完整官方文件，共 974 个任务。先检查划分，再取工程子集。")
    if limit_per_role is not None:
        integer(limit_per_role, "limit_per_role", 1)
    groups = duplicate_groups(records)
    group_splits: dict[str, set[str]] = {}
    for record in records:
        group_splits.setdefault(groups[record.task_id], set()).add(record.official_split)
    selected, excluded, roles = [], [], {}
    counts: Counter[str] = Counter()
    seen_groups = set()
    for record in sorted(records, key=lambda item: item.task_id):
        group = groups[record.task_id]
        role = engineering_role(record, group)
        reason = None
        if role in {"final_test", "excluded_prompting"}:
            reason = role
        elif len(group_splits[group]) > 1:
            reason = "duplicate_across_official_splits"
        elif group in seen_groups:
            reason = "duplicate_within_split"
        elif limit_per_role is not None and counts[role] >= limit_per_role:
            reason = "engineering_subset_limit"
        seen_groups.add(group)
        if reason:
            excluded.append({"task_id": record.task_id, "official_split": record.official_split, "reason": reason})
        else:
            selected.append(record)
            roles[record.task_id] = role
            counts[role] += 1
    executor = BubblewrapExecutor(SandboxLimits.parse(read_json(sandbox_path)))
    preflight = executor.preflight()
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, output / "upstream.jsonl")
    if file_sha256(output / "upstream.jsonl") != origin["sha256"]:
        fail("source", "源文件在导入期间变化。不能继续生成数据包。")
    write_json(output / "origin.json", origin)
    write_json(output / "sandbox.json", read_json(sandbox_path))
    write_json(output / "preflight.json", preflight)
    write_json(output / "selection.json", {"selected": [{"task_id": r.task_id, "role": roles[r.task_id]} for r in selected],
                                          "excluded": excluded, "limit_per_role": limit_per_role})
    sources, tasks, solutions, rejected = [], [], [], []
    calls = 0
    try:
        for record in selected:
            role, group = roles[record.task_id], groups[record.task_id]
            try:
                entry, signature = record.signature()
                prompt = record.prompt + "\nRequired signature: " + signature + "\nReturn complete Python code only."
                task = PythonTask.parse({"task_id": f"mbpp_{record.task_id}", "group": role if role in {"target", "retention"} else "target",
                    "split_group": group, "duplicate_group": group, "prompt": prompt, "entry_point": entry,
                    "setup": record.setup, "tests": list(record.tests)})
            except ValidationError as exc:
                rejected.append({"task_id": record.task_id, "reason": "format", "detail": str(exc)})
                continue
            results = []
            for index, test in enumerate(task.tests):
                # 每次测试独立隔离。调用前先留痕，进程中断也不会把尝试写成未执行。
                calls += 1
                write_json(output / "calls" / f"{calls:06d}.started.json", {"task_id": record.task_id, "test_index": index})
                result = executor.run(record.code, test, record.setup)
                write_json(output / "calls" / f"{calls:06d}.result.json", to_json_value(result))
                results.append(result)
            evidence = write_json(output / "reference_checks" / f"{record.task_id}.json",
                                  {"task_id": record.task_id, "tests": to_json_value(results)})
            if any(result.status != "passed" for result in results):
                rejected.append({"task_id": record.task_id, "reason": "reference_tests_failed", "evidence": to_json_value(evidence)})
                continue
            solution = (record.setup + "\n" if record.setup else "") + record.code
            if role in {"target", "retention"}:
                tasks.append(to_json_value(task))
                solutions.append({"task_id": task.task_id, "code": solution, "evidence_sha256": evidence.sha256})
            else:
                sources.append({"schema_version": "source-1", "data_id": task.task_id, "role": role,
                    "split_group": group, "duplicate_group": group, "skills": ["python_codegen"],
                    "quality": {"verification": "pass", "evidence_id": evidence.sha256, "integrity_valid": True, "leakage": "clear"},
                    "messages": [
                        {"role": "user", "content": prompt, "correctness": "context", "supervise": False,
                         "evidence_id": None, "diagnostics": []},
                        {"role": "assistant", "content": solution, "correctness": "valid", "supervise": True,
                         "evidence_id": evidence.sha256, "diagnostics": []}]})
        provenance = {key: origin[key] for key in ("dataset", "revision", "sha256", "license")}
        provenance["split_policy"] = "mbpp_official_then_group_hash_v1"
        task_suite = {"schema_version": "python-tasks-1", "purpose": "engineering_validation", "role": "feedback_execution",
                      "executor": "bubblewrap_seccomp_v1", "max_new_tokens": 512, "provenance": provenance, "tasks": tasks}
        PythonTaskSuite.parse(task_suite)
        if not sources or not any(row["role"] == "candidate" for row in sources):
            fail("sources", "参考解检查后没有可用训练候选。")
        write_jsonl(output / "sources.jsonl", sources)
        write_jsonl(output / "reference_solutions.jsonl", solutions)
        write_json(output / "feedback_tasks.json", task_suite)
        write_json(output / "rejected.json", rejected)
        manifest = {"schema_version": "import-bundle-1", "format": "mbpp_original", "data_kind": "real_public",
                    "purpose": "engineering_validation", "origin": origin, "raw_records": len(records),
                    "selected_counts": dict(counts), "exported_source_counts": dict(Counter(row["role"] for row in sources)),
                    "exported_feedback_tasks": len(tasks), "rejected_count": len(rejected), "test_calls_started": calls,
                    "final_test_executions": 0, "challenge_tests": "PRESERVED_UPSTREAM_NOT_EXECUTED",
                    "files": [{"name": str(path.relative_to(output)), "sha256": file_sha256(path)}
                              for path in sorted(output.rglob("*")) if path.is_file()]}
        artifact = write_json(output / "manifest.json", manifest)
        return {"status": "IMPORTED", "manifest": to_json_value(artifact), "sources": len(sources),
                "feedback_tasks": len(tasks), "rejected": len(rejected), "test_calls_started": calls}
    except BaseException as exc:
        write_json(output / "failure.json", {"status": "execution_error", "error": str(exc) or type(exc).__name__,
                   "test_calls_started": calls, "rejected_so_far": rejected})
        raise
