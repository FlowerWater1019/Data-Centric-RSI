"""生成可核验的开发对照摘要。任务收益与筛选额外差值分开报告。"""

from pathlib import Path
from typing import Any

from rsi.storage.artifacts import file_sha256, verify_training_artifact, write_json
from rsi.evaluation.acceptance import AcceptanceRule, TaskOutcome, decide
from rsi.models.backends import ArtifactRef
from rsi.storage.state import StateStore
from rsi.configuration.validation import fail, read_json, to_json_value


def report_comparison(training_path: str | Path, evaluation_path: str | Path,
                      output_path: str | Path) -> dict[str, Any]:
    """核验报告、账本和证据后输出 JSON 与 Markdown。输出目录必须尚不存在。"""
    training, evaluation, output = (Path(p).resolve() for p in (training_path, evaluation_path, output_path))
    trained, evaluated = read_json(training / "report.json"), read_json(evaluation / "report.json")
    provenance = read_json(evaluation / "provenance.json")
    if provenance["pilot_report_sha256"] != file_sha256(training / "report.json"):
        fail("report", "评估没有引用当前训练报告。")
    if trained["training_status"] != "COMPLETED" or evaluated["task_execution"] != "COMPLETED":
        fail("report", "训练与评估必须均已完成。")
    with StateStore(training / "state.sqlite3", create=False, read_only=True) as state:
        snapshot = state.snapshot("training")
        if snapshot != trained["training_state"]:
            fail("report", "训练报告与账本不一致。")
        context = state.context("training")
    with StateStore(evaluation / "state.sqlite3", create=False, read_only=True) as state:
        if state.snapshot("evaluation") != evaluated["state"]:
            fail("report", "评估报告与账本不一致。")
    for key in ("tasks", "acceptance_rule"):
        ref = ArtifactRef(**evaluated[key])
        if file_sha256(ref.path) != ref.sha256:
            fail("report", "冻结任务或验收规则已经变化。")
    rule = AcceptanceRule.parse(read_json(evaluated["acceptance_rule"]["path"]))
    rates, outcomes = {}, {}
    for arm, result in evaluated["evaluations"].items():
        artifact = ArtifactRef(**result["evidence"])
        if file_sha256(artifact.path) != artifact.sha256:
            fail("report", "评估证据已变化。")
        evidence = read_json(artifact.path)
        if evidence["task_manifest_sha256"] != evaluated["tasks"]["sha256"]:
            fail("report", "任务清单不一致。")
        rows = evidence["tasks"]
        outcomes[arm] = tuple(TaskOutcome.parse(row["outcome"]) for row in rows)
        rates[arm] = {group: {"passed": sum(row["outcome"]["passed"] for row in rows if row["outcome"]["group"] == group),
                             "tasks": sum(row["outcome"]["group"] == group for row in rows)} for group in ("target", "retention")}
        rates[arm]["format_errors"] = sum(bool(row.get("format_error")) for row in rows)
    decisions = {arm: to_json_value(decide(outcomes["baseline"], outcomes[arm], rule)) for arm in ("random", "fixed")}
    comparison = to_json_value(decide(outcomes["random"], outcomes["fixed"], rule))
    if decisions != evaluated["decisions"] or comparison != evaluated["fixed_vs_random"]:
        fail("report", "评估判定与逐任务证据不一致。")
    costs = {}
    for arm in ("random", "fixed"):
        result = trained["arms"][arm]["result"]
        recorded = next((e["payload"]["result"] for e in snapshot["events"] if e["event_id"] == f"training:{arm}:completed"), None)
        if result != recorded:
            fail("report", "候选产物与账本不一致。")
        artifact = ArtifactRef(**result["artifact"])
        verify_training_artifact(artifact)
        diagnostics = read_json(training / arm / "training.json")
        costs[arm] = {"supervised_tokens": result["actual_supervised_tokens"],
                      "optimizer_steps": len(diagnostics["steps"]),
                      "context_tokens": diagnostics["context_tokens"], "update_seconds": diagnostics["update_seconds"],
                      "adapter_delta_l2": diagnostics["adapter_delta_l2"]}
    plan = snapshot["comparisons"][0]
    selected = {arm: {item["data_id"] for item in plan["arms"][arm]["items"]} for arm in ("random", "fixed")}
    union = selected["random"] | selected["fixed"]
    summary = {"schema_version": "development-summary-1", "purpose": "development",
               "data_kind": trained["data_kind"], "seed": context.contract.seed,
               "training_report_sha256": file_sha256(training / "report.json"),
               "evaluation_report_sha256": file_sha256(evaluation / "report.json"),
               "same_batch": trained["same_batch"], "batch_jaccard": len(selected["random"] & selected["fixed"]) / len(union),
               "selection": {arm: sorted(ids) for arm, ids in selected.items()},
               "task_results": rates, "training_costs": costs, "scoring_cost": trained["scoring_state"]["budgets"],
               "evaluation_cost": evaluated["state"]["budgets"], "sft_vs_baseline": evaluated["decisions"],
               "fixed_vs_random": evaluated["fixed_vs_random"],
               "diagnostics": trained["diagnostic_summary"], "model_promotion": "NOT_APPLIED",
               "research_evidence": "NOT_ESTABLISHED", "uncertainty_scope": evaluated["uncertainty_scope"]}
    output.mkdir(parents=True, exist_ok=False)
    ref = write_json(output / "summary.json", summary)
    lines = ["# 真实数据开发对照", "", "本报告描述一次配对运行。它不能单独证明筛选能力提升。", "",
             f"随机种子：{summary['seed']}。批次相同：{summary['same_batch']}。批次 Jaccard 重合度：{summary['batch_jaccard']:.3f}。", "",
             "| 模型 | 目标通过数 | 保持通过数 | 格式错误数 |", "| --- | ---: | ---: | ---: |"]
    for arm, values in rates.items():
        lines.append(f"| {arm} | {values['target']['passed']} / {values['target']['tasks']} | {values['retention']['passed']} / {values['retention']['tasks']} | {values['format_errors']} |")
    lines += ["", "格式错误保留在任务分母中。", "", "| 分支 | 监督 token | 上下文 token | 更新步数 | 更新耗时（秒） |", "| --- | ---: | ---: | ---: | ---: |"]
    for arm, values in costs.items():
        lines.append(f"| {arm} | {values['supervised_tokens']} | {values['context_tokens']} | {values['optimizer_steps']} | {values['update_seconds']:.3f} |")
    lines += ["", "更新耗时包括前向、反向和优化器更新。它不包含模型加载和检查点保存。等监督 token 不表示等计算成本。", "",
              f"固定选择相对随机选择的判定：{summary['fixed_vs_random']['status']}。",
              "目标差值和保持差值见 summary.json。普通微调收益另存于 sft_vs_baseline。",
              "", "片段 CE 是教师强制诊断。它不能替代自由生成任务通过率。缺失片段标为 MISSING。",
              "", "本轮模型不晋级。多个种子或多个比较的联合显著性尚未建立。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"summary": ref.path, "sha256": ref.sha256, "report": str(output / "report.md")}
