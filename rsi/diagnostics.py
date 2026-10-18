"""聚合固定诊断集合。缺失片段不表示零误差或没有能力缺口。"""

from typing import Any

from .validation import fail, number


def summarize_metrics(metrics: dict[str, Any], skills: tuple[str, ...]) -> dict[str, Any]:
    """按有效 token 数聚合 CE。各技能分别计数，不跨重叠技能相加。"""
    total, ce_sum = 0, 0.0
    groups = {skill: {"token_count": 0, "ce_sum": 0.0, "record_ids": set()} for skill in skills}
    for data_id, metric in metrics.items():
        count = metric["supervised_tokens"]
        total += count
        ce_sum += number(metric["nll"], "diagnostics.nll") * count
        for span in metric["spans"]:
            if span["skill"] not in groups:
                fail("diagnostics.skill", "诊断技能不属于冻结配置。")
            group = groups[span["skill"]]
            group["token_count"] += span["token_count"]
            group["ce_sum"] += span["ce_sum"]
            group["record_ids"].add(data_id)
    return {"records": len(metrics), "supervised_tokens": total,
            "mean_ce": ce_sum / total if total else None,
            "skills": {skill: {"status": "MEASURED" if group["token_count"] else "MISSING",
                               "token_count": group["token_count"], "record_count": len(group["record_ids"]),
                               "mean_ce": group["ce_sum"] / group["token_count"] if group["token_count"] else None}
                       for skill, group in groups.items()},
            "interpretation": "teacher_forced_diagnostic_not_task_success"}
