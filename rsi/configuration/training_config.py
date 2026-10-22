"""将已校验合同显式导出为兼容的 1.1 配置，不修改原输入。"""

from typing import Any
from rsi.configuration.contracts import ExperimentContract
from rsi.configuration.validation import to_json_value


def contract_dict(contract: ExperimentContract) -> dict[str, Any]:
    raw = to_json_value(contract)
    raw["schema_version"] = "1.1"
    raw["selection"] = {"supervised_token_budget": raw.pop("token_budget"), "allowed_shortfall": raw.pop("allowed_shortfall"),
                        "min_skill_shares": raw.pop("min_skill_shares"), "packing_rule": raw.pop("packing_rule")}
    return raw
