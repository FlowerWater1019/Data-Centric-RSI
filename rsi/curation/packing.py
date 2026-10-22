"""按排名选择完整轨迹。后缀可达集合用于避免贪心预算缺口。"""

from rsi.datasets.data import TrajectoryRecord
from rsi.configuration.validation import BudgetError, fail, integer


def exact_ranked(records: list[TrajectoryRecord], budget: int) -> list[TrajectoryRecord]:
    """返回字典序最优的可行成员向量。优先保留排名靠前的完整轨迹。

    输入必须预先去重。本函数只求解 token 总量，不求解技能配额。
    搜索规模受限。超限或无解时抛出错误，不放宽预算。
    """
    integer(budget, "packing.budget", 1)
    if budget > 1_000_000 or len(records) > 4096 or (len(records) + 1) * (budget + 1) > 64_000_000:
        fail("packing", "精确组批超过搜索规模限制。请缩小候选池或预算。")
    for field in ("data_id", "duplicate_group", "content_hash"):
        if len({getattr(record, field) for record in records}) != len(records):
            fail("packing", "精确组批要求候选池已按编号、重复组和内容去重。")
    lengths = [integer(record.supervised_tokens, "supervised_tokens", 1) for record in records]
    limit = (1 << (budget + 1)) - 1
    suffix = [0] * (len(records) + 1)
    suffix[-1] = 1
    for i in range(len(records) - 1, -1, -1):
        suffix[i] = suffix[i + 1]
        if lengths[i] <= budget:
            suffix[i] |= (suffix[i + 1] << lengths[i]) & limit
    if not (suffix[0] >> budget) & 1:
        raise BudgetError("packing: 完整轨迹无法达到指定监督 token 总量。没有截断或重复数据。")
    selected, remaining = [], budget
    for i, record in enumerate(records):
        after = remaining - lengths[i]
        if after >= 0 and (suffix[i + 1] >> after) & 1:
            selected.append(record)
            remaining = after
    return selected
