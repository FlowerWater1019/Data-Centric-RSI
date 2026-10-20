"""显式批次历史。成员来自清单，不从自然语言推断。"""

from dataclasses import dataclass
from typing import Any

from .labels import fields
from .validation import array, choice, fail, string, strings, to_json_value


@dataclass(frozen=True)
class HistoricalBatch:
    batch_id: str
    policy_hash: str
    labels_hash: str
    dataset_hash: str
    member_ids: tuple[str, ...]


@dataclass(frozen=True)
class BatchHistory:
    schema_version: str = "batch-history-v1"
    batches: tuple[HistoricalBatch, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return to_json_value(self)

    @classmethod
    def parse(cls, raw: Any) -> "BatchHistory":
        v = fields(raw, "history", "schema_version batches")
        choice(v["schema_version"], "history.schema_version", {"batch-history-v1"})
        batches = []
        for i, item in enumerate(array(v["batches"], "history.batches")):
            path = f"history.batches[{i}]"
            item = fields(item, path, "batch_id policy_hash labels_hash dataset_hash member_ids")
            identities = {k: string(x, path + "." + k) for k, x in item.items() if k != "member_ids"}
            batches.append(HistoricalBatch(**identities, member_ids=strings(item["member_ids"], path + ".member_ids", True)))
        if len({b.batch_id for b in batches}) != len(batches):
            fail("history.batches.batch_id", "批次 ID 重复。")
        return cls(batches=tuple(batches))

    def resolve(self, batch_ids: tuple[str, ...]) -> frozenset[str]:
        lookup = {b.batch_id: b for b in self.batches}
        if set(batch_ids) - lookup.keys():
            fail("policy.eligibility.deferred_batch_ids", "历史批次 ID 无法解析。")
        return frozenset(member for key in batch_ids for member in lookup[key].member_ids)
