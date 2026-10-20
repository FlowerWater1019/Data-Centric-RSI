"""标签策略测试使用新的合成生成器，不读取正式数据。"""

import unittest

from rsi.contracts import ExperimentContract
from rsi.data import validate_dataset
from rsi.feedback import parse_feedback
from rsi.dispatch import apply_dispatch, parse_dispatch
from rsi.history import BatchHistory, HistoricalBatch
from rsi.labels import SkillOntology, parse_labels
from rsi.policy import parse_policy
from rsi.policy_selection import select_with_policy
from rsi.selection_demo import fixture_values
from rsi.validation import digest


class SelectionFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.values = fixture_values()
        self.ontology = SkillOntology.parse(self.values["ontology"])

    def inputs(self) -> tuple:
        contract = ExperimentContract.parse(self.values["config"])
        records = validate_dataset(self.values["data"], contract)
        labels = tuple(parse_labels(x, self.ontology) for x in self.values["labels"])
        policy = parse_policy(self.values["policy"], self.ontology)
        return contract, records, labels, self.ontology, policy

    def select(self, history: BatchHistory = BatchHistory(), *, synthetic: bool = True):
        return select_with_policy(*self.inputs(), history, allow_synthetic_labels=synthetic)

    def dispatch_context(self) -> tuple:
        result = self.select()
        policy = self.inputs()[-1]
        self.values["feedback"]["batch_id"] = "batch_001"
        self.values["dispatch"]["inputs"]["batch_id"] = "batch_001"
        ids = result.to_dict()["identities"]
        history = BatchHistory(batches=(HistoricalBatch("batch_001", digest(policy), ids["labels_hash"],
            ids["dataset_hash"], tuple(x.data_id for x in result.batch.items)),))
        return policy, parse_dispatch(self.values["dispatch"]), parse_feedback(self.values["feedback"], self.ontology), self.ontology, history

    def apply(self):
        return apply_dispatch(*self.dispatch_context())
