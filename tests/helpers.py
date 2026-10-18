"""测试专用的合成输入。不接触正式实验数据。"""

from pathlib import Path
import unittest

from rsi.contracts import ExperimentContract
from rsi.data import validate_dataset
from rsi.selection import SelectionContext
from rsi.validation import digest, read_json, read_jsonl


ROOT = Path(__file__).resolve().parents[1]


class Fixture(unittest.TestCase):
    def setUp(self) -> None:
        self.config = read_json(ROOT / "configs/smoke.json")
        self.data = read_jsonl(ROOT / "examples/trajectories.demo.jsonl")
        self.contract = ExperimentContract.parse(self.config)

    def context(self, data: list[dict] | None = None, config: dict | None = None) -> SelectionContext:
        data = self.data if data is None else data
        config = self.config if config is None else config
        contract = ExperimentContract.parse(config)
        return SelectionContext(contract, validate_dataset(data, contract), digest(config), digest(data))
