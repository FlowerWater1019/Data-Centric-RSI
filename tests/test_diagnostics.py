"""固定诊断聚合与关键字标注检查。"""

from rsi.diagnostics import summarize_metrics
from rsi.python_diagnostics import SKILLS, annotate_keywords
from rsi.tokenization import compile_trajectory
from rsi.contracts import ExperimentContract
from tests.helpers import Fixture
from tests.test_tokenization import CharacterTokenizer


class DiagnosticTests(Fixture):
    def test_weighted_mean_and_missing_are_distinct(self) -> None:
        metrics = {"a": {"supervised_tokens": 2, "nll": 1, "spans": [{"skill": "x", "token_count": 1, "ce_sum": 2}]},
                   "b": {"supervised_tokens": 6, "nll": 3, "spans": [{"skill": "x", "token_count": 3, "ce_sum": 3}]}}
        result = summarize_metrics(metrics, ("x", "y"))
        self.assertEqual(result["mean_ce"], 2.5)
        self.assertEqual(result["skills"]["x"]["mean_ce"], 1.25)
        self.assertEqual(result["skills"]["y"]["status"], "MISSING")
        self.assertIsNone(result["skills"]["y"]["mean_ce"])

    def test_keyword_annotation_ignores_strings_and_comments_and_keeps_loss_mask(self) -> None:
        code = "def f(x):\n    # if return\n    s = 'return'\n    if x:\n        return s\n    return x\n"
        source = {"schema_version": "source-1", "data_id": "fixture", "role": "candidate",
                  "split_group": "fixture", "duplicate_group": "fixture", "skills": [self.config["skills"][0]],
                  "quality": self.data[0]["quality"], "messages": [
                      {"role": "assistant", "content": code, "correctness": "valid", "supervise": True,
                       "evidence_id": "fixture", "diagnostics": []}]}
        self.config["skills"] += list(SKILLS)
        contract = ExperimentContract.parse(self.config)
        annotated = annotate_keywords(source)
        self.assertEqual(len(annotated["messages"][0]["diagnostics"]), 3)
        self.assertEqual(source["messages"][0]["diagnostics"], [])
        plain = compile_trajectory(source, CharacterTokenizer(), contract, 1024)
        marked = compile_trajectory(annotated, CharacterTokenizer(), contract, 1024)
        self.assertEqual(plain["tokenization"], marked["tokenization"])
        self.assertEqual(plain["supervision"], marked["supervision"])
        self.assertEqual(len(marked["diagnostic_spans"]), 3)
