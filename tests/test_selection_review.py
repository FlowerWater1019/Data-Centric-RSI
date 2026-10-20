"""复核演示与历史、模型身份、输入快照之间的衔接。"""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi.history import BatchHistory, HistoricalBatch
from rsi.policy_selection import SelectionResult
from rsi.selection_demo import make_fixture, run_demo, select_inputs
from rsi.selection_io import SelectionInputs, finish_manifest, verify_manifest
from rsi.validation import ValidationError, canonical, digest, read_json


def replace_fixture_file(root: Path, name: str, value: object) -> None:
    """测试显式建立修改后的合成输入快照。"""
    (root / name).write_text(canonical(value), encoding="utf-8")
    (root / "manifest.json").unlink()
    finish_manifest(root, "fixture")


class SelectionReviewTests(unittest.TestCase):
    def test_demo_preserves_existing_history_and_respects_defer_removal(self) -> None:
        for remove in (False, True):
            with self.subTest(remove=remove), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                fixture, output = root / "fixture", root / "demo"
                make_fixture(fixture)
                old = HistoricalBatch("old_batch", digest("policy"), digest("labels"), digest("data"), ("sample_002",))
                replace_fixture_file(fixture, "batch_history.json", BatchHistory(batches=(old,)))
                policy = read_json(fixture / "policy.json")
                policy["eligibility"]["deferred_batch_ids"] = [old.batch_id]
                replace_fixture_file(fixture, "policy.json", policy)
                if remove:
                    dispatch = read_json(fixture / "dispatch.fixture.json")
                    dispatch["policy_update"]["remove"] = {"eligibility.deferred_batch_ids": [old.batch_id]}
                    replace_fixture_file(fixture, "dispatch.fixture.json", dispatch)
                run_demo(fixture, output)
                history = BatchHistory.parse(read_json(output / "batch_history.json"))
                self.assertEqual(history.batches[0], old)
                self.assertEqual(len(history.batches), 2)
                before = read_json(output / "before/selection_report.json")
                after = read_json(output / "after/selection_report.json")
                self.assertIn("deferred_batch", before["candidates"][1]["exclusion_reasons"])
                self.assertEqual(after["candidates"][1]["eligibility_pass"], remove)
                verify_manifest(output)

    def test_demo_rejects_parent_model_mismatch_even_when_feedback_and_dispatch_agree(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture, output = root / "fixture", root / "demo"
            make_fixture(fixture)
            feedback = read_json(fixture / "feedback.fixture.json")
            dispatch = read_json(fixture / "dispatch.fixture.json")
            feedback["parent_checkpoint_id"] = "another_parent"
            dispatch["inputs"]["parent_checkpoint_id"] = "another_parent"
            dispatch["controller"]["checkpoint_id"] = "another_parent"
            replace_fixture_file(fixture, "feedback.fixture.json", feedback)
            replace_fixture_file(fixture, "dispatch.fixture.json", dispatch)
            with self.assertRaisesRegex(ValidationError, "parent_checkpoint_id"):
                run_demo(fixture, output)
            self.assertFalse(output.exists())

    def test_seed_without_target_skill_does_not_request_impossible_fixture_quota(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture, output = root / "fixture", root / "demo"
            make_fixture(fixture, seed=45)
            labels = (fixture / "labels.jsonl").read_text(encoding="utf-8")
            self.assertNotIn('"skill_id":"regression_check"', labels)
            comparison = run_demo(fixture, output)
            self.assertFalse(comparison["membership_changed"])
            self.assertEqual(comparison["before_ids"], comparison["after_ids"])
            self.assertEqual(read_json(output / "after/batch.json")["skill_shares"]["regression_check"], 0)

    def test_demo_rejects_changed_input_snapshot_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture, output = root / "fixture", root / "demo"
            make_fixture(fixture)
            calls = 0

            def change_source(inputs: SelectionInputs, *, allow_synthetic_labels: bool = False) -> SelectionResult:
                nonlocal calls
                result = select_inputs(inputs, allow_synthetic_labels=allow_synthetic_labels)
                calls += 1
                if calls == 1:
                    policy = read_json(fixture / "policy.json")
                    policy["description"] = "changed during selection"
                    # 即使外部进程重建了有效清单，也不能替换本次读取的快照。
                    replace_fixture_file(fixture, "policy.json", policy)
                return result

            with patch("rsi.selection_demo.select_inputs", side_effect=change_source):
                with self.assertRaisesRegex(ValidationError, "fixture"):
                    run_demo(fixture, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".*.pending-*")), [])

    def test_custom_skill_lists_and_multiple_labels_complete_demo(self) -> None:
        cases = ((('only_skill',), 1), (('logic', 'repair'), 2),
                 (('code_generation', 'error_analysis', 'regression_check'), 2))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for index, (skills, count) in enumerate(cases):
                with self.subTest(skills=skills, count=count):
                    fixture, output = root / f"fixture{index}", root / f"demo{index}"
                    make_fixture(fixture, seed=17, skills=skills, labels_per_sample=count)
                    result = run_demo(fixture, output)
                    self.assertTrue(all(result["same_inputs"].values()))
                    for stage in ("before", "after"):
                        batch = read_json(output / stage / "batch.json")
                        self.assertEqual(batch["supervised_tokens"], 16)
                        self.assertEqual(len({item["data_id"] for item in batch["items"]}), 4)
                    verify_manifest(output)
            self.assertEqual(list(root.glob(".*.pending-*")), [])

    def test_two_selection_demo_rejects_stop_instead_of_executing_next_selection(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture, output = root / "fixture", root / "demo"
            make_fixture(fixture)
            dispatch = read_json(fixture / "dispatch.fixture.json")
            dispatch["next_action"] = "stop"
            replace_fixture_file(fixture, "dispatch.fixture.json", dispatch)
            with patch("rsi.selection_demo.select_inputs", wraps=select_inputs) as select:
                with self.assertRaisesRegex(ValidationError, "next_action"):
                    run_demo(fixture, output)
                self.assertLessEqual(select.call_count, 1)
            self.assertFalse(output.exists())
