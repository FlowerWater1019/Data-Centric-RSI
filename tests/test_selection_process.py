"""独立进程离线命令、原子发布和重复请求。"""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from rsi.artifacts import write_json
from rsi.leases import run_lease
from rsi.selection_io import finish_manifest, publish_directory, verify_manifest
from rsi.validation import ValidationError, read_json
from tests.helpers import ROOT


class SelectionProcessTests(unittest.TestCase):
    def call(self, arguments: list[str], expected: int = 0) -> dict:
        result = subprocess.run([sys.executable, "-m", "rsi", *arguments], cwd=ROOT,
                                env=dict(os.environ, PYTHONUTF8="1"), capture_output=True,
                                text=True, encoding="utf-8", timeout=30)
        self.assertEqual(result.returncode, expected, result.stderr)
        if expected:
            self.assertEqual(result.stdout, "")
            self.assertIn("错误", result.stderr)
            return {"error": result.stderr}
        value = json.loads(result.stdout)
        self.assertEqual(value["training_status"], "NOT_RUN")
        return value

    def selection_args(self, fixture: Path) -> list[str]:
        return ["--config", str(fixture / "config.json"), "--data", str(fixture / "trajectories.jsonl"),
                "--labels", str(fixture / "labels.jsonl"), "--ontology", str(fixture / "ontology.json"),
                "--policy", str(fixture / "policy.json")]

    def apply_args(self, fixture: Path, demo: Path, output: Path) -> list[str]:
        return ["apply-dispatch", "--policy", str(fixture / "policy.json"), "--ontology", str(fixture / "ontology.json"),
                "--dispatch", str(demo / "dispatch.fixture.json"), "--feedback", str(demo / "feedback.fixture.json"),
                "--batch-history", str(demo / "batch_history.json"), "--output", str(output)]

    def test_full_offline_workflow_and_idempotent_application(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture, demo, applied = root / "fixture", root / "demo", root / "applied"
            self.call(["make-selection-fixture", "--output", str(fixture), "--seed", "42"])
            args = self.selection_args(fixture)
            denied = self.call(["validate-selection", *args], 2)
            self.assertIn("allow-synthetic-labels", denied["error"])
            self.call(["validate-selection", *args, "--allow-synthetic-labels"])
            self.call(["select-with-policy", *args, "--allow-synthetic-labels", "--output", str(root / "selected")])
            result = self.call(["selection-demo", "--fixture", str(fixture), "--output", str(demo)])
            self.assertTrue(result["membership_changed"])
            self.assertTrue(all(result["same_inputs"].values()))
            self.assertTrue(result["same_seed"])
            self.assertEqual(result["task_execution_status"], "NOT_RUN")
            arguments = self.apply_args(fixture, demo, applied)
            first = self.call(arguments)
            self.assertEqual(first["status"], "applied")
            before_files = verify_manifest(applied)
            self.assertEqual(self.call(arguments)["status"], "already_applied")
            self.assertEqual(before_files, verify_manifest(applied))
            self.assertEqual(read_json(applied / "policy.parent.json"), read_json(fixture / "policy.json"))
            self.assertEqual(read_json(applied / "policy.after.json"), read_json(demo / "policy.after.json"))
            after_args = args[:-2] + ["--policy", str(applied / "policy.after.json"),
                                     "--batch-history", str(demo / "batch_history.json")]
            after = self.call(["select-with-policy", *after_args, "--allow-synthetic-labels", "--output", str(root / "after")])
            self.assertEqual([i["data_id"] for i in after["batch"]["items"]], result["after_ids"])
            verify_manifest(demo)
            self.call(["make-selection-fixture", "--output", str(fixture)], 2)
            # 同一 ID 的不同内容不能借重试改写策略。
            changed = read_json(demo / "dispatch.fixture.json")
            changed["natural_language_dispatch"] = "同 ID 的另一份说明。"
            write_json(root / "changed.json", changed)
            changed_args = list(arguments)
            changed_args[changed_args.index("--dispatch") + 1] = str(root / "changed.json")
            self.assertIn("冲突", self.call(changed_args, 2)["error"])

    def test_deterministic_demo_and_fixture_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ids = []
            for i in (1, 2):
                fixture, demo = root / f"fixture{i}", root / f"demo{i}"
                made = self.call(["make-selection-fixture", "--output", str(fixture), "--seed", "42"])
                result = self.call(["selection-demo", "--fixture", str(fixture), "--output", str(demo)])
                ids.append((made["content_id"], result["content_id"]))
                self.assertEqual(len(read_json(demo / "before" / "batch.json")["items"]), 4)
            self.assertEqual(ids[0], ids[1])

    def test_failed_patch_and_failed_budget_publish_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixture, demo = root / "fixture", root / "demo"
            self.call(["make-selection-fixture", "--output", str(fixture)])
            self.call(["selection-demo", "--fixture", str(fixture), "--output", str(demo)])
            raw = read_json(demo / "dispatch.fixture.json")
            raw["policy_update"]["set"] = {"execution.seed": 1}
            write_json(root / "bad_dispatch.json", raw)
            arguments = self.apply_args(fixture, demo, root / "bad_output")
            arguments[arguments.index("--dispatch") + 1] = str(root / "bad_dispatch.json")
            self.assertIn("白名单", self.call(arguments, 2)["error"])
            self.assertFalse((root / "bad_output").exists())
            policy = read_json(fixture / "policy.json")
            policy["batch_constraints"]["supervised_token_budget"] = 15
            write_json(root / "budget.json", policy)
            args = self.selection_args(fixture)
            args[-1] = str(root / "budget.json")
            self.assertIn("BUDGET_FAILED", self.call(["select-with-policy", *args, "--allow-synthetic-labels",
                           "--output", str(root / "bad_budget")], 2)["error"])
            self.assertFalse((root / "bad_budget").exists())
            self.assertEqual(list(root.glob("*.pending-*")), [])
            verify_manifest(fixture)

    def test_demo_rejects_output_inside_fixture_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            fixture = Path(folder) / "fixture"
            self.call(["make-selection-fixture", "--output", str(fixture)])
            identity = verify_manifest(fixture)
            nested = fixture / "nested-demo"
            result = self.call(["selection-demo", "--fixture", str(fixture), "--output", str(nested)], 2)
            self.assertIn("output", result["error"])
            self.assertFalse(nested.exists())
            self.assertEqual(verify_manifest(fixture), identity)

    def test_demo_reports_rankings_with_excluded_candidates(self) -> None:
        for phase in ("before", "after"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                fixture, demo = root / "fixture", root / "demo"
                self.call(["make-selection-fixture", "--output", str(fixture)])
                name = "policy.json" if phase == "before" else "dispatch.fixture.json"
                path = fixture / name
                value = read_json(path)
                if phase == "before":
                    value["eligibility"]["excluded_data_ids"] = ["sample_002"]
                else:
                    value["policy_update"]["add"] = {"eligibility.excluded_data_ids": ["sample_002"]}
                path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
                # 修改合成输入后，建立新的完整 fixture 清单。
                (fixture / "manifest.json").unlink()
                finish_manifest(fixture, "fixture")
                identity = verify_manifest(fixture)
                result = self.call(["selection-demo", "--fixture", str(fixture), "--output", str(demo)])
                for stage in ("before", "after"):
                    report = read_json(demo / stage / "selection_report.json")
                    excluded = next(row for row in report["candidates"] if row["data_id"] == "sample_002")
                    ranking = result[stage + "_ranking"]
                    if phase == "before" or stage == "after":
                        self.assertIsNone(excluded["ranking_position"])
                        self.assertFalse(excluded["eligibility_pass"])
                        self.assertNotIn("sample_002", ranking)
                        self.assertNotIn("sample_002", result[stage + "_ids"])
                        self.assertEqual(len(ranking), 9)
                    else:
                        self.assertIn("sample_002", ranking)
                        self.assertEqual(len(ranking), 10)
                    ranked_rows = sorted((row for row in report["candidates"] if row["ranking_position"] is not None),
                                         key=lambda row: row["ranking_position"])
                    self.assertEqual(ranking, [row["data_id"] for row in ranked_rows])
                    self.assertEqual(report["counts"]["selected"], 4)
                verify_manifest(demo)
                self.assertEqual(verify_manifest(fixture), identity)

    def test_publication_failure_locking_and_manifest_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "result"
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                with publish_directory(target) as stage:
                    write_json(stage / "partial.json", {"started": True})
                    raise RuntimeError("interrupted")
            self.assertFalse(target.exists())
            self.assertEqual(list(root.glob(".*.pending-*")), [])
            with run_lease(root), self.assertRaisesRegex(ValidationError, "写入者"):
                with publish_directory(target):
                    self.fail("不应获得第二把锁。")
            with publish_directory(target) as stage:
                write_json(stage / "complete.json", {"done": True})
                finish_manifest(stage, "test")
            verify_manifest(target)
            (target / "complete.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValidationError, "哈希"):
                verify_manifest(target)

    def test_fixture_rejects_unknown_settings_and_gpu_imports_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            write_json(root / "settings.json", {"skills": ["a"], "labels_per_sample": 1, "unknown": True})
            self.call(["make-selection-fixture", "--output", str(root / "bad"), "--settings", str(root / "settings.json")], 2)
            # 阻止网络连接，并检查基础流程未导入模型依赖。
            script = (
                "import socket,sys; "
                "socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(AssertionError('network')); "
                "from rsi.cli import main; "
                "assert main(['make-selection-fixture','--output',sys.argv[1]])==0; "
                "assert not any(x.split('.')[0] in {'torch','transformers','peft'} for x in sys.modules)"
            )
            result = subprocess.run([sys.executable, "-c", script, str(root / "offline")], cwd=ROOT,
                                    capture_output=True, text=True, encoding="utf-8", timeout=30,
                                    env=dict(os.environ, PYTHONUTF8="1"))
            self.assertEqual(result.returncode, 0, result.stderr)
