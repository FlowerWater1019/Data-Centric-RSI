"""显式模拟后端。所有输出必须标记 synthetic_fixture，不能当作参数测量。"""

from collections.abc import Callable
from pathlib import Path

from rsi.storage.artifacts import file_sha256, write_json
from rsi.models.backends import TrainingRequest, TrainingResult
from rsi.storage.checkpoints import StepJournal, TrainingPaused, request_identity
from rsi.configuration.contracts import ModelRef
from rsi.datasets.data import TrajectoryRecord
from rsi.evaluation.fit_evaluation import TokenScorer
from rsi.models.scoring import TokenScore
from rsi.configuration.validation import fail, read_json


class MockTrainingBackend:
    def __init__(self, root: Path, device: str, records: tuple[TrajectoryRecord, ...],
                 on_step: Callable[[int, TrajectoryRecord, int], None], stop_after_step: int | None = None) -> None:
        self.root, self.device, self.records, self.on_step, self.stop = root, device, records, on_step, stop_after_step

    def train(self, request: TrainingRequest) -> TrainingResult:
        journal = StepJournal(self.root, request_identity(request, list(self.records), self.device))
        completed = 0
        if request.resume_artifact:
            path, completed = journal.validate_resume(request.resume_artifact)
            if read_json(path) != {"completed": completed, "source": "synthetic_fixture"}:
                fail("mock.checkpoint", "模拟状态无效。")
        else:
            self.root.mkdir(exist_ok=False)
        for step, record in enumerate(self.records, 1):
            if step > completed:
                journal.begin(step)
            self.on_step(step, record, record.supervised_tokens)
            if step > completed:
                artifact = journal.publish(step, lambda path, value=step: write_json(path, {"completed": value, "source": "synthetic_fixture"}))
                if self.stop == step:
                    raise TrainingPaused(artifact)
        write_json(self.root / "adapter/fixture.json", {"source": "synthetic_fixture", "parent": request.starting_model.checkpoint_id,
                    "steps": len(self.records), "real_parameters_updated": False})
        diagnostic = write_json(self.root / "training.json", {"source": "synthetic_fixture", "execution_mode": "mock",
                     "training_status": "COMPLETED", "actual_supervised_tokens": sum(r.supervised_tokens for r in self.records),
                     "real_parameters_updated": False, "loss_reduction": "mean_per_trajectory_one_optimizer_update"})
        artifact = write_json(self.root / "manifest.json", {"schema_version": "mock-training-v1", "source": "synthetic_fixture",
                  "files": [{"name": p.relative_to(self.root).as_posix(), "sha256": file_sha256(p)}
                            for p in sorted(self.root.rglob("*")) if p.is_file()]})
        parent = request.starting_model
        model = ModelRef("mock_candidate_" + artifact.sha256, parent.base_revision, artifact.sha256,
                         parent.tokenizer_revision, parent.template_revision)
        return TrainingResult(model, artifact, sum(r.supervised_tokens for r in self.records), diagnostic)


class MockTokenScorer:
    def __init__(self, candidate: bool = False) -> None:
        self.candidate = candidate

    def score_tokens(self, record: TrajectoryRecord) -> tuple[TokenScore, ...]:
        return tuple(TokenScore(i, 1.0 - .1 * self.candidate + (i % 3) * .05, 2.0 - .1 * self.candidate)
                     for i, enabled in enumerate(record.raw["supervision"]["loss_mask"]) if enabled)
