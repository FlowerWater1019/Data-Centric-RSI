"""模型评分与成本记录。候选和固定诊断集合共用评分规则。"""

import sys
from pathlib import Path
import time
from rsi.storage.artifacts import write_json
from rsi.configuration.validation import to_json_value
from rsi.datasets.data import TrajectoryRecord
from rsi.models.hf_backend import HfRuntime
from rsi.models.scoring import TokenMetrics
from rsi.storage.state import StateStore


def progress(stage: str) -> None:
    print("RSI pilot: " + stage, file=sys.stderr, flush=True)


def score_and_record(
    runtime: HfRuntime, records: tuple[TrajectoryRecord, ...], state: StateStore, label: str,
    evidence_dir: Path | None = None,
) -> dict[str, TokenMetrics]:
    total = sum(r.supervised_tokens for r in records)
    state.reserve("scoring", label, "scoring_tokens", total)
    metrics = {}
    try:
        for index, record in enumerate(records):
            prefix = evidence_dir / label / f"{index:06d}" if evidence_dir else None
            if prefix is not None:
                write_json(str(prefix) + ".started.json", {"data_id": record.data_id,
                           "supervised_tokens": record.supervised_tokens, "context_tokens": record.token_count})
            start = time.monotonic()
            metric = runtime.score(record)
            state.consume("scoring", label + ":" + record.data_id, label, metric.supervised_tokens)
            metrics[record.data_id] = metric
            if prefix is not None:
                write_json(str(prefix) + ".result.json", {"metric": to_json_value(metric), "elapsed_seconds": time.monotonic() - start})
    except BaseException as exc:
        state.record_error("scoring", label, str(exc) or type(exc).__name__)
        raise
    finally:
        state.release("scoring", label)
    return metrics
