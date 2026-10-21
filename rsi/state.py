"""SQLite 是运行状态的唯一写入来源。重复请求不能覆盖事实。"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from types import TracebackType
from typing import Any

from .backends import TrainingResult, TrainingUse
from .comparison import build_comparison
from .contracts import ExperimentContract
from .data import validate_dataset
from .schema import ensure_schema
from .selection import BatchManifest, SelectionContext
from .validation import canonical, choice, digest, fail, integer, loads, string, to_json_value


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class StateStore:
    """保存运行事实。每个线程使用独立实例。上下文退出时关闭连接。"""

    def __init__(self, path: str | Path, *, create: bool = True, read_only: bool = False) -> None:
        """默认允许创建空库。只读模式与 create=False 均禁止创建文件。"""
        path = Path(path).resolve()
        if create and not read_only:
            path.parent.mkdir(parents=True, exist_ok=True)
        mode = "ro" if read_only else ("rwc" if create else "rw")
        self.connection = sqlite3.connect(
            path.as_uri() + "?mode=" + mode, uri=True, timeout=30, isolation_level=None
        )
        try:
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys=ON")
            ensure_schema(self.connection, allow_create=create and not read_only)
        except BaseException:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "StateStore":
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc_value: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        self.close()

    @contextmanager
    def transaction(self, *, write: bool = True) -> Iterator[None]:
        """写事务提前申请写锁。读事务提供一致快照，且不申请写锁。"""
        if self.connection.in_transaction:
            fail("transaction", "不支持嵌套事务。请在事务外调用状态接口。")
        self.connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
        try:
            yield
            self.connection.execute("COMMIT")
        except BaseException:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise

    def _run(self, run_id: str) -> sqlite3.Row:
        row = self.connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            fail("run_id", "运行不存在。")
        return row

    def _event(self, run_id: str, event_id: str, kind: str, payload: dict[str, Any]) -> bool:
        content = canonical(payload)
        old = self.connection.execute("SELECT kind,payload_json FROM events WHERE run_id=? AND event_id=?",
                                      (run_id, event_id)).fetchone()
        if old:
            if old["kind"] != kind or old["payload_json"] != content:
                fail("event_id", "相同编号对应不同内容，禁止覆盖。")
            return False
        self.connection.execute("INSERT INTO events(run_id,event_id,kind,payload_json,created_at) VALUES (?,?,?,?,?)",
                                (run_id, event_id, kind, content, now()))
        return True

    def initialize(self, run_id: str, mode: str, contract_raw: dict[str, Any], data_raw: list[dict[str, Any]]) -> bool:
        """校验并冻结输入。相同请求返回 False；同编号的不同输入被拒绝。"""
        string(run_id, "run_id")
        choice(mode, "mode", {"online", "controlled"})
        contract = ExperimentContract.parse(contract_raw)
        validate_dataset(data_raw, contract)
        snapshot = digest({"mode": mode, "contract": contract_raw, "dataset": data_raw})
        with self.transaction():
            old = self.connection.execute("SELECT snapshot_hash FROM runs WHERE run_id=?", (run_id,)).fetchone()
            if old:
                if old[0] != snapshot:
                    fail("run_id", "已有运行的输入不同。请使用新的运行编号。")
                return False
            self.connection.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?)", (run_id, mode, canonical(contract_raw),
                                    canonical(data_raw), snapshot, "validated", now()))
            self._event(run_id, "initialize", "initialized", {"snapshot_hash": snapshot,
                        "validation_level": "structural", "training_status": "NOT_RUN", "purpose": contract.purpose})
        return True

    def context(self, run_id: str) -> SelectionContext:
        """从数据库快照重建选择情境，不读取外部输入文件。"""
        row = self._run(run_id)
        raw_contract = loads(row["contract_json"])
        raw_data = loads(row["dataset_json"])
        contract = ExperimentContract.parse(raw_contract)
        records = validate_dataset(raw_data, contract)
        return SelectionContext(contract, records, digest(raw_contract), digest(raw_data))

    def prepare_comparison(self, run_id: str, comparison_id: str) -> dict[str, Any]:
        """原子保存纯计算模块生成的计划。重复请求返回原计划。"""
        string(comparison_id, "comparison_id")
        with self.transaction():
            row = self._run(run_id)
            if row["mode"] != "controlled":
                fail("mode", "训练对照必须使用 controlled 运行。")
            old = self.connection.execute("SELECT payload_json FROM comparisons WHERE run_id=? AND comparison_id=?",
                                          (run_id, comparison_id)).fetchone()
            if old:
                return loads(old[0])
            if row["status"] not in {"validated", "planned"}:
                fail("run.status", "当前状态不允许创建新计划。")
            context = self.context(run_id)
            payload = build_comparison(context, run_id, comparison_id).to_dict()
            self.connection.execute("INSERT INTO comparisons VALUES (?,?,?)", (run_id, comparison_id, canonical(payload)))
            self.connection.execute("UPDATE runs SET status='planned' WHERE run_id=?", (run_id,))
            self._event(run_id, "comparison:" + comparison_id, "comparison_planned",
                        {"comparison_id": comparison_id, "plan_hash": digest(payload), "training_status": "NOT_RUN"})
            return payload

    def _budget(self, run_id: str, unit: str) -> dict[str, Any]:
        limits = loads(self._run(run_id)["contract_json"])["budgets"]
        if unit not in limits:
            fail("unit", "成本单位不在实验预算中。")
        used = self.connection.execute("SELECT COALESCE(SUM(amount),0) FROM costs WHERE run_id=? AND unit=?",
                                       (run_id, unit)).fetchone()[0]
        reserved = self.connection.execute("SELECT COALESCE(SUM(remaining),0) FROM reservations WHERE run_id=? AND unit=?",
                                           (run_id, unit)).fetchone()[0]
        return {"limit": limits[unit], "used": used, "reserved": reserved, "available": limits[unit] - used - reserved}

    def prepare_training_plan(self, run_id: str, attempt_id: str, batch: BatchManifest, plan_hash: str) -> bool:
        """原子登记单批次计划。不重选数据，不创建虚假 comparison。"""
        string(attempt_id, "attempt_id")
        string(plan_hash, "plan_hash")
        payload = {"schema_version": "single-training-plan-v1", "attempt_id": attempt_id,
                   "batch": batch.to_dict(), "plan_hash": plan_hash}
        with self.transaction():
            event_id = "training_plan:" + attempt_id
            if self.connection.execute("SELECT 1 FROM events WHERE run_id=? AND event_id=?", (run_id, event_id)).fetchone():
                return self._event(run_id, event_id, "single_training_plan", payload)
            if self._run(run_id)["status"] not in {"validated", "planned"}:
                fail("training_plan", "当前状态不能新增计划。")
            context = self.context(run_id)
            records = {r.data_id: r for r in context.records}
            if batch.model != context.contract.model or batch.supervised_tokens > context.contract.token_budget:
                fail("training_plan", "模型不匹配或单批预算超限。")
            if len({item.data_id for item in batch.items}) != len(batch.items) or not batch.items:
                fail("training_plan.items", "成员必须非空且不重复。")
            selected = []
            for item in batch.items:
                record = records.get(item.data_id)
                if record is None or not record.is_eligible or item.repetitions != 1 or (
                    item.version, item.mask_version, item.content_hash, item.supervised_tokens
                ) != (record.version, record.train_mask_version, record.content_hash, record.supervised_tokens):
                    fail("training_plan.items", "成员与冻结可选轨迹不一致。")
                selected.append(record)
            if sum(r.supervised_tokens for r in selected) != batch.supervised_tokens:
                fail("training_plan.supervised_tokens", "监督总量不一致。")
            inserted = self._event(run_id, event_id, "single_training_plan", payload)
            self.connection.execute("UPDATE runs SET status='planned' WHERE run_id=?", (run_id,))
            return inserted

    def _frozen_batches(self, run_id: str, attempt_id: str) -> list[dict[str, Any]]:
        """旧对照与新单计划共用合法批次读取，不在执行时重新选数。"""
        single = self.connection.execute("SELECT payload_json FROM events WHERE run_id=? AND event_id=?",
                       (run_id, "training_plan:" + attempt_id)).fetchone()
        if single is not None:
            return [loads(single[0])["batch"]]
        plans = self.connection.execute("SELECT payload_json FROM comparisons WHERE run_id=?", (run_id,)).fetchall()
        result = [arm for plan in plans for arm in loads(plan[0])["arms"].values()]
        return result

    def reserve(self, run_id: str, reservation_id: str, unit: str, amount: int) -> bool:
        string(reservation_id, "reservation_id")
        integer(amount, "amount", 1)
        with self.transaction():
            old = self.connection.execute("SELECT unit,original FROM reservations WHERE run_id=? AND reservation_id=?",
                                          (run_id, reservation_id)).fetchone()
            if old:
                if old["unit"] != unit or old["original"] != amount:
                    fail("reservation_id", "重复预留的内容不同。")
                return False
            if self._run(run_id)["status"] == "execution_error":
                fail("run.status", "错误状态不能预留新的作业预算。")
            if self._budget(run_id, unit)["available"] < amount:
                fail("budget", "已消耗与已预留之和将超过预算。")
            self.connection.execute("INSERT INTO reservations VALUES (?,?,?,?,?,0)",
                                    (run_id, reservation_id, unit, amount, amount))
            self._event(run_id, "reserve:" + reservation_id, "budget_reserved", {"unit": unit, "amount": amount})
            return True

    def consume(self, run_id: str, event_id: str, reservation_id: str, amount: int,
                training_use: TrainingUse | None = None) -> bool:
        """记录实际消耗。实际超额时仍记账，并阻止新的作业。"""
        integer(amount, "amount", 1)
        string(event_id, "event_id")
        with self.transaction():
            old = self.connection.execute("SELECT reservation_id,amount FROM costs WHERE run_id=? AND event_id=?",
                                          (run_id, event_id)).fetchone()
            if old:
                if old["reservation_id"] != reservation_id or old["amount"] != amount:
                    fail("event_id", "同一消耗编号对应不同内容。")
                old_event = self.connection.execute(
                    "SELECT payload_json FROM events WHERE run_id=? AND event_id=?", (run_id, "cost:" + event_id)
                ).fetchone()
                if loads(old_event[0]).get("training_use") != to_json_value(training_use):
                    fail("event_id", "同一消耗编号对应不同训练使用。")
                return False
            reservation = self.connection.execute("SELECT * FROM reservations WHERE run_id=? AND reservation_id=?",
                                                  (run_id, reservation_id)).fetchone()
            if reservation is None:
                fail("reservation_id", "必须先登记作业预算预留。")
            unit = reservation["unit"]
            if training_use is not None:
                if unit != "training_tokens":
                    fail("training_use", "训练使用只能写入 training_tokens 消耗。")
                integer(training_use.step, "training_use.step", 1)
                record = next((r for r in self.context(run_id).records if r.data_id == training_use.data_id), None)
                if record is None or not record.is_eligible or (
                    record.version, record.train_mask_version, record.supervised_tokens
                ) != (training_use.version, training_use.mask_version, amount):
                    fail("training_use", "训练使用与冻结轨迹不一致。")
                if self._training_phase(run_id, training_use.attempt_id) != "started":
                    fail("training_use", "必须先登记正在运行的训练尝试。")
                start = self.connection.execute(
                    "SELECT payload_json FROM events WHERE run_id=? AND event_id=?",
                    (run_id, f"training:{training_use.attempt_id}:started"),
                ).fetchone()
                fingerprint = loads(start[0])["batch_fingerprint"]
                allowed = {item["data_id"] for arm in self._frozen_batches(run_id, training_use.attempt_id)
                           if arm["fingerprint"] == fingerprint for item in arm["items"]}
                if training_use.data_id not in allowed:
                    fail("training_use", "数据不属于该训练尝试的冻结批次。")
                single = self.connection.execute("SELECT payload_json FROM events WHERE run_id=? AND event_id=?",
                              (run_id, "training_plan:" + training_use.attempt_id)).fetchone()
                if single is not None:
                    items = loads(single[0])["batch"]["items"]
                    if training_use.step > len(items) or items[training_use.step - 1]["data_id"] != training_use.data_id:
                        fail("training_use.step", "单批计划必须按冻结顺序执行，每条轨迹只执行一次。")
                    previous_uses = self.connection.execute("SELECT payload_json FROM events WHERE run_id=? AND kind='cost_recorded'", (run_id,)).fetchall()
                    uses = [p["training_use"] for e in previous_uses if (p := loads(e[0])).get("training_use", {}).get("attempt_id") == training_use.attempt_id]
                    if training_use.step != len(uses) + 1:
                        fail("training_use.step", "步骤不能重复或跳过。恢复须复用原消耗事件编号。")
            overrun = bool(reservation["closed"]) or amount > reservation["remaining"]
            self.connection.execute("INSERT INTO costs VALUES (?,?,?,?,?)", (run_id, event_id, reservation_id, unit, amount))
            self.connection.execute("UPDATE reservations SET remaining=MAX(0,remaining-?) WHERE run_id=? AND reservation_id=?",
                                    (amount, run_id, reservation_id))
            payload = {"reservation_id": reservation_id, "unit": unit, "amount": amount, "overrun": overrun}
            if training_use is not None:
                payload["training_use"] = to_json_value(training_use)
            self._event(run_id, "cost:" + event_id, "cost_recorded", payload)
            if overrun or self._budget(run_id, unit)["available"] < 0:
                self.connection.execute("UPDATE runs SET status='execution_error' WHERE run_id=?", (run_id,))
                self._event(run_id, "overrun:" + event_id, "budget_overrun", {"reservation_id": reservation_id})
            return True

    def _training_phase(self, run_id: str, attempt_id: str) -> str | None:
        row = self.connection.execute(
            "SELECT payload_json FROM events WHERE run_id=? AND kind='training_attempt' ORDER BY sequence DESC",
            (run_id,),
        ).fetchall()
        for event in row:
            payload = loads(event[0])
            if payload["attempt_id"] == attempt_id:
                return payload["phase"]
        return None

    def check_training_resume(self, run_id: str, attempt_id: str, reservation_id: str,
                              batch_fingerprint: str, completed_steps: int) -> None:
        """只读核对恢复边界。账本多一步或少一步均拒绝，禁止重记不明成本。"""
        integer(completed_steps, "completed_steps", 1)
        with self.transaction(write=False):
            if self._run(run_id)["status"] != "planned" or self._training_phase(run_id, attempt_id) != "started":
                fail("resume", "只有未失败、未完成的原训练尝试可以恢复。")
            reservation = self.connection.execute(
                "SELECT * FROM reservations WHERE run_id=? AND reservation_id=?", (run_id, reservation_id)
            ).fetchone()
            if reservation is None or reservation["closed"] or reservation["unit"] != "training_tokens":
                fail("resume", "原训练预算预留必须仍然开放。")
            start = self.connection.execute("SELECT payload_json FROM events WHERE run_id=? AND event_id=?",
                        (run_id, f"training:{attempt_id}:started")).fetchone()
            if loads(start[0])["batch_fingerprint"] != batch_fingerprint:
                fail("resume", "恢复批次与原尝试不同。")
            batches = [arm for arm in self._frozen_batches(run_id, attempt_id)
                       if arm["fingerprint"] == batch_fingerprint]
            if not batches or completed_steps > len(batches[0]["items"]):
                fail("resume", "检查点步骤数超过冻结批次。")
            expected = [[{"attempt_id": attempt_id, "step": step, "data_id": item["data_id"],
                         "version": item["version"], "mask_version": item["mask_version"]}
                         for step, item in enumerate(batch["items"][:completed_steps], 1)] for batch in batches]
            rows = self.connection.execute("SELECT payload_json FROM events WHERE run_id=? AND kind='cost_recorded' ORDER BY sequence", (run_id,)).fetchall()
            uses = [payload["training_use"] for row in rows if
                    (payload := loads(row[0])).get("training_use", {}).get("attempt_id") == attempt_id]
            if uses not in expected:
                fail("resume", "账本与检查点边界不同。保留原成本，拒绝自动恢复。")

    def training_boundary(self, run_id: str, attempt_id: str, step: int, phase: str,
                          checkpoint_path: str, checkpoint_sha256: str) -> bool:
        """记录暂停或恢复边界。该事件不撤销成本，也不结束训练尝试。"""
        integer(step, "step", 1)
        choice(phase, "phase", {"paused", "resumed"})
        string(checkpoint_path, "checkpoint_path")
        string(checkpoint_sha256, "checkpoint_sha256")
        with self.transaction():
            if self._training_phase(run_id, attempt_id) != "started":
                fail("training", "边界事件必须属于未结束的训练尝试。")
            return self._event(run_id, f"boundary:{attempt_id}:{step}:{phase}", "training_boundary",
                               {"attempt_id": attempt_id, "step": step, "phase": phase,
                                "checkpoint_path": checkpoint_path, "checkpoint_sha256": checkpoint_sha256})

    def training_event(
        self, run_id: str, attempt_id: str, phase: str, *, batch_fingerprint: str,
        result: TrainingResult | None = None, error: str | None = None,
    ) -> bool:
        """记录训练阶段。该接口不执行验收，也不修改已接受模型。"""
        string(attempt_id, "attempt_id")
        choice(phase, "phase", {"started", "completed", "failed"})
        payload = {"attempt_id": attempt_id, "phase": phase, "batch_fingerprint": batch_fingerprint,
                   "result": to_json_value(result), "error": error}
        event_id = f"training:{attempt_id}:{phase}"
        with self.transaction():
            self._run(run_id)
            if self.connection.execute("SELECT 1 FROM events WHERE run_id=? AND event_id=?",
                                       (run_id, event_id)).fetchone():
                return self._event(run_id, event_id, "training_attempt", payload)
            previous = self._training_phase(run_id, attempt_id)
            if phase == "started":
                if previous is not None or self._run(run_id)["status"] != "planned":
                    fail("training", "只能从已保存计划启动新的训练尝试。")
                allowed = {arm["fingerprint"] for arm in self._frozen_batches(run_id, attempt_id)}
                if batch_fingerprint not in allowed:
                    fail("batch_fingerprint", "批次不属于已保存的对照计划。")
            elif previous != "started":
                fail("training", "训练完成或失败前必须已有 started 事件。")
            else:
                start = self.connection.execute("SELECT payload_json FROM events WHERE run_id=? AND event_id=?",
                            (run_id, f"training:{attempt_id}:started")).fetchone()
                if loads(start[0])["batch_fingerprint"] != batch_fingerprint:
                    fail("batch_fingerprint", "不能更改训练尝试的批次。")
            if phase == "completed":
                uses = self.connection.execute("SELECT payload_json FROM events WHERE run_id=? AND kind='cost_recorded'", (run_id,)).fetchall()
                actual = sum(p["amount"] for e in uses if (p := loads(e[0])).get("training_use", {}).get("attempt_id") == attempt_id)
                if result is None or result.actual_supervised_tokens != actual:
                    fail("training.result", "训练结果与实际使用账本不一致。")
                batches = self._frozen_batches(run_id, attempt_id)
                planned = {arm["supervised_tokens"] for arm in batches
                           if arm["fingerprint"] == batch_fingerprint}
                if actual not in planned or self._run(run_id)["status"] == "execution_error":
                    fail("training.result", "训练尚未完成计划监督量，或运行已发生错误。")
                expected_usage = next({item["data_id"]: item["supervised_tokens"] for item in arm["items"]}
                                      for arm in batches
                                      if arm["fingerprint"] == batch_fingerprint)
                actual_usage = {}
                for event in uses:
                    use_payload = loads(event[0])
                    use = use_payload.get("training_use", {})
                    if use.get("attempt_id") == attempt_id:
                        actual_usage[use["data_id"]] = actual_usage.get(use["data_id"], 0) + use_payload["amount"]
                if actual_usage != expected_usage:
                    fail("training.result", "实际数据使用与冻结批次不一致。")
            if phase == "failed":
                string(error, "training.error")
                self.connection.execute("UPDATE runs SET status='execution_error' WHERE run_id=?", (run_id,))
            return self._event(run_id, event_id, "training_attempt", payload)

    def release(self, run_id: str, reservation_id: str) -> bool:
        with self.transaction():
            row = self.connection.execute("SELECT * FROM reservations WHERE run_id=? AND reservation_id=?",
                                          (run_id, reservation_id)).fetchone()
            if row is None:
                fail("reservation_id", "预留不存在。")
            if row["closed"]:
                return False
            self.connection.execute("UPDATE reservations SET remaining=0,closed=1 WHERE run_id=? AND reservation_id=?",
                                    (run_id, reservation_id))
            self._event(run_id, "release:" + reservation_id, "unused_budget_released", {"amount": row["remaining"]})
            return True

    def record_error(self, run_id: str, event_id: str, message: str) -> bool:
        string(event_id, "event_id")
        string(message, "message")
        with self.transaction():
            self._run(run_id)
            inserted = self._event(run_id, "error:" + event_id, "execution_error", {"message": message})
            if inserted:
                self.connection.execute("UPDATE runs SET status='execution_error' WHERE run_id=?", (run_id,))
            return inserted

    def snapshot(self, run_id: str) -> dict[str, Any]:
        """在同一读事务内返回状态、计划、事件和成本。返回值可独立修改。"""
        with self.transaction(write=False):
            row = self._run(run_id)
            contract = loads(row["contract_json"])
            events = self.connection.execute("SELECT event_id,kind,payload_json,created_at FROM events WHERE run_id=? ORDER BY sequence",
                                             (run_id,)).fetchall()
            comparisons = self.connection.execute("SELECT payload_json FROM comparisons WHERE run_id=? ORDER BY comparison_id", (run_id,)).fetchall()
            phases = {}
            for event in events:
                if event["kind"] == "training_attempt":
                    payload = loads(event["payload_json"])
                    phases[payload["attempt_id"]] = payload["phase"]
                elif event["kind"] == "training_boundary":
                    payload = loads(event["payload_json"])
                    phases[payload["attempt_id"]] = "paused" if payload["phase"] == "paused" else "started"
            training_status = "NOT_RUN"
            if phases:
                training_status = "FAILED" if "failed" in phases.values() else (
                    "RUNNING" if "started" in phases.values() else "PAUSED" if "paused" in phases.values() else "COMPLETED")
            return {"schema_version": "1.0", "run_id": run_id, "mode": row["mode"], "status": row["status"],
                    "purpose": contract["purpose"], "snapshot_hash": row["snapshot_hash"],
                    "accepted_model": contract["model"], "training_status": training_status,
                    "validation_level": "structural", "hardware_target": contract["hardware"],
                    "budgets": {unit: self._budget(run_id, unit) for unit in contract["budgets"]},
                    "comparisons": [loads(r[0]) for r in comparisons],
                    "events": [{"event_id": r["event_id"], "kind": r["kind"], "payload": loads(r["payload_json"]),
                                "created_at": r["created_at"]} for r in events]}
