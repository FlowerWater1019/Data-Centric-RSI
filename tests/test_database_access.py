"""数据库打开、版本拒绝与读事务的回归检查。"""

from contextlib import closing
import hashlib
from pathlib import Path
import sqlite3
import tempfile
from unittest.mock import patch

from rsi.state import StateStore
from rsi.validation import ValidationError
from tests.helpers import Fixture


class DatabaseAccessTests(Fixture):
    def setUp(self) -> None:
        super().setUp()
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.db = Path(folder.name) / "state.sqlite3"

    def test_unknown_version_rejected_without_changing_file(self) -> None:
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)")
            connection.execute("INSERT INTO metadata VALUES ('schema_version','999')")
        before = hashlib.sha256(self.db.read_bytes()).hexdigest()
        connection = sqlite3.connect(self.db)
        with patch("rsi.state.sqlite3.connect", return_value=connection):
            with self.assertRaisesRegex(ValidationError, "版本不支持"):
                StateStore(self.db)
        self.assertEqual(hashlib.sha256(self.db.read_bytes()).hexdigest(), before)
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")

    def test_unrelated_database_is_not_adopted(self) -> None:
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("CREATE TABLE external (value TEXT)")
        before = self.db.read_bytes()
        with self.assertRaisesRegex(ValidationError, "元数据"):
            StateStore(self.db)
        self.assertEqual(self.db.read_bytes(), before)

    def test_incomplete_schema_is_not_repaired(self) -> None:
        with StateStore(self.db):
            pass
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute("DROP TABLE costs")
        before = self.db.read_bytes()
        with self.assertRaisesRegex(ValidationError, "结构不完整"):
            StateStore(self.db)
        self.assertEqual(self.db.read_bytes(), before)

    def test_existing_only_mode_does_not_create_database(self) -> None:
        with self.assertRaises(sqlite3.OperationalError):
            StateStore(self.db, create=False)
        self.assertFalse(self.db.exists())

    def test_read_only_mode_does_not_create_parent_directory(self) -> None:
        missing = self.db.parent / "missing" / "state.db"
        with self.assertRaises(sqlite3.OperationalError):
            StateStore(missing, read_only=True)
        self.assertFalse(missing.parent.exists())

    def test_snapshot_does_not_request_writer_lock(self) -> None:
        with StateStore(self.db) as writer:
            writer.initialize("run", "controlled", self.config, self.data)
            with StateStore(self.db, read_only=True) as reader:
                reader.connection.execute("PRAGMA busy_timeout=0")
                with writer.transaction():
                    self.assertEqual(reader.snapshot("run")["status"], "validated")

    def test_read_only_connection_refuses_mutation(self) -> None:
        with StateStore(self.db) as writer:
            writer.initialize("run", "controlled", self.config, self.data)
        with StateStore(self.db, read_only=True) as reader:
            with self.assertRaises(sqlite3.OperationalError):
                reader.reserve("run", "job", "scoring_tokens", 1)
            self.assertEqual(reader.snapshot("run")["budgets"]["scoring_tokens"]["reserved"], 0)

    def test_nested_transaction_does_not_commit_outer_work(self) -> None:
        with StateStore(self.db) as state:
            state.initialize("run", "controlled", self.config, self.data)
            with self.assertRaisesRegex(ValidationError, "嵌套事务"):
                with state.transaction():
                    state.connection.execute("UPDATE runs SET status='execution_error' WHERE run_id='run'")
                    state.reserve("run", "job", "scoring_tokens", 1)
            self.assertEqual(state.snapshot("run")["status"], "validated")
