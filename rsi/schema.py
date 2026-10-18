"""SQLite 版本与结构检查。现有数据库不进行隐式修复。"""

import sqlite3

from .validation import fail


SCHEMA_VERSION = "1"
TABLES = {
    "metadata": "key TEXT PRIMARY KEY, value TEXT NOT NULL",
    "runs": """run_id TEXT PRIMARY KEY, mode TEXT NOT NULL, contract_json TEXT NOT NULL,
        dataset_json TEXT NOT NULL, snapshot_hash TEXT NOT NULL,
        status TEXT NOT NULL, created_at TEXT NOT NULL""",
    "events": """sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id TEXT NOT NULL REFERENCES runs(run_id), event_id TEXT NOT NULL,
        kind TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL,
        UNIQUE(run_id,event_id)""",
    "comparisons": """run_id TEXT NOT NULL REFERENCES runs(run_id), comparison_id TEXT NOT NULL,
        payload_json TEXT NOT NULL, PRIMARY KEY(run_id,comparison_id)""",
    "reservations": """run_id TEXT NOT NULL REFERENCES runs(run_id), reservation_id TEXT NOT NULL,
        unit TEXT NOT NULL, original INTEGER NOT NULL, remaining INTEGER NOT NULL,
        closed INTEGER NOT NULL DEFAULT 0, PRIMARY KEY(run_id,reservation_id)""",
    "costs": """run_id TEXT NOT NULL REFERENCES runs(run_id), event_id TEXT NOT NULL,
        reservation_id TEXT NOT NULL, unit TEXT NOT NULL, amount INTEGER NOT NULL,
        PRIMARY KEY(run_id,event_id),
        FOREIGN KEY(run_id,reservation_id) REFERENCES reservations(run_id,reservation_id)""",
}
REQUIRED_COLUMNS = {
    "metadata": {"key", "value"},
    "runs": {"run_id", "mode", "contract_json", "dataset_json", "snapshot_hash", "status", "created_at"},
    "events": {"sequence", "run_id", "event_id", "kind", "payload_json", "created_at"},
    "comparisons": {"run_id", "comparison_id", "payload_json"},
    "reservations": {"run_id", "reservation_id", "unit", "original", "remaining", "closed"},
    "costs": {"run_id", "event_id", "reservation_id", "unit", "amount"},
}


def table_names(connection: sqlite3.Connection) -> set[str]:
    return {row[0] for row in connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
    )}


def validate_schema(connection: sqlite3.Connection) -> None:
    """只读检查版本、表和列。不向不支持的数据库写入任何内容。"""
    names = table_names(connection)
    if "metadata" not in names:
        fail("database", "缺少版本元数据。请使用 init 创建独立数据库。")
    metadata_columns = {row[1] for row in connection.execute('PRAGMA table_info("metadata")')}
    if not REQUIRED_COLUMNS["metadata"] <= metadata_columns:
        fail("database.metadata", "版本表结构不完整。")
    version = connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
    if version is None or version[0] != SCHEMA_VERSION:
        fail("database", "数据库版本不支持。")
    for name, required in REQUIRED_COLUMNS.items():
        # 表名只来自本模块常量，不接收用户输入。
        columns = {row[1] for row in connection.execute(f'PRAGMA table_info("{name}")')}
        if name not in names or not required <= columns:
            fail(f"database.{name}", "数据库结构不完整。未自动补表或补列。")


def ensure_schema(connection: sqlite3.Connection, *, allow_create: bool) -> None:
    """仅在空数据库且允许创建时建表。新库的表与版本在同一事务中提交。"""
    if table_names(connection) or not allow_create:
        validate_schema(connection)
        return
    connection.execute("BEGIN IMMEDIATE")
    try:
        # 另一个连接可能已经在获取写锁前完成建表。
        if table_names(connection):
            validate_schema(connection)
        else:
            for name, definition in TABLES.items():
                connection.execute(f'CREATE TABLE "{name}" ({definition})')
            connection.execute("INSERT INTO metadata VALUES ('schema_version',?)", (SCHEMA_VERSION,))
        connection.execute("COMMIT")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
