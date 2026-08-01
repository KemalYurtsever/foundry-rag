from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path


SCHEMA_VERSION = 1
_SQLITE_VARIABLE_LIMIT = 900


class SQLiteStore:
    """Transactional persistence for extracted document sections and embeddings."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA busy_timeout = 30000")
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as connection:
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, SCHEMA_VERSION}:
                raise RuntimeError(
                    f"Unsupported RAG database schema {version}; expected {SCHEMA_VERSION}"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS document_files (
                    collection TEXT NOT NULL,
                    source TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    PRIMARY KEY (collection, source)
                );

                CREATE TABLE IF NOT EXISTS document_sections (
                    collection TEXT NOT NULL,
                    source TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    page INTEGER,
                    text TEXT NOT NULL,
                    PRIMARY KEY (collection, source, position),
                    FOREIGN KEY (collection, source)
                        REFERENCES document_files(collection, source) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS embedding_models (
                    model_alias TEXT PRIMARY KEY,
                    dimension INTEGER NOT NULL CHECK (dimension > 0)
                );

                CREATE TABLE IF NOT EXISTS embeddings (
                    model_alias TEXT NOT NULL,
                    text_hash TEXT NOT NULL,
                    vector_json TEXT NOT NULL,
                    PRIMARY KEY (model_alias, text_hash),
                    FOREIGN KEY (model_alias)
                        REFERENCES embedding_models(model_alias) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_document_sections_source
                    ON document_sections(collection, source);
                """
            )
            # SQLite does not support parameters in PRAGMA assignments. Keep
            # this statement fully static; no runtime or user value enters SQL.
            connection.execute("PRAGMA user_version = 1")

    def get_document_sections(
        self, collection: str, source: str, sha256: str
    ) -> list[tuple[str, int | None]] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT sha256 FROM document_files WHERE collection = ? AND source = ?",
                (collection, source),
            ).fetchone()
            if row is None or row[0] != sha256:
                return None
            rows = connection.execute(
                """
                SELECT text, page FROM document_sections
                WHERE collection = ? AND source = ? ORDER BY position
                """,
                (collection, source),
            ).fetchall()
            return [(str(text), int(page) if page is not None else None) for text, page in rows]

    def put_document_sections(
        self,
        collection: str,
        source: str,
        sha256: str,
        sections: Sequence[tuple[str, int | None]],
    ) -> None:
        rows = [
            (collection, source, position, page, text)
            for position, (text, page) in enumerate(sections)
        ]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO document_files(collection, source, sha256) VALUES (?, ?, ?)
                ON CONFLICT(collection, source) DO UPDATE SET sha256 = excluded.sha256
                """,
                (collection, source, sha256),
            )
            connection.execute(
                "DELETE FROM document_sections WHERE collection = ? AND source = ?",
                (collection, source),
            )
            connection.executemany(
                """
                INSERT INTO document_sections(collection, source, position, page, text)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

    def prune_documents(self, collection: str, active_sources: set[str]) -> None:
        with self._connect() as connection:
            existing = {
                str(row[0])
                for row in connection.execute(
                    "SELECT source FROM document_files WHERE collection = ?", (collection,)
                )
            }
            stale = sorted(existing - active_sources)
            connection.executemany(
                "DELETE FROM document_files WHERE collection = ? AND source = ?",
                [(collection, source) for source in stale],
            )

    def get_model_dimension(self, model_key: str) -> int | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT dimension FROM embedding_models WHERE model_alias = ?", (model_key,)
            ).fetchone()
            return int(row[0]) if row else None

    def get_embeddings(self, model_key: str, text_hashes: list[str]) -> dict[str, list[float]]:
        unique_hashes = list(dict.fromkeys(text_hashes))
        if not unique_hashes:
            return {}

        found: dict[str, list[float]] = {}
        with self._connect() as connection:
            for start in range(0, len(unique_hashes), _SQLITE_VARIABLE_LIMIT):
                batch = unique_hashes[start : start + _SQLITE_VARIABLE_LIMIT]
                # Only parameter markers are generated here. Every value,
                # including model aliases and hashes, remains a bound parameter.
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    f"""
                    SELECT text_hash, vector_json FROM embeddings
                    WHERE model_alias = ? AND text_hash IN ({placeholders})
                    """,
                    [model_key, *batch],
                ).fetchall()
                for text_hash, vector_json in rows:
                    try:
                        vector = [float(value) for value in json.loads(vector_json)]
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        raise RuntimeError(
                            f"Embedding cache entry is corrupt for hash {text_hash}"
                        ) from exc
                    if not vector or not all(math.isfinite(value) for value in vector):
                        raise RuntimeError(
                            f"Embedding cache entry is invalid for hash {text_hash}"
                        )
                    found[str(text_hash)] = vector
        return found

    def put_embeddings(
        self, model_key: str, dimension: int, vectors: dict[str, list[float]]
    ) -> None:
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        if any(
            len(vector) != dimension or not all(math.isfinite(value) for value in vector)
            for vector in vectors.values()
        ):
            raise ValueError("Cannot store invalid or inconsistently sized embeddings")

        with self._connect() as connection:
            existing = connection.execute(
                "SELECT dimension FROM embedding_models WHERE model_alias = ?", (model_key,)
            ).fetchone()
            if existing is not None and int(existing[0]) != dimension:
                raise ValueError(
                    f"Embedding dimension changed for '{model_key}': "
                    f"{existing[0]} -> {dimension}"
                )
            connection.execute(
                "INSERT OR IGNORE INTO embedding_models(model_alias, dimension) VALUES (?, ?)",
                (model_key, dimension),
            )
            connection.executemany(
                """
                INSERT INTO embeddings(model_alias, text_hash, vector_json) VALUES (?, ?, ?)
                ON CONFLICT(model_alias, text_hash)
                DO UPDATE SET vector_json = excluded.vector_json
                """,
                [
                    (
                        model_key,
                        text_hash,
                        json.dumps(vector, separators=(",", ":"), allow_nan=False),
                    )
                    for text_hash, vector in vectors.items()
                ],
            )

    def counts(self) -> dict[str, int]:
        with self._connect() as connection:
            return {
                "documents": int(
                    connection.execute("SELECT COUNT(*) FROM document_files").fetchone()[0]
                ),
                "sections": int(
                    connection.execute("SELECT COUNT(*) FROM document_sections").fetchone()[0]
                ),
                "embeddings": int(
                    connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
                ),
            }
