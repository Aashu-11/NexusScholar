"""
metadata_store.py — SQLite interface for all structured data.
Papers, chunks, citation edges, conversations, evidence tables, answers.
"""

from __future__ import annotations
import json
import aiosqlite
from pathlib import Path
from typing import Optional

from backend.config import settings


class MetadataStore:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or settings.DB_PATH

    async def _conn(self) -> aiosqlite.Connection:
        db = await aiosqlite.connect(self.db_path)
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA journal_mode=WAL")
        await db.execute("PRAGMA foreign_keys=ON")
        return db

    async def init_db(self):
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        db = await self._conn()
        try:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors TEXT DEFAULT '[]',
                    year INTEGER,
                    venue TEXT,
                    doi TEXT,
                    arxiv_id TEXT,
                    openalex_id TEXT,
                    abstract TEXT DEFAULT '',
                    sections TEXT DEFAULT '{}',
                    references_json TEXT DEFAULT '[]',
                    is_peer_reviewed INTEGER DEFAULT 0,
                    is_retracted INTEGER DEFAULT 0,
                    citation_count INTEGER DEFAULT 0,
                    source_url TEXT DEFAULT '',
                    pdf_url TEXT DEFAULT '',
                    pdf_path TEXT,
                    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    paper_id TEXT NOT NULL,
                    granularity TEXT NOT NULL,
                    section_tag TEXT DEFAULT 'unknown',
                    text TEXT NOT NULL,
                    token_count INTEGER DEFAULT 0,
                    embedding BLOB,
                    start_char INTEGER DEFAULT 0,
                    end_char INTEGER DEFAULT 0,
                    FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_gran ON chunks(granularity);

                CREATE TABLE IF NOT EXISTS citation_edges (
                    source_paper_id TEXT NOT NULL,
                    target_paper_id TEXT NOT NULL,
                    PRIMARY KEY (source_paper_id, target_paper_id)
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT DEFAULT 'New conversation',
                    corpus_id TEXT DEFAULT 'default',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    answer_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
                );

                CREATE TABLE IF NOT EXISTS evidence_tables (
                    answer_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    rows_json TEXT DEFAULT '[]',
                    confidence_score REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS answers (
                    answer_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    intent TEXT NOT NULL,
                    markdown_text TEXT DEFAULT '',
                    citations_json TEXT DEFAULT '[]',
                    is_abstention INTEGER DEFAULT 0,
                    abstention_reason TEXT,
                    uncertainty_flags TEXT DEFAULT '[]',
                    total_sources INTEGER DEFAULT 0,
                    peer_reviewed_count INTEGER DEFAULT 0,
                    preprint_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)
            await self._ensure_column(db, "papers", "openalex_id", "TEXT")
            await self._ensure_column(db, "papers", "source_url", "TEXT")
            await self._ensure_column(db, "papers", "pdf_url", "TEXT")
            await db.commit()

            # Migration: add columns if upgrading from older schema
            for col, col_type, default in [
                ("source_url", "TEXT", "''"),
                ("pdf_url", "TEXT", "''"),
                ("openalex_id", "TEXT", "NULL"),
            ]:
                try:
                    await db.execute(
                        f"ALTER TABLE papers ADD COLUMN {col} {col_type} DEFAULT {default}"
                    )
                    await db.commit()
                except Exception:
                    pass  # column already exists
        finally:
            await db.close()

    async def _ensure_column(self, db: aiosqlite.Connection, table: str, column: str, column_type: str):
        cur = await db.execute(f"PRAGMA table_info({table})")
        columns = {row[1] for row in await cur.fetchall()}
        if column not in columns:
            await db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    # ── Papers ────────────────────────────────────────────

    async def insert_paper(self, paper: dict):
        db = await self._conn()
        try:
            await db.execute(
                """INSERT OR REPLACE INTO papers
                   (paper_id, title, authors, year, venue, doi, arxiv_id,
                    openalex_id, abstract, sections, references_json, is_peer_reviewed,
                    is_retracted, citation_count, source_url, pdf_url, pdf_path)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (paper["paper_id"], paper["title"],
                 json.dumps(paper.get("authors", [])),
                 paper.get("year"), paper.get("venue"),
                 paper.get("doi"), paper.get("arxiv_id"),
                 paper.get("openalex_id"),
                 paper.get("abstract", ""),
                 json.dumps(paper.get("sections", {})),
                 json.dumps(paper.get("references", [])),
                 int(paper.get("is_peer_reviewed", False)),
                 int(paper.get("is_retracted", False)),
                 paper.get("citation_count", 0),
                 paper.get("source_url", ""),
                 paper.get("pdf_url", ""),
                 paper.get("pdf_path"))
            )
            await db.commit()
        finally:
            await db.close()

    async def get_paper(self, paper_id: str) -> Optional[dict]:
        db = await self._conn()
        try:
            cur = await db.execute("SELECT * FROM papers WHERE paper_id = ?", (paper_id,))
            row = await cur.fetchone()
            return self._paper_row_to_dict(row) if row else None
        finally:
            await db.close()

    async def get_papers_batch(self, paper_ids: list[str]) -> dict[str, dict]:
        """
        Fetch multiple papers in a single SQL query.
        Returns a dict mapping paper_id → paper_dict.
        Missing paper_ids are simply absent from the result.
        """
        if not paper_ids:
            return {}
        # Deduplicate while preserving order isn't required here
        unique_ids = list(dict.fromkeys(paper_ids))
        placeholders = ",".join("?" * len(unique_ids))
        db = await self._conn()
        try:
            cur = await db.execute(
                f"SELECT * FROM papers WHERE paper_id IN ({placeholders})",
                unique_ids,
            )
            rows = await cur.fetchall()
            return {row["paper_id"]: self._paper_row_to_dict(row) for row in rows}
        finally:
            await db.close()

    async def get_all_papers(self) -> list[dict]:
        db = await self._conn()
        try:
            cur = await db.execute("SELECT * FROM papers ORDER BY ingested_at DESC")
            rows = await cur.fetchall()
            return [self._paper_row_to_dict(r) for r in rows]
        finally:
            await db.close()

    async def search_papers(self, query: str, limit: int = 20,
                            year_min: int = None, year_max: int = None,
                            peer_reviewed_only: bool = False, venue: str = None) -> list[dict]:
        db = await self._conn()
        try:
            sql = "SELECT * FROM papers WHERE (title LIKE ? OR abstract LIKE ?)"
            params: list = [f"%{query}%", f"%{query}%"]
            if year_min:
                sql += " AND year >= ?"
                params.append(year_min)
            if year_max:
                sql += " AND year <= ?"
                params.append(year_max)
            if peer_reviewed_only:
                sql += " AND is_peer_reviewed = 1"
            if venue:
                sql += " AND venue LIKE ?"
                params.append(f"%{venue}%")
            sql += f" ORDER BY citation_count DESC LIMIT {limit}"
            cur = await db.execute(sql, params)
            rows = await cur.fetchall()
            return [self._paper_row_to_dict(r) for r in rows]
        finally:
            await db.close()

    async def search_papers_by_title(self, title: str, limit: int = 5) -> list[dict]:
        db = await self._conn()
        try:
            cur = await db.execute(
                "SELECT * FROM papers WHERE title LIKE ? LIMIT ?",
                (f"%{title[:80]}%", limit)
            )
            rows = await cur.fetchall()
            return [self._paper_row_to_dict(r) for r in rows]
        finally:
            await db.close()

    async def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[dict]:
        db = await self._conn()
        try:
            cur = await db.execute(
                "SELECT * FROM papers WHERE arxiv_id = ? LIMIT 1",
                (arxiv_id,),
            )
            row = await cur.fetchone()
            return self._paper_row_to_dict(row) if row else None
        finally:
            await db.close()

    async def get_paper_by_openalex_id(self, openalex_id: str) -> Optional[dict]:
        db = await self._conn()
        try:
            cur = await db.execute(
                "SELECT * FROM papers WHERE openalex_id = ? LIMIT 1",
                (openalex_id,),
            )
            row = await cur.fetchone()
            return self._paper_row_to_dict(row) if row else None
        finally:
            await db.close()

    async def update_paper_metadata(self, paper_id: str, updates: dict):
        """Update specific fields of an existing paper (e.g. from OpenAlex enrichment)."""
        if not updates:
            return
        db = await self._conn()
        try:
            allowed_fields = {
                "doi", "venue", "year", "citation_count", "is_peer_reviewed",
                "is_retracted", "source_url", "pdf_url", "openalex_id", "abstract",
            }
            set_clauses = []
            params = []
            for key, value in updates.items():
                if key in allowed_fields:
                    set_clauses.append(f"{key} = ?")
                    if key in ("is_peer_reviewed", "is_retracted"):
                        params.append(int(value))
                    else:
                        params.append(value)
            if not set_clauses:
                return
            params.append(paper_id)
            sql = f"UPDATE papers SET {', '.join(set_clauses)} WHERE paper_id = ?"
            await db.execute(sql, params)
            await db.commit()
        finally:
            await db.close()

    def _paper_row_to_dict(self, row) -> dict:
        return {
            "paper_id": row["paper_id"], "title": row["title"],
            "authors": json.loads(row["authors"]), "year": row["year"],
            "venue": row["venue"], "doi": row["doi"], "arxiv_id": row["arxiv_id"], "openalex_id": row["openalex_id"],
            "abstract": row["abstract"], "sections": json.loads(row["sections"]),
            "references": json.loads(row["references_json"]),
            "is_peer_reviewed": bool(row["is_peer_reviewed"]),
            "is_retracted": bool(row["is_retracted"]),
            "citation_count": row["citation_count"],
            "source_url": row["source_url"] if "source_url" in row.keys() else "",
            "pdf_url": row["pdf_url"] if "pdf_url" in row.keys() else "",
            "pdf_path": row["pdf_path"],
            "openalex_id": row["openalex_id"] if "openalex_id" in row.keys() else None,
        }

    # ── Chunks ────────────────────────────────────────────

    async def insert_chunks(self, chunks: list[dict]):
        db = await self._conn()
        try:
            await db.executemany(
                """INSERT OR REPLACE INTO chunks
                   (chunk_id, paper_id, granularity, section_tag, text,
                    token_count, embedding, start_char, end_char)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                [(c["chunk_id"], c["paper_id"], c["granularity"],
                  c.get("section_tag", "unknown"), c["text"],
                  c.get("token_count", 0),
                  json.dumps(c["embedding"]) if c.get("embedding") else None,
                  c.get("start_char", 0), c.get("end_char", 0))
                 for c in chunks]
            )
            await db.commit()
        finally:
            await db.close()

    async def get_all_chunks(self, granularity: str = None) -> list[dict]:
        db = await self._conn()
        try:
            sql = "SELECT * FROM chunks"
            params = []
            if granularity:
                sql += " WHERE granularity = ?"
                params.append(granularity)
            cur = await db.execute(sql, params)
            rows = await cur.fetchall()
            return [self._chunk_row_to_dict(r) for r in rows]
        finally:
            await db.close()

    async def get_chunks_by_paper(self, paper_id: str, granularity: str = None) -> list[dict]:
        db = await self._conn()
        try:
            sql = "SELECT * FROM chunks WHERE paper_id = ?"
            params = [paper_id]
            if granularity:
                sql += " AND granularity = ?"
                params.append(granularity)
            cur = await db.execute(sql, params)
            rows = await cur.fetchall()
            return [self._chunk_row_to_dict(r) for r in rows]
        finally:
            await db.close()

    def _chunk_row_to_dict(self, row) -> dict:
        return {
            "chunk_id": row["chunk_id"], "paper_id": row["paper_id"],
            "granularity": row["granularity"], "section_tag": row["section_tag"],
            "text": row["text"], "token_count": row["token_count"],
            "embedding": json.loads(row["embedding"]) if row["embedding"] else None,
            "start_char": row["start_char"], "end_char": row["end_char"],
        }

    # ── Citation Edges ────────────────────────────────────

    async def insert_citation_edge(self, source_id: str, target_id: str):
        db = await self._conn()
        try:
            await db.execute(
                "INSERT OR IGNORE INTO citation_edges VALUES (?, ?)",
                (source_id, target_id)
            )
            await db.commit()
        finally:
            await db.close()

    async def get_all_citation_edges(self) -> list[tuple[str, str]]:
        db = await self._conn()
        try:
            cur = await db.execute("SELECT source_paper_id, target_paper_id FROM citation_edges")
            return [(r[0], r[1]) for r in await cur.fetchall()]
        finally:
            await db.close()

    async def get_citation_neighbors(self, paper_id: str) -> dict:
        db = await self._conn()
        try:
            c1 = await db.execute(
                "SELECT source_paper_id FROM citation_edges WHERE target_paper_id = ?", (paper_id,))
            c2 = await db.execute(
                "SELECT target_paper_id FROM citation_edges WHERE source_paper_id = ?", (paper_id,))
            return {
                "cited_by": [r[0] for r in await c1.fetchall()],
                "cites": [r[0] for r in await c2.fetchall()],
            }
        finally:
            await db.close()

    # ── Conversations & Messages ──────────────────────────

    async def upsert_conversation(self, conv_id: str, title: str = "New conversation", corpus_id: str = "default"):
        db = await self._conn()
        try:
            await db.execute(
                """INSERT OR REPLACE INTO conversations (conversation_id, title, corpus_id, updated_at)
                   VALUES (?, ?, ?, CURRENT_TIMESTAMP)""", (conv_id, title, corpus_id))
            await db.commit()
        finally:
            await db.close()

    async def insert_message(self, msg: dict):
        db = await self._conn()
        try:
            await db.execute(
                """INSERT INTO messages (message_id, conversation_id, role, content, answer_id)
                   VALUES (?,?,?,?,?)""",
                (msg["message_id"], msg["conversation_id"], msg["role"],
                 msg["content"], msg.get("answer_id")))
            await db.commit()
        finally:
            await db.close()

    async def get_conversations(self) -> list[dict]:
        db = await self._conn()
        try:
            cur = await db.execute("SELECT * FROM conversations ORDER BY updated_at DESC")
            return [dict(r) for r in await cur.fetchall()]
        finally:
            await db.close()

    async def get_messages(self, conv_id: str) -> list[dict]:
        db = await self._conn()
        try:
            cur = await db.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC", (conv_id,))
            return [dict(r) for r in await cur.fetchall()]
        finally:
            await db.close()

    # ── Evidence & Answers ────────────────────────────────

    async def save_evidence_table(self, et: dict):
        db = await self._conn()
        try:
            await db.execute(
                """INSERT OR REPLACE INTO evidence_tables
                   (answer_id, query, intent, rows_json, confidence_score)
                   VALUES (?,?,?,?,?)""",
                (et["answer_id"], et["query"], et["intent"],
                 json.dumps(et.get("rows", [])), et.get("confidence_score", 0)))
            await db.commit()
        finally:
            await db.close()

    async def get_evidence_table(self, answer_id: str) -> Optional[dict]:
        db = await self._conn()
        try:
            cur = await db.execute("SELECT * FROM evidence_tables WHERE answer_id = ?", (answer_id,))
            row = await cur.fetchone()
            if not row:
                return None
            return {
                "answer_id": row["answer_id"], "query": row["query"],
                "intent": row["intent"], "rows": json.loads(row["rows_json"]),
                "confidence_score": row["confidence_score"],
            }
        finally:
            await db.close()

    async def save_answer(self, answer: dict):
        db = await self._conn()
        try:
            await db.execute(
                """INSERT OR REPLACE INTO answers
                   (answer_id, query, intent, markdown_text, citations_json,
                    is_abstention, abstention_reason, uncertainty_flags,
                    total_sources, peer_reviewed_count, preprint_count)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (answer["answer_id"], answer["query"], answer["intent"],
                 answer.get("markdown_text", ""),
                 json.dumps(answer.get("citations", [])),
                 int(answer.get("is_abstention", False)),
                 answer.get("abstention_reason"),
                 json.dumps(answer.get("uncertainty_flags", [])),
                 answer.get("total_sources", 0),
                 answer.get("peer_reviewed_count", 0),
                 answer.get("preprint_count", 0)))
            await db.commit()
        finally:
            await db.close()
