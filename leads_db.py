"""
leads_db.py — SQLite persistence for scraped leads + call tracking.

Stores every lead that passes through the app, deduplicates by place_id
or phone number, and provides persistent call-tracking columns
(status, notes, last_called).
"""

from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_NAME = "leads.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS leads (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    place_id        TEXT,
    phone_digits    TEXT,
    business_name   TEXT NOT NULL,
    address         TEXT DEFAULT '',
    phone           TEXT DEFAULT '',
    website         TEXT DEFAULT '',
    rating          REAL,
    owner_name      TEXT DEFAULT '',
    confidence_score REAL DEFAULT 0,
    num_reviews     INTEGER DEFAULT 0,
    solo            INTEGER DEFAULT 0,
    source          TEXT DEFAULT '',
    search_city     TEXT DEFAULT '',
    search_niche    TEXT DEFAULT '',
    scraped_at      TEXT DEFAULT '',
    -- Call tracking
    status          TEXT DEFAULT 'New',
    notes           TEXT DEFAULT '',
    last_called     TEXT DEFAULT ''
);
"""

_CREATE_INDEXES = [
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_place_id ON leads(place_id) WHERE place_id IS NOT NULL AND place_id != '';",
    "CREATE INDEX IF NOT EXISTS idx_phone_digits ON leads(phone_digits) WHERE phone_digits IS NOT NULL AND phone_digits != '';",
    "CREATE INDEX IF NOT EXISTS idx_city_niche ON leads(search_city, search_niche);",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_phone(phone: str | None) -> str:
    """Strip a phone string to digits only. Returns '' if < 7 digits."""
    if not phone:
        return ""
    digits = re.sub(r"\D", "", str(phone))
    return digits if len(digits) >= 7 else ""


def _extract_place_id(row: dict) -> str:
    """Pull a Google place_id from the row's debug info."""
    debug = row.get("_debug") or {}
    # DataForSEO path: place_id is directly available
    pid = debug.get("place_id") or ""
    if pid:
        return str(pid).strip()
    # Playwright path: try to parse from the place URL
    url = debug.get("place_url") or ""
    if url:
        # Google Maps URLs sometimes contain ChIJ... or 0x... IDs
        m = re.search(r"!1s(0x[0-9a-fA-F]+:0x[0-9a-fA-F]+|ChIJ[A-Za-z0-9_\-]+)", url)
        if m:
            return m.group(1)
    return ""


def _row_to_db_fields(row: dict, city: str, niche: str) -> dict:
    """Convert a scraper result row into DB column values."""
    debug = row.get("_debug") or {}
    return {
        "place_id": _extract_place_id(row),
        "phone_digits": _normalize_phone(row.get("phone")),
        "business_name": str(row.get("business_name") or "").strip(),
        "address": str(row.get("address") or "").strip(),
        "phone": str(row.get("phone") or "").strip(),
        "website": str(row.get("website") or "").strip(),
        "rating": float(row.get("rating") or 0),
        "owner_name": str(row.get("owner_name") or "").strip(),
        "confidence_score": float(row.get("confidence_score") or 0),
        "num_reviews": int(row.get("num_reviews") or 0),
        "solo": 1 if row.get("solo") else 0,
        "source": str(debug.get("source") or ""),
        "search_city": city,
        "search_niche": niche,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_db(db_path: str | None = None) -> sqlite3.Connection:
    """Create the DB and tables if needed. Returns a connection."""
    if db_path is None:
        db_path = str(Path(__file__).with_name(DB_NAME))
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(_CREATE_TABLE)
    for idx_sql in _CREATE_INDEXES:
        conn.execute(idx_sql)
    conn.commit()
    return conn


def _find_existing(conn: sqlite3.Connection, place_id: str, phone_digits: str, name: str, address: str) -> dict | None:
    """Find an existing lead by place_id, phone, or name+address."""
    # Priority 1: place_id
    if place_id:
        row = conn.execute("SELECT * FROM leads WHERE place_id = ?", (place_id,)).fetchone()
        if row:
            return dict(row)
    # Priority 2: phone digits
    if phone_digits:
        row = conn.execute("SELECT * FROM leads WHERE phone_digits = ?", (phone_digits,)).fetchone()
        if row:
            return dict(row)
    # Priority 3: name + address
    if name and address:
        row = conn.execute(
            "SELECT * FROM leads WHERE business_name = ? AND address = ?",
            (name, address),
        ).fetchone()
        if row:
            return dict(row)
    return None


def upsert_leads(conn: sqlite3.Connection, rows: list[dict], city: str, niche: str) -> tuple[int, int]:
    """
    Save leads to DB. Deduplicates by place_id / phone / name+address.
    Updates scrape-related fields but preserves call tracking data.

    Returns (new_count, updated_count).
    Each row dict gets a '_db_id' key added in-place.
    """
    new_count = 0
    updated_count = 0

    for row in rows:
        fields = _row_to_db_fields(row, city, niche)
        existing = _find_existing(
            conn,
            fields["place_id"],
            fields["phone_digits"],
            fields["business_name"],
            fields["address"],
        )

        if existing:
            # Update scrape data, keep call tracking intact
            conn.execute(
                """UPDATE leads SET
                    owner_name = ?, confidence_score = ?, solo = ?,
                    num_reviews = ?, rating = ?, source = ?,
                    website = ?, scraped_at = ?
                WHERE id = ?""",
                (
                    fields["owner_name"],
                    fields["confidence_score"],
                    fields["solo"],
                    fields["num_reviews"],
                    fields["rating"],
                    fields["source"],
                    fields["website"],
                    fields["scraped_at"],
                    existing["id"],
                ),
            )
            row["_db_id"] = existing["id"]
            row["_db_status"] = existing["status"] or "New"
            row["_db_notes"] = existing["notes"] or ""
            row["_db_last_called"] = existing["last_called"] or ""
            updated_count += 1
        else:
            cur = conn.execute(
                """INSERT INTO leads (
                    place_id, phone_digits, business_name, address, phone,
                    website, rating, owner_name, confidence_score, num_reviews,
                    solo, source, search_city, search_niche, scraped_at,
                    status, notes, last_called
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'New', '', '')""",
                (
                    fields["place_id"],
                    fields["phone_digits"],
                    fields["business_name"],
                    fields["address"],
                    fields["phone"],
                    fields["website"],
                    fields["rating"],
                    fields["owner_name"],
                    fields["confidence_score"],
                    fields["num_reviews"],
                    fields["solo"],
                    fields["source"],
                    fields["search_city"],
                    fields["search_niche"],
                    fields["scraped_at"],
                ),
            )
            row["_db_id"] = cur.lastrowid
            row["_db_status"] = "New"
            row["_db_notes"] = ""
            row["_db_last_called"] = ""
            new_count += 1

    conn.commit()
    return new_count, updated_count


def update_lead_tracking(conn: sqlite3.Connection, lead_id: int, status: str, notes: str) -> None:
    """Update call tracking fields for a lead."""
    last_called = ""
    if status and status != "New":
        last_called = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE leads SET status = ?, notes = ?, last_called = ? WHERE id = ?",
        (status, notes, last_called, lead_id),
    )
    conn.commit()


def get_all_leads(conn: sqlite3.Connection) -> list[dict]:
    """Return all leads from the DB, most recent first."""
    rows = conn.execute(
        "SELECT * FROM leads ORDER BY scraped_at DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def get_leads_for_search(conn: sqlite3.Connection, city: str, niche: str) -> list[dict]:
    """Return leads matching a specific city+niche search."""
    rows = conn.execute(
        "SELECT * FROM leads WHERE search_city = ? AND search_niche = ? ORDER BY confidence_score DESC",
        (city, niche),
    ).fetchall()
    return [dict(r) for r in rows]


def get_lead_stats(conn: sqlite3.Connection) -> dict:
    """Return summary stats for the sidebar."""
    total = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
    qualified = conn.execute(
        "SELECT COUNT(*) FROM leads WHERE solo = 1 AND confidence_score > 0.7"
    ).fetchone()[0]
    called = conn.execute(
        "SELECT COUNT(*) FROM leads WHERE status != 'New' AND status != ''"
    ).fetchone()[0]
    interested = conn.execute(
        "SELECT COUNT(*) FROM leads WHERE status = 'Interested'"
    ).fetchone()[0]
    return {
        "total": total,
        "qualified": qualified,
        "called": called,
        "interested": interested,
    }
