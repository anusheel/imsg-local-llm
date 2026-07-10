"""
Extract iMessage history from the local chat.db into JSONL.

Handles the parts a plain `SELECT text FROM message` gets wrong:
  * attributedBody: since macOS Ventura most message text lives in a binary
    NSAttributedString typedstream blob, not the `text` column.
  * date is nanoseconds since the 2001-01-01 Apple epoch.
  * tapbacks/reactions (associated_message_type != 0) are skipped.
  * attachment placeholders (U+FFFC) are stripped.
  * contacts are aliased to Friend 1, Friend 2, ... unless a contacts.csv maps
    handle -> name.

Usage:
    python -m imsg_local_llm.extract [--db PATH] [--out PATH] [--contacts PATH]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Apple's reference date: 2001-01-01 00:00:00 UTC
APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)
OBJ_REPLACEMENT = "￼"  # placeholder iMessage uses for attachments


def decode_attributed_body(blob: bytes | None) -> str | None:
    """Extract the message text from an NSAttributedString typedstream blob.

    The blob is an Apple `streamtyped` archive. The visible message text is the
    first NSString inside it, stored as a length-prefixed byte string introduced
    by the typedstream code 0x2b ('+') that follows the "NSString" class marker.

    Length uses typedstream's variable-length unsigned int encoding:
        b < 0x81            -> length = b
        b == 0x81           -> next 2 bytes, little-endian
        b == 0x82           -> next 4 bytes, little-endian
        b == 0x83           -> next 8 bytes, little-endian
    """
    if not blob:
        return None
    marker = b"NSString"
    i = blob.find(marker)
    if i == -1:
        return None
    i += len(marker)
    # The inline string is introduced by 0x2b ('+') a few bytes after the class
    # marker (typical preamble: 01 94 84 01 2b). Scan a small window for it.
    plus = blob.find(b"+", i, i + 16)
    if plus == -1:
        return None
    j = plus + 1
    if j >= len(blob):
        return None
    n = blob[j]
    j += 1
    try:
        if n == 0x81:
            n = int.from_bytes(blob[j : j + 2], "little"); j += 2
        elif n == 0x82:
            n = int.from_bytes(blob[j : j + 4], "little"); j += 4
        elif n == 0x83:
            n = int.from_bytes(blob[j : j + 8], "little"); j += 8
        elif n >= 0x80:
            # Unexpected marker; bail rather than emit garbage.
            return None
        text = blob[j : j + n].decode("utf-8", errors="replace")
    except Exception:
        return None
    text = text.replace(OBJ_REPLACEMENT, "").strip()
    return text or None


def apple_ns_to_iso(date_ns: int | None) -> str | None:
    """Convert Apple nanosecond epoch to an ISO-8601 UTC string."""
    if not date_ns:
        return None
    try:
        dt = APPLE_EPOCH + timedelta(seconds=date_ns / 1_000_000_000)
        return dt.isoformat()
    except Exception:
        return None


def load_contacts(path: Path) -> dict[str, str]:
    """Optional handle -> display name mapping from a CSV (handle,name)."""
    mapping: dict[str, str] = {}
    if not path.exists():
        return mapping
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if len(row) >= 2 and row[0].strip() and not row[0].strip().startswith("#"):
                mapping[row[0].strip()] = row[1].strip()
    return mapping


QUERY = """
SELECT
    m.ROWID                     AS rowid,
    m.text                      AS text,
    m.attributedBody            AS attributed_body,
    m.is_from_me                AS is_from_me,
    m.date                      AS date_ns,
    h.id                        AS handle,
    c.ROWID                     AS chat_id,
    c.style                     AS chat_style,      -- 43 = group, 45 = 1:1
    c.chat_identifier           AS chat_identifier,
    c.display_name              AS chat_display_name
FROM message m
LEFT JOIN handle h              ON m.handle_id = h.ROWID
LEFT JOIN chat_message_join cmj ON m.ROWID = cmj.message_id
LEFT JOIN chat c                ON cmj.chat_id = c.ROWID
WHERE (m.associated_message_type IS NULL OR m.associated_message_type = 0)
ORDER BY c.ROWID, m.date;
"""


def extract(db_path: Path, out_path: Path, contacts_path: Path) -> dict:
    if not db_path.exists():
        sys.exit(
            f"chat.db not found at {db_path}\n"
            "Give your terminal Full Disk Access: System Settings > Privacy & "
            "Security > Full Disk Access."
        )

    contacts = load_contacts(contacts_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Open read-only so we never touch your live database.
    uri = f"file:{db_path}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row

    alias_for: dict[str, str] = {}

    def alias(handle: str | None) -> str:
        if not handle:
            return "Unknown"
        if handle in contacts:
            return contacts[handle]
        if handle not in alias_for:
            alias_for[handle] = f"Friend {len(alias_for) + 1}"
        return alias_for[handle]

    stats = {
        "rows": 0,
        "written": 0,
        "from_me": 0,
        "empty_after_decode": 0,
        "used_text_col": 0,
        "used_attributed_body": 0,
    }

    with out_path.open("w", encoding="utf-8") as out:
        for row in conn.execute(QUERY):
            stats["rows"] += 1

            text = (row["text"] or "").strip()
            if text:
                stats["used_text_col"] += 1
            else:
                text = decode_attributed_body(row["attributed_body"]) or ""
                if text:
                    stats["used_attributed_body"] += 1
            text = text.replace(OBJ_REPLACEMENT, "").strip()
            if not text:
                stats["empty_after_decode"] += 1
                continue

            is_from_me = bool(row["is_from_me"])
            sender = "Me" if is_from_me else alias(row["handle"])
            record = {
                "rowid": row["rowid"],
                "chat_id": row["chat_id"],
                "is_group": row["chat_style"] == 43,
                "chat_name": row["chat_display_name"] or row["chat_identifier"],
                "timestamp": apple_ns_to_iso(row["date_ns"]),
                "is_from_me": is_from_me,
                "sender": sender,
                "text": text,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written"] += 1
            if is_from_me:
                stats["from_me"] += 1

    conn.close()
    # Persist the alias map so downstream steps / you can read it (git-ignored).
    alias_out = out_path.parent / "aliases.json"
    alias_out.write_text(json.dumps(alias_for, indent=2, ensure_ascii=False), encoding="utf-8")
    stats["distinct_contacts"] = len(alias_for)
    return stats


def main() -> None:
    default_db = Path(os.path.expanduser("~/Library/Messages/chat.db"))
    ap = argparse.ArgumentParser(description="Extract iMessage history to JSONL (local-only).")
    ap.add_argument("--db", type=Path, default=default_db, help="Path to chat.db")
    ap.add_argument("--out", type=Path, default=Path("data/raw/messages.jsonl"))
    ap.add_argument("--contacts", type=Path, default=Path("contacts.csv"),
                    help="Optional CSV: handle,name (git-ignored)")
    args = ap.parse_args()

    stats = extract(args.db, args.out, args.contacts)

    print("iMessage extraction complete (all local):")
    print(f"  rows scanned .............. {stats['rows']:,}")
    print(f"  messages written .......... {stats['written']:,}")
    print(f"    from the `text` column .. {stats['used_text_col']:,}")
    print(f"    decoded attributedBody .. {stats['used_attributed_body']:,}")
    print(f"  from me ................... {stats['from_me']:,}")
    print(f"  empty/undecodable (skipped) {stats['empty_after_decode']:,}")
    print(f"  distinct contacts ......... {stats['distinct_contacts']:,}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
