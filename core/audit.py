from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class HashChainAuditLog:
    """Append-only JSONL audit log in which every event hashes the previous event."""

    def __init__(self, path: str | Path = "data/audit_log.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return "0" * 64
        last = self.path.read_text(encoding="utf-8").strip().splitlines()[-1]
        return json.loads(last)["event_hash"]

    @staticmethod
    def _digest(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def append(self, event_type: str, data: dict[str, Any]) -> dict[str, Any]:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "data": data,
            "previous_hash": self._last_hash(),
        }
        event["event_hash"] = self._digest(event)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def verify(self) -> tuple[bool, int]:
        if not self.path.exists():
            return True, 0
        previous = "0" * 64
        count = 0
        for line in self.path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            stored_hash = event.pop("event_hash")
            if event["previous_hash"] != previous or self._digest(event) != stored_hash:
                return False, count
            previous = stored_hash
            count += 1
        return True, count

