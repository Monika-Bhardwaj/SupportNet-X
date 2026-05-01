from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class PromptLogger:
    def __init__(self, log_file: Path) -> None:
        self.log_file = log_file
        self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def append(self, tag: str, content: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {tag}\n{content}\n\n")
