from __future__ import annotations

import json
from pathlib import Path

from .types import DecodedRecord


def write_jsonl(path: str | Path, records: list[DecodedRecord]) -> None:
    out_path = Path(path)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record.to_jsonable(), sort_keys=True))
            fh.write("\n")
