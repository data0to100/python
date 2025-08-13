from typing import Any, Dict, List


def validate_records(records: List[Dict[str, Any]]) -> None:
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("'data' must be a non-empty list of records")
    if not all(isinstance(r, dict) for r in records):
        raise ValueError("Each record in 'data' must be an object/dict")