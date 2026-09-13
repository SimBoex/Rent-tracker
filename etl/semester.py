"""OMI semester parse / sort helpers (shared by transform + ml.split)."""

from __future__ import annotations


def semester_key(sem: str) -> tuple[int, int]:
    """Sort key for 'YYYY-S' or loose strings."""
    parts = str(sem).replace("_", "-").split("-")
    try:
        year = int(parts[0])
        half = int(parts[1]) if len(parts) > 1 else 1
        return year, half
    except ValueError:
        return (0, 0)
