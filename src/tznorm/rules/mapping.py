"""Chargement des tables de correspondance (resources/mapping/*.tsv)."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import lru_cache

from ..io import read_tsv, resources_dir


@dataclass(frozen=True)
class Tables:
    latin_informal: dict[str, str]
    arabic: dict[str, str]


def _load(path) -> dict[str, str]:
    table: dict[str, str] = {}
    for row in read_tsv(path):
        src = unicodedata.normalize("NFC", row[0])
        dst = unicodedata.normalize("NFC", row[1]) if len(row) > 1 else ""
        if src:
            table[src] = dst
    return table


@lru_cache(maxsize=8)
def load_tables(resources: str | None = None) -> Tables:
    base = resources_dir(resources) / "mapping"
    return Tables(latin_informal=_load(base / "latin_informal.tsv"), arabic=_load(base / "arabic_latin.tsv"))
