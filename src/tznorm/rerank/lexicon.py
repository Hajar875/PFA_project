"""Lexique Tifinagh par variété, indexé par squelette consonantique (recherche à distance 0 ou 1)."""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from ..io import read_tsv
from ..tifinagh import skeleton, tifinagh_words

MIN_SKELETON_FOR_FUZZY = 4  # en dessous, seule la correspondance exacte du squelette est admise


def _keys(skel: str) -> set[str]:
    keys = {skel}
    if len(skel) >= MIN_SKELETON_FOR_FUZZY:
        keys.update(skel[:i] + skel[i + 1:] for i in range(len(skel)))
    return keys


class Lexicon:
    def __init__(self):
        self.entries: dict[str, dict] = {}
        self._index: dict[str, set[str]] = defaultdict(set)

    def add(self, form: str, variety: str = "*", freq: int = 1) -> None:
        form = unicodedata.normalize("NFC", form)
        entry = self.entries.setdefault(form, {"freq": 0, "varieties": set()})
        entry["freq"] += freq
        entry["varieties"].add(variety or "*")
        for key in _keys(skeleton(form)):
            self._index[key].add(form)

    def __len__(self) -> int:
        return len(self.entries)

    def contains(self, word: str) -> bool:
        return word in self.entries

    def has_variety(self, form: str, variety: str) -> bool:
        v = self.entries.get(form, {}).get("varieties", set())
        return "*" in v or variety in v

    def freq(self, form: str) -> int:
        return self.entries.get(form, {}).get("freq", 0)

    def candidates(self, word: str) -> list[str]:
        found: set[str] = set()
        for key in _keys(skeleton(word)):
            found |= self._index.get(key, set())
        return sorted(found)

    def coverage(self, text: str) -> float:
        words = tifinagh_words(text)
        if not words:
            return 0.0
        return sum(w in self.entries for w in words) / len(words)

    @classmethod
    def load(cls, paths: Iterable[str | Path]) -> "Lexicon":
        lex = cls()
        for path in paths:
            for row in read_tsv(path):
                variety = row[1] if len(row) > 1 and row[1] else "*"
                freq = int(row[2]) if len(row) > 2 and row[2].strip().isdigit() else 1
                lex.add(row[0].strip(), variety, freq)
        return lex

    @classmethod
    def from_corpus(cls, lines: Iterable[str], variety: str = "*", min_freq: int = 2) -> "Lexicon":
        counts = Counter(w for line in lines for w in tifinagh_words(unicodedata.normalize("NFC", line)))
        lex = cls()
        for form, n in counts.items():
            if n >= min_freq:
                lex.add(form, variety, n)
        return lex

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("# forme\tvariété\tfréquence\n")
            for form, e in sorted(self.entries.items(), key=lambda kv: -kv[1]["freq"]):
                for v in sorted(e["varieties"]):
                    f.write(f"{form}\t{v}\t{e['freq']}\n")
