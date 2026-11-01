"""Écriture arabe → transcription canonique → Tifinagh.

و et ي valent une voyelle (u, i) ou une semi-voyelle (w, y) selon le contexte ; la chadda double la consonne.
"""

from __future__ import annotations

import re
import unicodedata

from ..tifinagh import latin_to_tfng

SHADDA = "ّ"
VOWEL_CARRIERS = set("اأإآٱى")   # ا أ إ آ ٱ ى
HARAKAT = set("َُِ")                              # fatha, kasra, damma
WAW, YEH = "و", "ي"
ALEF_HAMZA_ABOVE, ALEF_HAMZA_BELOW = "أ", "إ"
_DOUBLE_VOWEL = re.compile(r"([aiu])\1+")
_SPLIT = re.compile(r"(\s+)")


def _is_vowel(ch: str | None) -> bool:
    return ch is not None and (ch in VOWEL_CARRIERS or ch in HARAKAT)


class ArabicConverter:
    def __init__(self, table: dict[str, str]):
        self.table = table

    def word_to_canonical(self, word: str) -> str:
        w = unicodedata.normalize("NFC", word)
        out: list[str] = []
        n, i = len(w), 0
        while i < n:
            c = w[i]
            prev = w[i - 1] if i > 0 else None
            nxt = w[i + 1] if i + 1 < n else None
            initial = i == 0 or not w[i - 1].isalpha()
            if initial and c == ALEF_HAMZA_ABOVE and nxt == WAW:
                after = w[i + 2] if i + 2 < n else None
                out.append("aw" if _is_vowel(after) else "u")  # أوال = awal ; أور = ur
                i += 2
                continue
            if initial and c == ALEF_HAMZA_BELOW and nxt == YEH:
                out.append("i")
                i += 2
                continue
            if c == SHADDA:
                last = next((x[-1] for x in reversed(out) if x), "")
                if last.isalpha() and last not in "aiu":
                    out.append(last)
                i += 1
                continue
            if c in (WAW, YEH):
                semi, vowel = ("w", "u") if c == WAW else ("y", "i")
                if initial:
                    out.append(semi if _is_vowel(nxt) else vowel)
                else:
                    out.append(semi if (_is_vowel(prev) or _is_vowel(nxt) or nxt == SHADDA) else vowel)
                i += 1
                continue
            out.append(self.table.get(c, c))
            i += 1
        return _DOUBLE_VOWEL.sub(r"\1", "".join(out))

    def to_canonical(self, text: str) -> str:
        return "".join(p if p.isspace() else self.word_to_canonical(p) for p in _SPLIT.split(text) if p)

    def to_tifinagh(self, text: str) -> str:
        return latin_to_tfng(self.to_canonical(text))

    def naive(self, text: str) -> str:
        """B0 : lettre par lettre ; و -> ⵓ, ي -> ⵉ, chadda ignorée."""
        simple = {**self.table, WAW: "u", YEH: "i", SHADDA: ""}
        return latin_to_tfng("".join(simple.get(c, c) for c in unicodedata.normalize("NFC", text)))
