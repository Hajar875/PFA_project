"""Latin (IRCAM ou informel) → transcription canonique → Tifinagh."""

from __future__ import annotations

import re
import unicodedata

from ..tifinagh import LAT2TFNG, latin_to_tfng

_DOUBLE_VOWEL = re.compile(r"([aiu])\1+")


class LatinConverter:
    def __init__(self, informal: dict[str, str], delete_schwa: bool = True, collapse_vowels: bool = True):
        self.map = informal
        self.lengths = sorted({len(k) for k in informal}, reverse=True)
        self.delete_schwa = delete_schwa
        self.collapse_vowels = collapse_vowels

    def to_canonical(self, token: str) -> str:
        """`Thamazighth` -> `tamaziɣt`. Un token sans lettre (nombre, ponctuation) est laissé tel quel."""
        if not any(ch.isalpha() for ch in token):
            return token
        t = unicodedata.normalize("NFC", token.lower())
        out, i = [], 0
        while i < len(t):
            for n in self.lengths:  # correspondance la plus longue d'abord
                seg = t[i:i + n]
                if len(seg) == n and seg in self.map:
                    out.append(self.map[seg])
                    i += n
                    break
            else:
                out.append(t[i])
                i += 1
        s = "".join(out)
        if self.delete_schwa:
            s = s.replace("e", "")  # l'IRCAM n'écrit pas le schwa
        if self.collapse_vowels:
            s = _DOUBLE_VOWEL.sub(r"\1", s)
        return s

    def to_tifinagh(self, token: str) -> str:
        return latin_to_tfng(self.to_canonical(token))

    @staticmethod
    def naive(token: str) -> str:
        """B0 : lettre par lettre, sans contexte ni digrammes."""
        return "".join(LAT2TFNG.get(c, c) for c in unicodedata.normalize("NFC", token.lower()))
