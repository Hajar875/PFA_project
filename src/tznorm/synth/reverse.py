"""Translittération inverse : Tifinagh propre → latin (IRCAM ou « réseaux sociaux ») ou écriture arabe."""

from __future__ import annotations

import random
import re

from ..tifinagh import latin_to_tfng, tfng_to_latin

SHADDA = "ّ"
_SPLIT = re.compile(r"(\s+)")

DEFAULT_LATIN_SOCIAL: dict[str, dict[str, float]] = {
    "ɣ": {"gh": 0.75, "ɣ": 0.10, "rh": 0.10, "g": 0.05},
    "x": {"kh": 0.60, "x": 0.35, "5": 0.05},
    "c": {"ch": 0.65, "sh": 0.25, "c": 0.10},
    "ɛ": {"3": 0.45, "ɛ": 0.20, "a": 0.20, "": 0.15},
    "ḥ": {"h": 0.60, "7": 0.30, "ḥ": 0.10},
    "q": {"q": 0.85, "9": 0.10, "k": 0.05},
    "ṭ": {"t": 0.85, "ṭ": 0.10, "T": 0.05},
    "ḍ": {"d": 0.85, "ḍ": 0.10, "D": 0.05},
    "ṣ": {"s": 0.85, "ṣ": 0.10, "S": 0.05},
    "ẓ": {"z": 0.85, "ẓ": 0.10, "Z": 0.05},
    "ṛ": {"r": 0.95, "ṛ": 0.05},
    "u": {"u": 0.75, "ou": 0.25},
    "j": {"j": 0.90, "dj": 0.10},
    "ʷ": {"w": 0.80, "": 0.20},
}

# Lettre canonique -> lettre arabe (ou distribution de variantes)
LAT2ARA: dict[str, str | dict[str, float]] = {
    "b": "ب", "g": {"ڭ": 0.6, "گ": 0.25, "ك": 0.15}, "d": "د", "ḍ": {"ض": 0.85, "ظ": 0.15},
    "f": "ف", "k": "ك", "h": "ه", "ḥ": "ح", "ɛ": "ع", "x": "خ", "q": "ق", "j": "ج", "l": "ل", "m": "م",
    "n": "ن", "r": "ر", "ṛ": "ر", "ɣ": "غ", "s": "س", "ṣ": "ص", "c": "ش", "t": "ت", "ṭ": "ط", "w": "و",
    "y": "ي", "z": "ز", "ẓ": {"ز": 0.8, "ژ": 0.2}, "ʷ": "و",
}
ARA_INITIAL_VOWEL = {"a": "أ", "i": "إ", "u": "أو"}
ARA_MEDIAL_VOWEL = {"a": "ا", "i": "ي", "u": "و"}


def _merge(base: dict, override: dict | None) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        out[str(k)] = {str(a): float(b) for a, b in v.items()} if isinstance(v, dict) else v
    return out


class Renderer:
    def __init__(self, cfg: dict | None = None, rng: random.Random | None = None):
        cfg = cfg or {}
        self.rng = rng or random.Random(0)
        self.latin_social_table = _merge(DEFAULT_LATIN_SOCIAL, cfg.get("latin_social"))
        self.variety_overrides = cfg.get("variety_overrides") or {}
        ara = cfg.get("arabic") or {}
        self.p_shadda = float(ara.get("shadda", 0.4))
        self.ara_table = _merge(LAT2ARA, ara.get("letters"))

    def _pick(self, choice):
        if isinstance(choice, str):
            return choice
        keys = list(choice)
        return self.rng.choices(keys, weights=[choice[k] for k in keys], k=1)[0]

    def latin_social(self, canon: str, variety: str | None = None) -> str:
        table = _merge(self.latin_social_table, self.variety_overrides.get(variety or "", None))
        return "".join(self._pick(table[c]) if c in table else c for c in canon)

    def _arabic_word(self, w: str) -> str:
        out, i, n = [], 0, len(w)
        while i < n:
            c = w[i]
            if c in "aiu":
                initial = i == 0 or not w[i - 1].isalpha()
                out.append(ARA_INITIAL_VOWEL[c] if initial else ARA_MEDIAL_VOWEL[c])
                i += 1
                continue
            if c == "e":
                i += 1
                continue
            if c not in self.ara_table:
                out.append(c)
                i += 1
                continue
            letter = self._pick(self.ara_table[c])
            if i + 1 < n and w[i + 1] == c:  # consonne géminée : chadda ou lettre simple
                out.append(letter + (SHADDA if self.rng.random() < self.p_shadda else ""))
                i += 2
                continue
            out.append(letter)
            i += 1
        return "".join(out)

    def arabic(self, canon: str) -> str:
        return "".join(p if p.isspace() else self._arabic_word(p) for p in _SPLIT.split(canon) if p)

    def render(self, tfng_text: str, script: str, convention: str | None = None,
               variety: str | None = None) -> str:
        if script == "tfng":
            return tfng_text
        canon = tfng_to_latin(tfng_text)
        if script == "ara":
            return self.arabic(canon)
        if convention == "latin_ircam":
            return canon
        return self.latin_social(canon, variety)

    @staticmethod
    def back_to_tifinagh(canon: str) -> str:
        return latin_to_tfng(canon)
