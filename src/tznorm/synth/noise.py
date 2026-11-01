"""Bruiteur paramétrable : reproduit les écarts typiques de l'écrit informel, à 4 niveaux d'intensité."""

from __future__ import annotations

import random

from ..tifinagh import EMPHATIC_TO_PLAIN, PLAIN_TO_EMPHATIC, VOWELS_TFNG

DEFAULT_NOISE = {
    "drop_vowel": 0.12,         # voyelles omises (latin, Tifinagh)
    "degeminate": 0.30,         # gémination non notée (ou chadda supprimée)
    "geminate": 0.03,           # consonne doublée à tort
    "insert_schwa": 0.15,       # e inséré entre deux consonnes (latin)
    "repeat_char": 0.03,        # répétition expressive
    "typo": 0.03,               # faute de frappe
    "merge_particle": 0.10,     # particule collée au mot suivant
    "capitalize": 0.10,         # majuscule initiale (latin)
    "drop_vowel_letter": 0.10,  # ا / و / ي omis (arabe)
    "confuse_emphatic": 0.15,   # ⵜ / ⵟ, ⵙ / ⵚ… (Tifinagh)
}
DEFAULT_LEVEL_FACTOR = (0.0, 0.5, 1.0, 2.0)

LAT_VOWELS = set("aiu")
ARA_VOWEL_LETTERS = set("اوي")
SHADDA = "ّ"
_ROWS = ["azertyuiop", "qsdfghjklm", "wxcvbn"]
NEIGHBORS: dict[str, str] = {}
for _r, _row in enumerate(_ROWS):
    for _i, _c in enumerate(_row):
        near = _row[max(0, _i - 1):_i] + _row[_i + 1:_i + 2]
        if _r + 1 < len(_ROWS):
            near += _ROWS[_r + 1][max(0, _i - 1):_i + 1]
        NEIGHBORS[_c] = near


def _is_consonant(c: str, vowels: set) -> bool:
    return c.isalpha() and c not in vowels


class Noiser:
    def __init__(self, probs: dict | None = None, level_factor=DEFAULT_LEVEL_FACTOR,
                 rng: random.Random | None = None):
        self.probs = {**DEFAULT_NOISE, **(probs or {})}
        self.level_factor = list(level_factor)
        self.rng = rng or random.Random(0)

    def apply(self, text: str, script: str, level: int) -> str:
        f = self.level_factor[level] if level < len(self.level_factor) else self.level_factor[-1]
        if f <= 0:
            return text
        p = {k: min(1.0, float(v) * f) for k, v in self.probs.items()}
        words = [self._word(w, script, p) for w in text.split(" ")]
        words = self._merge(words, p["merge_particle"])
        out = " ".join(words)
        if script == "lat" and out and self.rng.random() < p["capitalize"]:
            out = out[0].upper() + out[1:]
        return out

    # --- opérations ---------------------------------------------------------------------------------

    def _word(self, w: str, script: str, p: dict) -> str:
        if not any(c.isalpha() for c in w):
            return w
        chars = list(w)
        if script == "lat":
            chars = self._drop_vowels(chars, LAT_VOWELS, p["drop_vowel"])
            chars = self._degeminate(chars, LAT_VOWELS, p["degeminate"])
            chars = self._geminate(chars, LAT_VOWELS, p["geminate"])
            chars = self._insert_schwa(chars, p["insert_schwa"])
        elif script == "ara":
            chars = self._drop_vowels(chars, ARA_VOWEL_LETTERS, p["drop_vowel_letter"])
            chars = [c for c in chars if not (c == SHADDA and self.rng.random() < p["degeminate"])]
        elif script == "tfng":
            chars = self._confuse_emphatic(chars, p["confuse_emphatic"])
            chars = self._drop_vowels(chars, VOWELS_TFNG, p["drop_vowel"] * 0.5)
            chars = self._degeminate(chars, VOWELS_TFNG, p["degeminate"] * 0.5)
        chars = self._repeat(chars, LAT_VOWELS | ARA_VOWEL_LETTERS | VOWELS_TFNG, p["repeat_char"])
        chars = self._typo(chars, script, p["typo"])
        return "".join(chars) or w

    def _drop_vowels(self, chars, vowels, prob):
        if len(chars) < 3:
            return chars
        # la voyelle initiale est conservée, comme dans l'écrit réel
        out = [c for i, c in enumerate(chars) if i == 0 or c not in vowels or self.rng.random() >= prob]
        return out if any(c.isalpha() for c in out) else chars

    def _degeminate(self, chars, vowels, prob):
        out = []
        for c in chars:
            if out and out[-1] == c and _is_consonant(c, vowels) and self.rng.random() < prob:
                continue
            out.append(c)
        return out

    def _geminate(self, chars, vowels, prob):
        out = []
        for c in chars:
            out.append(c)
            if _is_consonant(c, vowels) and self.rng.random() < prob * 0.2:
                out.append(c)
        return out

    def _insert_schwa(self, chars, prob):
        out = []
        for i, c in enumerate(chars):
            out.append(c)
            if i + 1 < len(chars) and _is_consonant(c, LAT_VOWELS) and _is_consonant(chars[i + 1], LAT_VOWELS) \
                    and chars[i + 1] != c and self.rng.random() < prob:
                out.append("e")
        return out

    def _repeat(self, chars, vowels, prob):
        if self.rng.random() >= prob:
            return chars
        idx = [i for i, c in enumerate(chars) if c in vowels]
        if not idx:
            return chars
        i = self.rng.choice(idx)
        return chars[:i] + [chars[i]] * self.rng.randint(2, 4) + chars[i + 1:]

    def _typo(self, chars, script, prob):
        if len(chars) < 3 or self.rng.random() >= prob:
            return chars
        i = self.rng.randrange(len(chars))
        op = self.rng.choice(["sub", "del", "dup", "swap"])
        c = chars[i]
        if op == "sub" and script == "lat" and c.lower() in NEIGHBORS:
            return chars[:i] + [self.rng.choice(NEIGHBORS[c.lower()])] + chars[i + 1:]
        if op == "del":
            return chars[:i] + chars[i + 1:]
        if op == "swap" and i + 1 < len(chars):
            return chars[:i] + [chars[i + 1], c] + chars[i + 2:]
        return chars[:i + 1] + [c] + chars[i + 1:]

    def _confuse_emphatic(self, chars, prob):
        swap = {**EMPHATIC_TO_PLAIN, **PLAIN_TO_EMPHATIC}
        return [swap[c] if c in swap and self.rng.random() < prob else c for c in chars]

    def _merge(self, words, prob):
        out, i = [], 0
        while i < len(words):
            w = words[i]
            if 0 < len(w) <= 2 and i + 1 < len(words) and words[i + 1] and self.rng.random() < prob:
                out.append(w + words[i + 1])
                i += 2
                continue
            out.append(w)
            i += 1
        return out
