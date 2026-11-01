"""Alphabet Tifinagh IRCAM et correspondance avec la transcription latine canonique.

Les caractères sont écrits en séquences d'échappement pour éviter toute confusion entre glyphes proches.
"""

from __future__ import annotations

import re
import unicodedata

# Transcription latine canonique (IRCAM) -> Tifinagh IRCAM
LAT2TFNG: dict[str, str] = {
    "a": "ⴰ",       # ⴰ
    "b": "ⴱ",       # ⴱ
    "g": "ⴳ",       # ⴳ
    "d": "ⴷ",       # ⴷ
    "ḍ": "ⴹ",  # ḍ -> ⴹ
    "e": "ⴻ",       # ⴻ (schwa, rarement écrit)
    "f": "ⴼ",       # ⴼ
    "k": "ⴽ",       # ⴽ
    "h": "ⵀ",       # ⵀ
    "ḥ": "ⵃ",  # ḥ -> ⵃ
    "ɛ": "ⵄ",  # ɛ -> ⵄ
    "x": "ⵅ",       # ⵅ
    "q": "ⵇ",       # ⵇ
    "i": "ⵉ",       # ⵉ
    "j": "ⵊ",       # ⵊ
    "l": "ⵍ",       # ⵍ
    "m": "ⵎ",       # ⵎ
    "n": "ⵏ",       # ⵏ
    "u": "ⵓ",       # ⵓ
    "r": "ⵔ",       # ⵔ
    "ṛ": "ⵕ",  # ṛ -> ⵕ
    "ɣ": "ⵖ",  # ɣ -> ⵖ
    "s": "ⵙ",       # ⵙ
    "ṣ": "ⵚ",  # ṣ -> ⵚ
    "c": "ⵛ",       # ⵛ
    "t": "ⵜ",       # ⵜ
    "ṭ": "ⵟ",  # ṭ -> ⵟ
    "w": "ⵡ",       # ⵡ
    "y": "ⵢ",       # ⵢ
    "z": "ⵣ",       # ⵣ
    "ẓ": "ⵥ",  # ẓ -> ⵥ
    "ʷ": "ⵯ",  # ʷ -> ⵯ (labialisation)
}
TFNG2LAT: dict[str, str] = {v: k for k, v in LAT2TFNG.items()}

TIFINAGH_LETTERS = frozenset(LAT2TFNG.values())
VOWELS_TFNG = frozenset("ⴰⵉⵓⴻ")  # ⴰ ⵉ ⵓ ⴻ
VOWELS_LAT = frozenset("aiue")
LABIALIZATION = "ⵯ"

EMPHATIC_TO_PLAIN = {
    "ⵟ": "ⵜ",  # ⵟ -> ⵜ
    "ⴹ": "ⴷ",  # ⴹ -> ⴷ
    "ⵚ": "ⵙ",  # ⵚ -> ⵙ
    "ⵥ": "ⵣ",  # ⵥ -> ⵣ
    "ⵕ": "ⵔ",  # ⵕ -> ⵔ
}
PLAIN_TO_EMPHATIC = {v: k for k, v in EMPHATIC_TO_PLAIN.items()}

# Classes phonologiques utilisées pour l'analyse d'erreurs (comme DATASHI)
PHON_CLASSES: dict[str, frozenset[str]] = {
    "emphatique": frozenset(EMPHATIC_TO_PLAIN),
    "uvulaire": frozenset("ⵖⵇⵅ"),     # ⵖ ⵇ ⵅ
    "pharyngale": frozenset("ⵃⵄ"),         # ⵃ ⵄ
    "voyelle": VOWELS_TFNG,
    "labialisation": frozenset(LABIALIZATION),
}

_TFNG_WORD = re.compile("[ⴰ-⵿]+")


def is_tifinagh(ch: str) -> bool:
    return "ⴰ" <= ch <= "⵿"


def phon_class(ch: str) -> str:
    for name, chars in PHON_CLASSES.items():
        if ch in chars:
            return name
    if is_tifinagh(ch):
        return "autre_consonne"
    return "hors_tifinagh"


def tfng_to_latin(text: str) -> str:
    return "".join(TFNG2LAT.get(c, c) for c in unicodedata.normalize("NFC", text))


def latin_to_tfng(text: str) -> str:
    return "".join(LAT2TFNG.get(c, c) for c in unicodedata.normalize("NFC", text))


def tifinagh_words(text: str) -> list[str]:
    return _TFNG_WORD.findall(text)


def skeleton(word: str) -> str:
    """Squelette consonantique : sans voyelles ni labialisation, emphatiques ramenées aux simples,
    géminées réduites. Deux graphies d'un même mot ont en général le même squelette."""
    out: list[str] = []
    for c in word:
        if c in VOWELS_TFNG or c == LABIALIZATION:
            continue
        c = EMPHATIC_TO_PLAIN.get(c, c)
        if not out or out[-1] != c:
            out.append(c)
    return "".join(out)
