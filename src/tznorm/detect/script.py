"""Identification de l'écriture par caractère, par token et par phrase (plages Unicode)."""

from __future__ import annotations

import unicodedata
from collections import Counter

from ..tifinagh import is_tifinagh

SCRIPTS = ("lat", "ara", "tfng")
_LATIN_EXTRA = set("ɛɣʷγε")  # ɛ ɣ ʷ γ ε


def _is_arabic(ch: str) -> bool:
    o = ord(ch)
    return 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F or 0x08A0 <= o <= 0x08FF \
        or 0xFB50 <= o <= 0xFDFF or 0xFE70 <= o <= 0xFEFF


def char_script(ch: str) -> str | None:
    if is_tifinagh(ch):
        return "tfng" if ch.isalpha() else None
    if _is_arabic(ch):
        return "ara" if ch.isalpha() else None
    if ch in _LATIN_EXTRA:
        return "lat"
    if ch.isalpha() and unicodedata.name(ch, "").startswith("LATIN"):
        return "lat"
    return None


def _majority(counts: Counter, default: str) -> str:
    if not counts:
        return default
    return counts.most_common(1)[0][0]


def token_script(token: str, default: str = "?") -> str:
    return _majority(Counter(s for s in map(char_script, token) if s), default)


def text_script(text: str, default: str = "?") -> str:
    """Écriture majoritaire de la phrase : lat, ara, tfng, ou `default` si aucune lettre."""
    return _majority(Counter(s for s in map(char_script, text) if s), default)
