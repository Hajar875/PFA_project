"""M4 · Post-traitement : validation de la sortie et réinsertion des éléments masqués."""

from __future__ import annotations

import unicodedata

from .preprocess.masking import PLACEHOLDER, unmask
from .tifinagh import is_tifinagh


def is_valid_output(text: str) -> bool:
    """Vrai si toutes les lettres appartiennent au bloc Tifinagh (ponctuation, chiffres, ¤ et espaces admis)."""
    for ch in text:
        if ch == PLACEHOLDER or ch.isspace() or is_tifinagh(ch):
            continue
        if ch.isalpha() or unicodedata.category(ch) == "Mn":
            return False
    return True


def postprocess(text: str, originals: list[str]) -> tuple[str, bool]:
    restored, ok = unmask(text, originals)
    restored = " ".join(unicodedata.normalize("NFC", restored).split())
    return restored, ok
