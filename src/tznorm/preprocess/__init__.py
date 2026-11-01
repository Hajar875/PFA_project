"""M0 · Prétraitement."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from .masking import PLACEHOLDER, Span, find_spans, mask, unmask
from .textclean import clean, reduce_repetitions

__all__ = ["PLACEHOLDER", "Preprocessed", "Span", "clean", "find_spans", "mask", "preprocess",
           "reduce_repetitions", "unmask"]


@dataclass
class Preprocessed:
    text: str                      # texte nettoyé, éléments protégés remplacés par ¤
    originals: list[str] = field(default_factory=list)
    raw: str = ""


def preprocess(text: str, extra_spans: list[Span] | None = None, max_run: int = 2) -> Preprocessed:
    """NFC → masquage → nettoyage → réduction des répétitions.

    Le masquage précède le nettoyage pour que les emojis composés (avec ZWJ) soient restitués intacts.
    Les positions de `extra_spans` se rapportent au texte en NFC.
    """
    nfc = unicodedata.normalize("NFC", text)
    masked, originals = mask(nfc, extra_spans)
    return Preprocessed(reduce_repetitions(clean(masked), max_run), originals, text)
