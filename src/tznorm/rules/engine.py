"""Baselines à règles : B0 (lettre à lettre), B1 (règles contextuelles), B2 (B1 + correction lexicale)."""

from __future__ import annotations

import re
import unicodedata

from ..detect.script import token_script
from .arabic import ArabicConverter
from .latin import LatinConverter
from .lexical import LexicalCorrector
from .mapping import load_tables

_SPLIT = re.compile(r"(\s+)")


class RuleNormalizer:
    def __init__(self, level: int = 1, resources: str | None = None, lexicon=None, charlm=None,
                 **corrector_kwargs):
        if level not in (0, 1, 2):
            raise ValueError("level doit valoir 0, 1 ou 2")
        tables = load_tables(str(resources) if resources else None)
        self.level = level
        self.latin = LatinConverter(tables.latin_informal)
        self.arabic = ArabicConverter(tables.arabic)
        self.corrector = None
        if level == 2:
            if lexicon is None:
                raise ValueError("B2 a besoin d'un lexique")
            self.corrector = LexicalCorrector(lexicon, charlm, **corrector_kwargs)

    @property
    def name(self) -> str:
        return f"b{self.level}"

    def convert_token(self, token: str) -> str:
        script = token_script(token)
        if script == "lat":
            return self.latin.naive(token) if self.level == 0 else self.latin.to_tifinagh(token)
        if script == "ara":
            return self.arabic.naive(token) if self.level == 0 else self.arabic.to_tifinagh(token)
        return token  # Tifinagh, ponctuation, nombres, ¤

    def normalize(self, text: str, variety: str | None = None) -> str:
        out = "".join(p if (not p or p.isspace()) else self.convert_token(p) for p in _SPLIT.split(text))
        if self.corrector is not None:
            out = self.corrector.correct(out, variety)
        return unicodedata.normalize("NFC", out)
