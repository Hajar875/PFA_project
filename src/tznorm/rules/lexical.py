"""Correction lexicale de B2 : pour chaque mot inconnu, candidats du lexique de même squelette consonantique,
classés par distance pondérée, variété, fréquence et modèle de langue caractère."""

from __future__ import annotations

import math
import re

from ..tifinagh import is_tifinagh
from .distance import weighted_distance

_CORE = re.compile(r"^(\W*)(.*?)(\W*)$", re.S)
_SPLIT = re.compile(r"(\s+)")


class LexicalCorrector:
    def __init__(self, lexicon, charlm=None, max_cost: float = 2.5, lm_weight: float = 0.2,
                 variety_penalty: float = 0.5, freq_weight: float = 0.05):
        self.lex = lexicon
        self.lm = charlm
        self.max_cost = max_cost
        self.lm_weight = lm_weight
        self.variety_penalty = variety_penalty
        self.freq_weight = freq_weight

    def correct_word(self, word: str, variety: str | None = None, context: str = "") -> str:
        if not word or not all(is_tifinagh(c) for c in word) or self.lex.contains(word):
            return word
        best, best_cost = word, math.inf
        for form in self.lex.candidates(word):
            cost = weighted_distance(word, form)
            if variety and variety != "?" and not self.lex.has_variety(form, variety):
                cost += self.variety_penalty
            if self.lm is not None:
                cost -= self.lm_weight * self.lm.avg_logprob(form, context)
            cost -= self.freq_weight * math.log1p(self.lex.freq(form))
            if cost < best_cost:
                best, best_cost = form, cost
        return best if best_cost <= self.max_cost else word

    def correct(self, text: str, variety: str | None = None) -> str:
        out, context = [], ""
        for part in _SPLIT.split(text):
            if not part or part.isspace():
                out.append(part)
                continue
            pre, core, post = _CORE.match(part).groups()
            part = pre + self.correct_word(core, variety, context) + post
            out.append(part)
            context = (context + " " + part)[-32:]
        return "".join(out)
