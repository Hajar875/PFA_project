"""Modèle de langue à n-grammes de caractères (lissage de Witten-Bell interpolé), en Python pur.

Remplace KenLM pour éviter une compilation native ; suffisant pour reclasser quelques candidats.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

BOS, EOS = "\x02", "\x03"


class CharNgramLM:
    def __init__(self, order: int = 6):
        self.order = order
        self.counts: list[dict[str, Counter]] = [defaultdict(Counter) for _ in range(order)]
        self.totals: list[dict[str, int]] = [{} for _ in range(order)]
        self.types: list[dict[str, int]] = [{} for _ in range(order)]
        self.vocab: set[str] = set()

    def fit(self, lines: Iterable[str]) -> "CharNgramLM":
        pad = BOS * (self.order - 1)
        for line in lines:
            s = pad + line + EOS
            for i in range(self.order - 1, len(s)):
                c = s[i]
                self.vocab.add(c)
                for k in range(self.order):
                    self.counts[k][s[i - k:i]][c] += 1
        for k in range(self.order):
            self.counts[k] = dict(self.counts[k])
            self.totals[k] = {h: sum(c.values()) for h, c in self.counts[k].items()}
            self.types[k] = {h: len(c) for h, c in self.counts[k].items()}
        return self

    def prob(self, c: str, history: str) -> float:
        p = 1.0 / (len(self.vocab) + 1)
        for k in range(self.order):
            h = history[len(history) - k:] if k else ""
            cnt = self.counts[k].get(h)
            if not cnt:
                continue
            t = self.types[k][h]
            p = (cnt.get(c, 0) + t * p) / (self.totals[k][h] + t)
        return p

    def logprob(self, text: str, context: str = "") -> float:
        """log p(text | context), fin de séquence comprise."""
        h = (BOS * (self.order - 1) + context)[-(self.order - 1):] if self.order > 1 else ""
        lp = 0.0
        for c in text + EOS:
            lp += math.log(self.prob(c, h))
            if self.order > 1:
                h = (h + c)[-(self.order - 1):]
        return lp

    def avg_logprob(self, text: str, context: str = "") -> float:
        return self.logprob(text, context) / (len(text) + 1)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"order": self.order, "counts": [{h: dict(c) for h, c in d.items()} for d in self.counts],
                         "vocab": sorted(self.vocab)}, f)

    @classmethod
    def load(cls, path: str | Path) -> "CharNgramLM":
        with open(path, "rb") as f:
            d = pickle.load(f)
        lm = cls(d["order"])
        lm.vocab = set(d["vocab"])
        lm.counts = [{h: Counter(c) for h, c in level.items()} for level in d["counts"]]
        lm.totals = [{h: sum(c.values()) for h, c in level.items()} for level in lm.counts]
        lm.types = [{h: len(c) for h, c in level.items()} for level in lm.counts]
        return lm
