"""Classifieur bayésien naïf sur n-grammes de caractères, pour la langue et la variété.

Implémenté sans dépendance (fastText ne s'installe pas toujours sous Windows). Les données d'amorçage
peuvent venir des lexiques d'Adouane et al. (2016) et des métadonnées du jeu Gold.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from ..preprocess.masking import Span


class NgramNB:
    def __init__(self, n_min: int = 1, n_max: int = 4, alpha: float = 0.5):
        self.n_min, self.n_max, self.alpha = n_min, n_max, alpha
        self.counts: dict[str, Counter] = {}
        self.totals: dict[str, int] = {}
        self.docs: Counter = Counter()
        self.vocab_size = 1

    def _features(self, text: str):
        t = f" {text.lower()} "
        for n in range(self.n_min, self.n_max + 1):
            for i in range(len(t) - n + 1):
                yield t[i:i + n]

    def fit(self, texts: list[str], labels: list[str]) -> "NgramNB":
        vocab = set()
        for text, lab in zip(texts, labels):
            c = self.counts.setdefault(lab, Counter())
            feats = list(self._features(text))
            c.update(feats)
            vocab.update(feats)
            self.docs[lab] += 1
        self.totals = {lab: sum(c.values()) for lab, c in self.counts.items()}
        self.vocab_size = max(len(vocab), 1)
        return self

    def log_scores(self, text: str) -> dict[str, float]:
        n_docs = sum(self.docs.values())
        feats = list(self._features(text))
        scores = {}
        for lab, c in self.counts.items():
            denom = self.totals[lab] + self.alpha * self.vocab_size
            s = math.log(self.docs[lab] / n_docs)
            for f in feats:
                s += math.log((c.get(f, 0) + self.alpha) / denom)
            scores[lab] = s
        return scores

    def predict_proba(self, text: str) -> dict[str, float]:
        scores = self.log_scores(text)
        if not scores:
            return {}
        m = max(scores.values())
        exp = {lab: math.exp(s - m) for lab, s in scores.items()}
        z = sum(exp.values())
        return {lab: v / z for lab, v in exp.items()}

    def predict(self, text: str) -> tuple[str, float]:
        proba = self.predict_proba(text)
        if not proba:
            return "?", 0.0
        lab = max(proba, key=proba.get)
        return lab, proba[lab]

    def to_dict(self) -> dict:
        return {"n_min": self.n_min, "n_max": self.n_max, "alpha": self.alpha, "vocab_size": self.vocab_size,
                "docs": dict(self.docs), "counts": {k: dict(v) for k, v in self.counts.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "NgramNB":
        m = cls(d["n_min"], d["n_max"], d["alpha"])
        m.vocab_size = d["vocab_size"]
        m.docs = Counter(d["docs"])
        m.counts = {k: Counter(v) for k, v in d["counts"].items()}
        m.totals = {k: sum(v.values()) for k, v in m.counts.items()}
        return m


class Detector:
    """Variété de la phrase (rif / tzm / shi) et segments dans une autre langue (fr, darija…)."""

    def __init__(self, variety_clf: NgramNB | None = None, lang_clf: NgramNB | None = None,
                 variety_threshold: float = 0.6, foreign_threshold: float = 0.9, min_foreign_run: int = 3,
                 native_label: str = "zgh"):
        self.variety_clf = variety_clf
        self.lang_clf = lang_clf
        self.variety_threshold = variety_threshold
        self.foreign_threshold = foreign_threshold
        self.min_foreign_run = min_foreign_run
        self.native_label = native_label

    def variety(self, text: str) -> str:
        if self.variety_clf is None:
            return "?"
        lab, p = self.variety_clf.predict(text)
        return lab if p >= self.variety_threshold else "?"

    def foreign_spans(self, text: str) -> list[Span]:
        """Suites d'au moins `min_foreign_run` tokens classés dans une autre langue, à conserver telles quelles."""
        if self.lang_clf is None:
            return []
        spans, run = [], []
        for m in re.finditer(r"\S+", text):
            tok = m.group(0)
            foreign = False
            if sum(ch.isalpha() for ch in tok) >= 2:
                lab, p = self.lang_clf.predict(tok)
                foreign = lab != self.native_label and p >= self.foreign_threshold
            if foreign:
                run.append(m)
                continue
            if len(run) >= self.min_foreign_run:
                spans.append(Span(run[0].start(), run[-1].end(), text[run[0].start():run[-1].end()], "foreign"))
            run = []
        if len(run) >= self.min_foreign_run:
            spans.append(Span(run[0].start(), run[-1].end(), text[run[0].start():run[-1].end()], "foreign"))
        return spans

    def save(self, path: str | Path) -> None:
        d = {"variety_threshold": self.variety_threshold, "foreign_threshold": self.foreign_threshold,
             "min_foreign_run": self.min_foreign_run, "native_label": self.native_label,
             "variety_clf": self.variety_clf.to_dict() if self.variety_clf else None,
             "lang_clf": self.lang_clf.to_dict() if self.lang_clf else None}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Detector":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(NgramNB.from_dict(d["variety_clf"]) if d.get("variety_clf") else None,
                   NgramNB.from_dict(d["lang_clf"]) if d.get("lang_clf") else None,
                   d["variety_threshold"], d["foreign_threshold"], d["min_foreign_run"], d["native_label"])
