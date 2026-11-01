"""M3 · Reclassement des candidats (n-best du modèle + sortie des règles).

score = λ_modèle · log p(modèle) + λ_LM · log p(LM caractère) + λ_lex · part de mots connus
La confiance est la probabilité softmax du meilleur candidat ; sous le seuil, le pipeline revient aux règles.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..eval.metrics import corpus_cer


@dataclass
class Candidate:
    text: str
    model_score: float
    origin: str = "modele"  # modele | regles


@dataclass
class RerankWeights:
    model: float = 1.0
    lm: float = 0.5
    lex: float = 1.0
    threshold: float = 0.3

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "RerankWeights":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class TuneItem:
    candidates: list[Candidate]
    references: list[str]
    rules_text: str


DEFAULT_GRID = {
    "model": [1.0],
    "lm": [0.0, 0.25, 0.5, 1.0],
    "lex": [0.0, 0.5, 1.0, 2.0],
    "threshold": [0.0, 0.2, 0.3, 0.4, 0.5],
}


@dataclass
class Reranker:
    weights: RerankWeights = field(default_factory=RerankWeights)
    lm: object | None = None
    lexicon: object | None = None

    def score(self, cand: Candidate, w: RerankWeights | None = None) -> float:
        w = w or self.weights
        s = w.model * cand.model_score
        if self.lm is not None and w.lm:
            s += w.lm * self.lm.avg_logprob(cand.text)
        if self.lexicon is not None and w.lex:
            s += w.lex * self.lexicon.coverage(cand.text)
        return s

    @staticmethod
    def _dedupe(cands: list[Candidate]) -> list[Candidate]:
        best: dict[str, Candidate] = {}
        for c in cands:
            cur = best.get(c.text)
            if cur is None or c.model_score > cur.model_score:
                origin = "modele" if (cur and cur.origin == "modele") else c.origin
                best[c.text] = Candidate(c.text, c.model_score, origin)
        return list(best.values())

    def choose(self, cands: list[Candidate], w: RerankWeights | None = None):
        """Renvoie (meilleur candidat, confiance, liste classée [(candidat, probabilité)])."""
        cands = self._dedupe(cands)
        scores = [self.score(c, w) for c in cands]
        m = max(scores)
        exp = [math.exp(s - m) for s in scores]
        z = sum(exp)
        ranked = sorted(zip(cands, (e / z for e in exp)), key=lambda x: -x[1])
        return ranked[0][0], ranked[0][1], ranked

    def decide(self, item: TuneItem, w: RerankWeights) -> str:
        best, conf, _ = self.choose(item.candidates, w)
        return best.text if conf >= w.threshold else item.rules_text

    def tune(self, items: list[TuneItem], grid: dict | None = None) -> tuple[RerankWeights, float]:
        """Recherche exhaustive des poids et du seuil minimisant le CER sur le Gold dev."""
        grid = grid or DEFAULT_GRID
        refs = [it.references for it in items]
        best_w, best_cer = self.weights, math.inf
        for model, lm, lex, thr in itertools.product(grid["model"], grid["lm"], grid["lex"], grid["threshold"]):
            w = RerankWeights(model, lm, lex, thr)
            cer = corpus_cer([self.decide(it, w) for it in items], refs)
            if cer < best_cer:
                best_w, best_cer = w, cer
        self.weights = best_w
        return best_w, best_cer
