"""C · Pipeline d'inférence : M0 prétraitement → M1 détection → M2 modèle → M3 reclassement → M4 post-traitement."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .detect.lang import Detector
from .detect.script import text_script
from .io import load_yaml, resolve, resources_dir
from .models.data import make_source
from .postprocess import is_valid_output, postprocess
from .preprocess import Preprocessed, preprocess
from .rerank import Candidate, CharNgramLM, Lexicon, Reranker, RerankWeights
from .rules import RuleNormalizer


@dataclass
class NormResult:
    entree: str
    sortie: str
    ecriture_detectee: str
    variete: str
    confiance: float | None
    alternatives: list[str] = field(default_factory=list)
    source: str = "regles"  # modele | regles

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Prepared:
    pre: Preprocessed
    script: str
    variety: str
    rules_out: str
    source: str  # entrée du modèle, préfixée des jetons


class Normalizer:
    def __init__(self, rules: RuleNormalizer, model=None, reranker: Reranker | None = None,
                 detector: Detector | None = None, num_beams: int = 5, use_reranker: bool = True,
                 fallback_rules: bool = True, protect_foreign: bool = True):
        self.rules = rules
        self.model = model
        self.reranker = reranker or Reranker()
        self.detector = detector
        self.num_beams = num_beams
        self.use_reranker = use_reranker
        self.fallback_rules = fallback_rules
        self.protect_foreign = protect_foreign

    # --- construction --------------------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: str | Path | dict = "configs/pipeline.yaml", **overrides) -> "Normalizer":
        cfg = dict(config) if isinstance(config, dict) else load_yaml(config)
        res = str(resources_dir(cfg.get("resources_dir")))
        lexicon = Lexicon.load([resolve(p) for p in cfg.get("lexicon") or []]) if cfg.get("lexicon") else None
        charlm = CharNgramLM.load(resolve(cfg["charlm"])) if cfg.get("charlm") else None
        detector = Detector.load(resolve(cfg["detector"])) if cfg.get("detector") else None
        level = int(overrides.pop("rules_level", cfg.get("rules_level", 2)))
        if level == 2 and lexicon is None:
            level = 1
        rules = RuleNormalizer(level=level, resources=res, lexicon=lexicon, charlm=charlm)

        mcfg = dict(cfg.get("model") or {})
        if "model_path" in overrides:
            mcfg["path"] = overrides.pop("model_path")
        if "model_type" in overrides:
            mcfg["type"] = overrides.pop("model_type")
        model = None
        if mcfg.get("type") and overrides.pop("load_model", True):
            from .models.base import load_model

            model = load_model(mcfg["type"], resolve(mcfg["path"]), device=mcfg.get("device", "auto"),
                               max_length=int(mcfg.get("max_length", 512)),
                               batch_size=int(mcfg.get("batch_size", 16)))
        rcfg = cfg.get("rerank") or {}
        weights = RerankWeights.load(resolve(rcfg["weights"])) if rcfg.get("weights") else RerankWeights()
        kwargs = dict(num_beams=int(mcfg.get("num_beams", 5)), use_reranker=bool(rcfg.get("enabled", True)),
                      fallback_rules=bool(cfg.get("fallback_rules", True)),
                      protect_foreign=bool(cfg.get("protect_foreign_segments", True)))
        kwargs.update(overrides)
        return cls(rules, model, Reranker(weights, charlm, lexicon), detector, **kwargs)

    @classmethod
    def load(cls, model_path: str | Path | None = None, model_type: str = "byt5",
             config: str | Path = "configs/pipeline.yaml", **overrides) -> "Normalizer":
        if model_path is not None:
            overrides.update(model_path=str(model_path), model_type=model_type)
        return cls.from_config(config, **overrides)

    # --- inférence -----------------------------------------------------------------------------------

    def __call__(self, text: str, variety: str | None = None, script: str | None = None) -> NormResult:
        return self.normalize_batch([text], [variety], [script])[0]

    def prepare(self, texts, varieties=None, scripts=None) -> list[Prepared]:
        n = len(texts)
        varieties = varieties or [None] * n
        scripts = scripts or [None] * n
        out = []
        for text, var, scr in zip(texts, varieties, scripts):
            nfc = unicodedata.normalize("NFC", text)
            extra = self.detector.foreign_spans(nfc) if (self.detector and self.protect_foreign) else []
            pre = preprocess(nfc, extra)
            scr = scr or text_script(pre.text)
            var = var or (self.detector.variety(pre.text) if self.detector else "?")
            out.append(Prepared(pre, scr, var, self.rules.normalize(pre.text, var), make_source(pre.text, var, scr)))
        return out

    def candidates(self, preps: list[Prepared]) -> list[list[Candidate]]:
        """n-best du modèle + sortie des règles (scorée par le modèle), dans l'espace masqué."""
        sources = [p.source for p in preps]
        nbest = self.model.generate(sources, num_beams=self.num_beams, num_return=self.num_beams)
        rule_scores = self.model.score(sources, [p.rules_out for p in preps]) if self.use_reranker else None
        out = []
        for i, hyps in enumerate(nbest):
            cands = [Candidate(t, s, "modele") for t, s in hyps]
            if rule_scores is not None:
                cands.append(Candidate(preps[i].rules_out, rule_scores[i], "regles"))
            out.append(cands)
        return out

    def _decide(self, prep: Prepared, cands: list[Candidate]) -> tuple[str, str, float, list[str]]:
        if self.use_reranker:
            best, conf, ranked = self.reranker.choose(cands)
            alternatives = [c.text for c, _ in ranked if c.text != best.text]
            choice, origin = best.text, best.origin
            low_conf = conf < self.reranker.weights.threshold
        else:
            model_cands = [c for c in cands if c.origin == "modele"]
            m = max(c.model_score for c in model_cands)
            z = sum(math.exp(c.model_score - m) for c in model_cands)
            choice, origin, conf = model_cands[0].text, "modele", math.exp(model_cands[0].model_score - m) / z
            alternatives = [c.text for c in model_cands[1:]]
            low_conf = False
        if self.fallback_rules and (low_conf or not is_valid_output(choice)):
            choice, origin = prep.rules_out, "regles"
        return choice, origin, conf, alternatives

    def normalize_batch(self, texts: list[str], varieties=None, scripts=None) -> list[NormResult]:
        preps = self.prepare(texts, varieties, scripts)
        if self.model is None:
            results = []
            for text, p in zip(texts, preps):
                final, _ = postprocess(p.rules_out, p.pre.originals)
                results.append(NormResult(text, final, p.script, p.variety, None, [], "regles"))
            return results

        results = []
        for text, p, cands in zip(texts, preps, self.candidates(preps)):
            choice, origin, conf, alts = self._decide(p, cands)
            final, ok = postprocess(choice, p.pre.originals)
            if not ok and self.fallback_rules and origin != "regles":
                final, _ = postprocess(p.rules_out, p.pre.originals)
                origin = "regles"
            alternatives = [postprocess(a, p.pre.originals)[0] for a in alts[:4]]
            results.append(NormResult(text, final, p.script, p.variety, round(conf, 4), alternatives, origin))
        return results
