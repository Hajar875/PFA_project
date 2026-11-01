"""Pipeline complet avec un faux modèle : vérifie le reclassement, le repli et la réinsertion des éléments masqués."""

from tznorm.pipeline import Normalizer
from tznorm.rerank import Lexicon, Reranker, RerankWeights
from tznorm.rules import RuleNormalizer


class FakeModel:
    def __init__(self, outputs):
        self.outputs = outputs

    def generate(self, sources, num_beams=5, num_return=5):
        return [self.outputs for _ in sources]

    def score(self, sources, targets):
        return [-3.0 for _ in targets]


def _lexicon():
    lex = Lexicon()
    for w in ["ⵜⴰⵏⵎⵉⵔⵜ", "ⵙ", "ⵡⴰⵟⴰⵙ"]:
        lex.add(w)
    return lex


def test_model_output_with_placeholder():
    model = FakeModel([("ⵜⴰⵏⵎⵉⵔⵜ ⵙ ⵡⴰⵟⴰⵙ ¤", -0.1)])
    norm = Normalizer(RuleNormalizer(level=1), model, Reranker(RerankWeights(threshold=0.0), lexicon=_lexicon()))
    res = norm("tanmirt s watas 😊")
    assert res.sortie == "ⵜⴰⵏⵎⵉⵔⵜ ⵙ ⵡⴰⵟⴰⵙ 😊"
    assert res.source == "modele"


def test_invalid_model_output_falls_back_to_rules():
    model = FakeModel([("tanmirt", -0.1)])
    norm = Normalizer(RuleNormalizer(level=1), model, Reranker(RerankWeights(threshold=0.0)))
    res = norm("tanmirt")
    assert res.sortie == "ⵜⴰⵏⵎⵉⵔⵜ" and res.source == "regles"


def test_rules_only():
    res = Normalizer(RuleNormalizer(level=1))("azul fellawen")
    assert res.sortie == "ⴰⵣⵓⵍ ⴼⵍⵍⴰⵡⵏ" and res.confiance is None
