import random

import pytest

from tznorm.rerank import CharNgramLM, Lexicon
from tznorm.rules import RuleNormalizer
from tznorm.synth import Noiser, Renderer
from tznorm.tifinagh import skeleton

WORDS = ["ⴰⵣⵓⵍ", "ⵜⴰⵏⵎⵎⵉⵔⵜ", "ⴰⵎⴰⵣⵉⵖ", "ⵜⴰⵎⴰⵣⵉⵖⵜ", "ⵜⵉⴼⵉⵏⴰⵖ", "ⵜⴰⴼⵓⴽⵜ", "ⴰⵎⴰⵏ", "ⴰⵔⴳⴰⵣ",
         "ⵉⴷⵍⵉⵙ", "ⵡⴰⵟⴰⵙ", "ⴼⵍⵍⴰⵡⵏ", "ⵙⵙⵉⵏⴰⵅ", "ⵓⵔ"]

DETERMINISTIC = {"arabic": {"shadda": 1.0, "letters": {"g": {"ڭ": 1.0}, "ḍ": {"ض": 1.0}, "ẓ": {"ژ": 1.0}}}}


@pytest.mark.parametrize("word", WORDS)
def test_roundtrip_latin_ircam(word):
    src = Renderer().render(word, "lat", "latin_ircam")
    assert RuleNormalizer(level=1).normalize(src) == word


@pytest.mark.parametrize("word", WORDS)
def test_roundtrip_arabic(word):
    src = Renderer(DETERMINISTIC).render(word, "ara")
    assert RuleNormalizer(level=1).normalize(src) == word


def test_noise_level_zero_is_identity():
    noiser = Noiser(rng=random.Random(1))
    assert noiser.apply("tanmirt s watas", "lat", 0) == "tanmirt s watas"


def test_noise_is_reproducible():
    a = Noiser(rng=random.Random(7)).apply("tanmmirt s waṭas bahra", "lat", 3)
    b = Noiser(rng=random.Random(7)).apply("tanmmirt s waṭas bahra", "lat", 3)
    assert a == b


def test_skeleton_and_lexicon_candidates():
    assert skeleton("ⵙⵙⵉⵏⴰⵅ") == skeleton("ⵙⵙⵏⵅ")
    lex = Lexicon()
    lex.add("ⵡⴰⵟⴰⵙ")
    assert "ⵡⴰⵟⴰⵙ" in lex.candidates("ⵡⴰⵜⴰⵙ")


def test_charlm_prefers_seen_text():
    lm = CharNgramLM(order=4).fit(["ⵜⴰⵏⵎⵉⵔⵜ ⵙ ⵡⴰⵟⴰⵙ"] * 5)
    assert lm.avg_logprob("ⵜⴰⵏⵎⵉⵔⵜ") > lm.avg_logprob("ⵔⵜⵎⵏⴰⵜ")
