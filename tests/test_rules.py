import pytest

from tznorm.rules import RuleNormalizer
from tznorm.rules.latin import LatinConverter
from tznorm.rules.mapping import load_tables


@pytest.fixture(scope="module")
def latin():
    return LatinConverter(load_tables().latin_informal)


@pytest.fixture(scope="module")
def b1():
    return RuleNormalizer(level=1)


@pytest.mark.parametrize("inp, canon", [
    ("Amazigh", "amaziɣ"),
    ("fellawen", "fllawn"),
    ("thamazighth", "tamaziɣt"),
    ("chukran", "cukran"),
    ("3ari", "ɛari"),
    ("azuul", "azul"),
    ("2024", "2024"),
])
def test_latin_canonical(latin, inp, canon):
    assert latin.to_canonical(inp) == canon


@pytest.mark.parametrize("inp, out", [
    ("أزول", "ⴰⵣⵓⵍ"),
    ("تانمّيرت", "ⵜⴰⵏⵎⵎⵉⵔⵜ"),
    ("واطاس", "ⵡⴰⵟⴰⵙ"),
    ("أور", "ⵓⵔ"),
    ("أوال", "ⴰⵡⴰⵍ"),
    ("فلّاون", "ⴼⵍⵍⴰⵡⵏ"),
])
def test_arabic(b1, inp, out):
    assert b1.normalize(inp) == out


def test_b0_has_no_digraphs():
    assert RuleNormalizer(level=0).normalize("tamazight") == "ⵜⴰⵎⴰⵣⵉⴳⵀⵜ"
    assert RuleNormalizer(level=1).normalize("tamazight") == "ⵜⴰⵎⴰⵣⵉⵖⵜ"


def test_tifinagh_identity_and_punctuation(b1):
    assert b1.normalize("ⴰⵣⵓⵍ, tanmirt!") == "ⴰⵣⵓⵍ, ⵜⴰⵏⵎⵉⵔⵜ!"


def test_b2_requires_lexicon():
    with pytest.raises(ValueError):
        RuleNormalizer(level=2)
