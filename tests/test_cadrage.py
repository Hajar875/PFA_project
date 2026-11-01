"""Les 20 exemples du document de cadrage, en test de non-régression (baseline B2 + lexique d'exemple)."""

import pytest

from tznorm.io import read_jsonl, resources_dir
from tznorm.pipeline import Normalizer
from tznorm.rerank import Lexicon
from tznorm.rules import RuleNormalizer

EXAMPLES = read_jsonl("data/samples/cadrage.jsonl")


@pytest.fixture(scope="module")
def b2():
    lexicon = Lexicon.load([resources_dir() / "lexicon" / "lexique_exemple.tsv"])
    return Normalizer(RuleNormalizer(level=2, lexicon=lexicon))


@pytest.mark.parametrize("rec", EXAMPLES, ids=[r["id"] for r in EXAMPLES])
def test_cadrage(b2, rec):
    assert b2(rec["entree"]).sortie in rec["references"]
