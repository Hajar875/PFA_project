import pytest

from tznorm.eval.align import align
from tznorm.eval.bootstrap import paired_bootstrap
from tznorm.eval.metrics import chrf, corpus_cer, corpus_wer, exact_match, levenshtein
from tznorm.eval.phon import phon_analysis


def test_levenshtein():
    assert levenshtein("abc", "abd") == 1
    assert levenshtein("", "abc") == 3


def test_cer_multi_reference():
    assert corpus_cer(["abc"], [["abd"]]) == pytest.approx(1 / 3)
    assert corpus_cer(["abc"], [["xyz", "abc"]]) == 0.0


def test_wer_and_exact_match():
    assert corpus_wer(["a b c"], [["a b d"]]) == pytest.approx(1 / 3)
    assert exact_match(["ⴰⵣⵓⵍ", "x"], [["ⴰⵣⵓⵍ"], ["y"]]) == 0.5


def test_chrf_identity():
    assert chrf(["ⵜⴰⵏⵎⵉⵔⵜ ⵙ ⵡⴰⵟⴰⵙ"], [["ⵜⴰⵏⵎⵉⵔⵜ ⵙ ⵡⴰⵟⴰⵙ"]]) == pytest.approx(100.0)


def test_align_ops():
    kinds = [op.kind for op in align("ⵡⴰⵟⴰⵙ", "ⵡⴰⵜⴰⵙ")]
    assert kinds.count("S") == 1 and kinds.count("=") == 4


def test_phon_emphatic_error():
    res = phon_analysis(["ⵡⴰⵜⴰⵙ"], [["ⵡⴰⵟⴰⵙ"]])
    assert res["emphatique"]["erreurs"] == 1


def test_bootstrap_sign():
    refs = [["ⴰⵣⵓⵍ"]] * 20
    res = paired_bootstrap(["ⴰⵣⵓⵍ"] * 20, ["ⴰⵣⵍ"] * 20, refs, n_samples=200)
    assert res["delta"] < 0 and res["p_value"] == 0.0
