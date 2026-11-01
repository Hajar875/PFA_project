import unicodedata

from tznorm.postprocess import is_valid_output, postprocess
from tznorm.preprocess import PLACEHOLDER, clean, preprocess, reduce_repetitions


def test_invisibles_and_nfc():
    assert clean("a​zu­l") == "azul"
    assert clean("waṭas") == "waṭas"  # t + point souscrit combinant -> ṭ précomposé


def test_homoglyphs():
    assert clean("amaziγ") == "amaziɣ"  # γ grec -> ɣ


def test_repetitions_letters_only():
    assert reduce_repetitions("azuuuul!!!") == "azuul!!!"


def test_mask_and_restore():
    text = "azul https://x.ma 2024 😊"
    pre = preprocess(text)
    assert pre.text == f"azul {PLACEHOLDER} {PLACEHOLDER} {PLACEHOLDER}"
    assert pre.originals == ["https://x.ma", "2024", "😊"]
    restored, ok = postprocess(pre.text, pre.originals)
    assert ok and restored == unicodedata.normalize("NFC", text)


def test_digits_inside_words_are_not_masked():
    assert preprocess("3la").text == "3la"


def test_valid_output():
    assert is_valid_output("ⴰⵣⵓⵍ, ¤ 2024 !")
    assert not is_valid_output("ⴰⵣⵓⵍ azul")
