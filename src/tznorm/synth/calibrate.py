"""Calibration du bruiteur sur des paires réelles annotées (entrée latine informelle, référence Tifinagh).

Les estimations sont volontairement simples (comptages, sans alignement fin) : elles servent à rapprocher
la distribution du bruit synthétique de celle de l'échantillon pilote.
"""

from __future__ import annotations

import re
import unicodedata

from ..detect.script import text_script
from ..tifinagh import tfng_to_latin
from .reverse import DEFAULT_LATIN_SOCIAL

_DOUBLE = re.compile(r"([^aiue\W\d])\1")


def _ratio(num: float, den: float, default: float) -> float:
    return round(min(max(num / den, 0.0), 1.0), 3) if den else default


def calibrate(records: list[dict]) -> dict:
    """Renvoie un fragment de configuration `noise` + `latin_social` à fusionner dans configs/synth.yaml."""
    vowels_in = vowels_ref = schwa = words = gem_in = gem_ref = 0
    variant_counts = {letter: {v: 0 for v in dist} for letter, dist in DEFAULT_LATIN_SOCIAL.items()}
    letter_counts = {letter: 0 for letter in DEFAULT_LATIN_SOCIAL}
    n = 0
    for rec in records:
        src = unicodedata.normalize("NFC", rec["entree"]).lower()
        if text_script(src) != "lat":
            continue
        canon = tfng_to_latin(rec["references"][0])
        n += 1
        vowels_in += sum(src.count(v) for v in "aiu") - src.count("ou")
        vowels_ref += sum(canon.count(v) for v in "aiu")
        schwa += src.count("e")
        words += len(src.split())
        gem_in += len(_DOUBLE.findall(src))
        gem_ref += len(_DOUBLE.findall(canon))
        for letter in letter_counts:
            occ = canon.count(letter)
            if not occ:
                continue
            letter_counts[letter] += occ
            for variant in variant_counts[letter]:
                # les variantes d'une lettre ambiguë (t, d, a…) ne sont pas comptables sans alignement
                if variant and (len(variant) >= 2 or variant in "3579ɛɣḥṭḍṣẓṛ"):
                    variant_counts[letter][variant] += src.count(variant)

    noise = {
        "drop_vowel": _ratio(vowels_ref - vowels_in, vowels_ref, 0.12),
        "insert_schwa": _ratio(schwa, words, 0.15),
        "degeminate": _ratio(gem_ref - gem_in, gem_ref, 0.30),
    }
    latin_social = {}
    for letter, counts in variant_counts.items():
        total = letter_counts[letter]
        if not total:
            continue
        dist = {v: c / total for v, c in counts.items() if c}
        rest = max(0.0, 1.0 - sum(dist.values()))
        default = max(DEFAULT_LATIN_SOCIAL[letter], key=DEFAULT_LATIN_SOCIAL[letter].get)
        dist[default] = dist.get(default, 0.0) + rest
        latin_social[letter] = {k: round(v, 3) for k, v in dist.items()}
    return {"n_paires_latines": n, "noise": noise, "latin_social": latin_social}
