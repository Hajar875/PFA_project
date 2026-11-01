"""Nettoyage Unicode : NFC, caractères invisibles, homoglyphes, répétitions expressives."""

from __future__ import annotations

import re
import unicodedata

# Trait d'union conditionnel, espaces de largeur nulle, marques de direction, BOM, tatouil arabe
_INVISIBLE = re.compile("[­​-‏‪-‮⁠-⁤﻿ـ]")
_SPACES = re.compile(r"[ \t  -  　]+")

HOMOGLYPHS = str.maketrans({
    "γ": "ɣ",  # γ grec -> ɣ latin
    "ε": "ɛ",  # ε grec -> ɛ latin
    "ک": "ك",  # ک (keheh) -> ك
    "ی": "ي",  # ی (yeh persan) -> ي
    "ہ": "ه",  # ہ -> ه
    "ⵝ": "ⵜ",  # ⵝ (yath) -> ⵜ : spirantisation non notée par l'IRCAM
    "’": "'",
    "ʼ": "'",
})


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE.sub("", text)
    text = text.translate(HOMOGLYPHS)
    text = unicodedata.normalize("NFC", text)
    return _SPACES.sub(" ", text).strip()


def reduce_repetitions(text: str, max_run: int = 2) -> str:
    """Réduit les lettres répétées 3 fois ou plus à `max_run` (azuuuul -> azuul). La ponctuation est conservée."""
    pattern = re.compile(r"([^\W\d_])\1{%d,}" % max_run)
    return pattern.sub(lambda m: m.group(1) * max_run, text)
