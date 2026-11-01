"""Découpages du jeu Gold et garde-fous contre les fuites entre entraînement et évaluation."""

from __future__ import annotations

import random
import unicodedata
from collections import defaultdict

from .tifinagh import tifinagh_words

CELL_KEYS = ("variete", "ecriture", "registre")


def _cell(rec: dict) -> tuple:
    return tuple(rec.get(k, "?") for k in CELL_KEYS)


def stratified_split(records: list[dict], dev_ratio: float = 1 / 3, seed: int = 13) -> tuple[list[dict], list[dict]]:
    """Coupe chaque cellule variété × écriture × registre en dev / test avec la même proportion."""
    rng = random.Random(seed)
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        cells[_cell(r)].append(r)
    dev, test = [], []
    for key in sorted(cells):
        recs = cells[key][:]
        rng.shuffle(recs)
        k = round(len(recs) * dev_ratio)
        dev += recs[:k]
        test += recs[k:]
    return dev, test


def _targets(records: list[dict]) -> set[str]:
    return {unicodedata.normalize("NFC", ref) for r in records for ref in r.get("references", [])}


def training_vocabulary(records: list[dict]) -> set[str]:
    return {w for t in _targets(records) for w in tifinagh_words(t)}


def mark_oov(gold: list[dict], train: list[dict]) -> list[dict]:
    """Ajoute `oov: true` aux phrases du Gold contenant au moins un mot absent de l'entraînement."""
    vocab = training_vocabulary(train)
    out = []
    for r in gold:
        words = {w for ref in r.get("references", []) for w in tifinagh_words(ref)}
        out.append({**r, "oov": any(w not in vocab for w in words)})
    return out


def _ngrams(text: str, n: int) -> set[tuple]:
    w = text.split()
    return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}


def check_leak(train: list[dict], gold: list[dict], n: int = 4) -> dict:
    """Recouvrement exact des cibles et part des n-grammes de mots du Gold présents à l'entraînement."""
    train_t, gold_t = _targets(train), _targets(gold)
    exact = sorted(gold_t & train_t)
    train_ng = set().union(*(_ngrams(t, n) for t in train_t)) if train_t else set()
    gold_ng = [g for t in gold_t for g in _ngrams(t, n)]
    shared = sum(g in train_ng for g in gold_ng)
    return {"cibles_communes": len(exact), "exemples": exact[:10],
            f"part_{n}grammes_communs": shared / len(gold_ng) if gold_ng else 0.0}
