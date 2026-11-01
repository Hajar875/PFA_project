"""Mise en forme des paires pour les modèles : jetons de variété et d'écriture, abandon des jetons,
sur-échantillonnage des cellules rares."""

from __future__ import annotations

import math
import random
import re
from collections import defaultdict

from ..detect.script import text_script
from ..preprocess import preprocess

TAG_RE = re.compile(r"^<v:([^>]*)> <s:([^>]*)> ")
UNKNOWN = "?"


def make_source(text: str, variety: str | None = None, script: str | None = None) -> str:
    return f"<v:{variety or UNKNOWN}> <s:{script or UNKNOWN}> {text}"


def record_target(rec: dict) -> str:
    refs = rec.get("references") or [rec.get("sortie", "")]
    return refs[0]


def record_to_pair(rec: dict, rng: random.Random | None = None, tag_dropout: float = 0.0) -> tuple[str, str]:
    """(source étiquetée et masquée, cible masquée). Le même prétraitement qu'à l'inférence est appliqué."""
    src = preprocess(rec["entree"]).text
    tgt = preprocess(record_target(rec)).text
    variety = rec.get("variete") or UNKNOWN
    script = rec.get("ecriture") or text_script(src)
    if rng is not None and tag_dropout > 0:
        if rng.random() < tag_dropout:
            variety = UNKNOWN
        if rng.random() < tag_dropout:
            script = UNKNOWN
    return make_source(src, variety, script), tgt


def cell_of(rec: dict) -> tuple[str, str]:
    return rec.get("variete") or UNKNOWN, rec.get("ecriture") or UNKNOWN


def oversample(records: list[dict], min_fraction: float, rng: random.Random) -> list[dict]:
    """Duplique les exemples des cellules variété × écriture rares jusqu'à `min_fraction` de la plus grande."""
    if min_fraction <= 0 or not records:
        return list(records)
    cells: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        cells[cell_of(r)].append(r)
    target = math.ceil(min_fraction * max(len(v) for v in cells.values()))
    out = list(records)
    for recs in cells.values():
        missing = target - len(recs)
        if missing > 0:
            out.extend(rng.choice(recs) for _ in range(missing))
    rng.shuffle(out)
    return out
