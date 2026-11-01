"""Analyse des erreurs par classe phonologique (géminées, emphatiques, uvulaires, pharyngales, voyelles)."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from ..tifinagh import phon_class
from .align import align
from .metrics import best_ref, norm


def _is_gemination(s: str, pos: int) -> bool:
    return (pos > 0 and s[pos - 1] == s[pos]) or (pos + 1 < len(s) and s[pos + 1] == s[pos])


def phon_analysis(hyps: Sequence[str], refs: Sequence[Sequence[str]]) -> dict[str, dict]:
    """Pour chaque classe : nombre de caractères de référence, nombre d'erreurs, taux d'erreur."""
    n_ref: Counter = Counter()
    n_err: Counter = Counter()
    gem_err = gem_n = 0
    for h, rs in zip(hyps, refs):
        r, _ = best_ref(h, rs)
        h = norm(h)
        for pos, ch in enumerate(r):
            if not ch.isspace():
                n_ref[phon_class(ch)] += 1
            if pos > 0 and r[pos - 1] == ch and not ch.isspace():
                gem_n += 1
        for op in align(r, h):
            if op.kind == "=":
                continue
            if op.kind in ("S", "D"):
                n_err[phon_class(op.ref)] += 1
                if op.kind == "D" and _is_gemination(r, op.ref_pos):
                    gem_err += 1
            else:
                n_err[phon_class(op.hyp)] += 1
                if _is_gemination(h, op.hyp_pos):
                    gem_err += 1
    out = {}
    for cls in sorted(set(n_ref) | set(n_err)):
        out[cls] = {"n": n_ref[cls], "erreurs": n_err[cls], "taux": n_err[cls] / n_ref[cls] if n_ref[cls] else None}
    out["gemination"] = {"n": gem_n, "erreurs": gem_err, "taux": gem_err / gem_n if gem_n else None}
    return out
