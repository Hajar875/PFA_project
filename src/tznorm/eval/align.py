"""Alignement caractère à caractère (Levenshtein avec retour arrière)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Op:
    kind: str          # "=", "S" (substitution), "D" (suppression), "I" (insertion)
    ref: str           # caractère de la référence ("" pour I)
    hyp: str           # caractère de l'hypothèse ("" pour D)
    ref_pos: int       # position dans la référence (-1 pour I)
    hyp_pos: int       # position dans l'hypothèse (-1 pour D)


def align(ref: str, hyp: str) -> list[Op]:
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    ops: list[Op] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            kind = "=" if ref[i - 1] == hyp[j - 1] else "S"
            ops.append(Op(kind, ref[i - 1], hyp[j - 1], i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(Op("D", ref[i - 1], "", i - 1, -1))
            i -= 1
        else:
            ops.append(Op("I", "", hyp[j - 1], -1, j - 1))
            j -= 1
    return ops[::-1]
