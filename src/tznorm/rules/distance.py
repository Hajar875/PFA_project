"""Distance d'édition pondérée pour le Tifinagh : les erreurs typiques de l'écrit informel coûtent moins cher."""

from __future__ import annotations

from ..tifinagh import EMPHATIC_TO_PLAIN, LABIALIZATION, VOWELS_TFNG

COST_EMPHATIC = 0.3   # ⵜ / ⵟ, ⵙ / ⵚ…
COST_VOWEL_SUB = 0.6  # ⴰ / ⵉ / ⵓ
COST_VOWEL_INDEL = 0.4
COST_GEMINATION = 0.3
COST_OTHER = 1.0


def _sub_cost(x: str, y: str) -> float:
    if x == y:
        return 0.0
    if EMPHATIC_TO_PLAIN.get(x, x) == EMPHATIC_TO_PLAIN.get(y, y):
        return COST_EMPHATIC
    if x in VOWELS_TFNG and y in VOWELS_TFNG:
        return COST_VOWEL_SUB
    return COST_OTHER


def _indel_cost(s: str, j: int) -> float:
    c = s[j]
    if c in VOWELS_TFNG or c == LABIALIZATION:
        return COST_VOWEL_INDEL
    if (j > 0 and s[j - 1] == c) or (j + 1 < len(s) and s[j + 1] == c):
        return COST_GEMINATION
    return COST_OTHER


def weighted_distance(a: str, b: str) -> float:
    m = len(b)
    prev = [0.0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + _indel_cost(b, j - 1)
    for i in range(1, len(a) + 1):
        cur = [prev[0] + _indel_cost(a, i - 1)] + [0.0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + _indel_cost(a, i - 1),
                         cur[j - 1] + _indel_cost(b, j - 1),
                         prev[j - 1] + _sub_cost(a[i - 1], b[j - 1]))
        prev = cur
    return prev[m]
