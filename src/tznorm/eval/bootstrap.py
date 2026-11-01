"""Bootstrap apparié : la différence de CER entre deux systèmes est-elle significative ?"""

from __future__ import annotations

import random
from typing import Sequence

from .metrics import sentence_cer


def paired_bootstrap(hyps_a: Sequence[str], hyps_b: Sequence[str], refs: Sequence[Sequence[str]],
                     n_samples: int = 1000, seed: int = 13) -> dict:
    """delta = CER(A) - CER(B). Une valeur négative signifie que A est meilleur.
    p_value : part des rééchantillonnages où A n'est pas meilleur que B."""
    stats_a = [sentence_cer(h, r) for h, r in zip(hyps_a, refs)]
    stats_b = [sentence_cer(h, r) for h, r in zip(hyps_b, refs)]
    n = len(refs)

    def cer(stats, idx):
        e = sum(stats[i][0] for i in idx)
        length = sum(stats[i][1] for i in idx)
        return e / max(length, 1)

    full = list(range(n))
    delta = cer(stats_a, full) - cer(stats_b, full)
    rng = random.Random(seed)
    deltas = []
    for _ in range(n_samples):
        idx = [rng.randrange(n) for _ in range(n)]
        deltas.append(cer(stats_a, idx) - cer(stats_b, idx))
    deltas.sort()
    lo, hi = deltas[int(0.025 * n_samples)], deltas[min(int(0.975 * n_samples), n_samples - 1)]
    p_value = sum(d >= 0 for d in deltas) / n_samples
    return {"delta": delta, "ic95": [lo, hi], "p_value": p_value, "n": n}
