"""Métriques : CER, WER, exact match, chrF. Références multiples : on retient la plus proche."""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def best_ref(hyp: str, refs: Sequence[str], unit: str = "char") -> tuple[str, int]:
    """Référence la plus proche de l'hypothèse et distance correspondante."""
    h = norm(hyp)
    best, best_d = "", None
    for r in refs:
        r = norm(r)
        d = levenshtein(h, r) if unit == "char" else levenshtein(h.split(), r.split())
        if best_d is None or d < best_d:
            best, best_d = r, d
    return best, best_d or 0


def sentence_cer(hyp: str, refs: Sequence[str]) -> tuple[int, int]:
    """(nombre d'éditions, longueur de la référence retenue)."""
    r, d = best_ref(hyp, refs)
    return d, len(r)


def corpus_cer(hyps: Sequence[str], refs: Sequence[Sequence[str]]) -> float:
    edits = length = 0
    for h, rs in zip(hyps, refs):
        d, n = sentence_cer(h, rs)
        edits += d
        length += n
    return edits / max(length, 1)


def corpus_wer(hyps: Sequence[str], refs: Sequence[Sequence[str]]) -> float:
    edits = length = 0
    for h, rs in zip(hyps, refs):
        r, d = best_ref(h, rs, unit="word")
        edits += d
        length += len(r.split())
    return edits / max(length, 1)


def exact_match(hyps: Sequence[str], refs: Sequence[Sequence[str]]) -> float:
    if not hyps:
        return 0.0
    return sum(norm(h) in {norm(r) for r in rs} for h, rs in zip(hyps, refs)) / len(hyps)


def chrf(hyps: Sequence[str], refs: Sequence[Sequence[str]], n: int = 6, beta: float = 2.0) -> float:
    """chrF au niveau du corpus (espaces ignorés), sur 0-100."""
    match = [0] * n
    hyp_tot = [0] * n
    ref_tot = [0] * n
    for h, rs in zip(hyps, refs):
        r, _ = best_ref(h, rs)
        hs, rss = norm(h).replace(" ", ""), r.replace(" ", "")
        for k in range(1, n + 1):
            hc = Counter(hs[i:i + k] for i in range(len(hs) - k + 1))
            rc = Counter(rss[i:i + k] for i in range(len(rss) - k + 1))
            match[k - 1] += sum((hc & rc).values())
            hyp_tot[k - 1] += sum(hc.values())
            ref_tot[k - 1] += sum(rc.values())
    precs = [m / t for m, t in zip(match, hyp_tot) if t]
    recs = [m / t for m, t in zip(match, ref_tot) if t]
    if not precs or not recs:
        return 0.0
    p, r = sum(precs) / len(precs), sum(recs) / len(recs)
    if p + r == 0:
        return 0.0
    b2 = beta ** 2
    return 100 * (1 + b2) * p * r / (b2 * p + r)


def compute_all(hyps: Sequence[str], refs: Sequence[Sequence[str]]) -> dict:
    return {
        "n": len(hyps),
        "cer": corpus_cer(hyps, refs),
        "wer": corpus_wer(hyps, refs),
        "exact_match": exact_match(hyps, refs),
        "chrf": chrf(hyps, refs),
    }
