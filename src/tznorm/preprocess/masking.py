"""Masquage des éléments à conserver tels quels (URL, mentions, hashtags, emojis, nombres).

Chaque élément masqué est remplacé par le caractère ¤ ; le modèle apprend à le recopier, puis le
post-traitement réinsère les originaux dans l'ordre.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PLACEHOLDER = "¤"  # ¤

_PATTERNS = [
    ("placeholder", re.escape(PLACEHOLDER)),
    ("url", r"(?:https?://|www\.)\S+"),
    ("email", r"[\w.+-]+@[\w-]+\.[\w.]+"),
    ("mention", r"@\w+"),
    ("hashtag", r"#\w+"),
    ("emoji", r"[\U0001F000-\U0001FAFF☀-➿⬀-⯿️‍]+"),
    ("number", r"(?<!\w)\d+(?:[.,:/]\d+)*(?!\w)"),
]
_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _PATTERNS))


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str
    kind: str


def find_spans(text: str) -> list[Span]:
    return [Span(m.start(), m.end(), m.group(0), m.lastgroup or "?") for m in _RE.finditer(text)]


def mask(text: str, extra: list[Span] | None = None) -> tuple[str, list[str]]:
    """Remplace les éléments protégés par ¤. `extra` ajoute des segments (ex. phrases en français)."""
    spans = sorted(find_spans(text) + list(extra or []), key=lambda s: (s.start, -(s.end - s.start)))
    kept: list[Span] = []
    for s in spans:
        if kept and s.start < kept[-1].end:
            continue  # chevauchement : on garde le premier (le plus long à position égale)
        kept.append(s)
    out, originals, pos = [], [], 0
    for s in kept:
        out.append(text[pos:s.start])
        out.append(PLACEHOLDER)
        originals.append(text[s.start:s.end])
        pos = s.end
    out.append(text[pos:])
    return "".join(out), originals


def unmask(text: str, originals: list[str]) -> tuple[str, bool]:
    """Réinsère les originaux. Renvoie (texte, ok) ; ok est faux si le nombre de ¤ ne correspond pas."""
    parts = text.split(PLACEHOLDER)
    n_slots = len(parts) - 1
    ok = n_slots == len(originals)
    out = [parts[0]]
    for i, part in enumerate(parts[1:]):
        out.append(originals[i] if i < len(originals) else "")
        out.append(part)
    if n_slots < len(originals):
        out.append(" " + " ".join(originals[n_slots:]))
    return "".join(out), ok
