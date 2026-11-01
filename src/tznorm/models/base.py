"""Interface commune des modèles séquence-à-séquence (M1, M2)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Seq2SeqModel(Protocol):
    def generate(self, sources: list[str], num_beams: int = 5, num_return: int = 5) -> list[list[tuple[str, float]]]:
        """Pour chaque source, les `num_return` meilleures hypothèses avec leur log-probabilité moyenne."""

    def score(self, sources: list[str], targets: list[str]) -> list[float]:
        """Log-probabilité moyenne par unité de chaque cible sachant sa source (décodage forcé)."""


def pick_device(device: str = "auto") -> str:
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def load_model(kind: str, path: str | Path, device: str = "auto", **kwargs) -> Seq2SeqModel:
    if kind == "byt5":
        from .byt5 import ByT5Normalizer

        return ByT5Normalizer.load(path, device=device, **kwargs)
    if kind == "char":
        from .char_transformer import CharTransformerNormalizer

        return CharTransformerNormalizer.load(path, device=device, **kwargs)
    raise ValueError(f"type de modèle inconnu : {kind!r} (byt5 ou char)")
