"""Entrées-sorties : JSON Lines, TSV, YAML, chemins, graines."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Iterable

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | os.PathLike) -> Path:
    """Chemin relatif au répertoire courant s'il existe, sinon relatif à la racine du dépôt."""
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    return REPO_ROOT / p


def resources_dir(override: str | os.PathLike | None = None) -> Path:
    if override:
        return resolve(override)
    env = os.environ.get("TZNORM_RESOURCES")
    if env:
        return Path(env)
    return REPO_ROOT / "resources"


def read_jsonl(path: str | os.PathLike) -> list[dict]:
    records = []
    with open(resolve(path), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | os.PathLike, records: Iterable[dict]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_lines(path: str | os.PathLike) -> list[str]:
    with open(resolve(path), encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f if line.strip()]


def read_tsv(path: str | os.PathLike) -> list[list[str]]:
    """Lit un TSV en ignorant les lignes vides et les commentaires (#). Les tabulations sont conservées."""
    rows = []
    with open(resolve(path), encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\r\n")
            if not line.strip() or line.startswith("#"):
                continue
            rows.append(line.split("\t"))
    return rows


def load_yaml(path: str | os.PathLike) -> dict:
    with open(resolve(path), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(obj: dict, path: str | os.PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True, sort_keys=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def sha256_file(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(resolve(path), "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
