"""Génération du corpus synthétique (configs/synth.yaml)."""

from __future__ import annotations

import random
import unicodedata

from ..io import read_jsonl, read_lines, resolve, write_jsonl
from ..preprocess import PLACEHOLDER, clean
from .noise import Noiser
from .reverse import Renderer


def _pick(rng: random.Random, dist: dict):
    keys = list(dist)
    return rng.choices(keys, weights=[float(dist[k]) for k in keys], k=1)[0]


def load_clean_sources(sources: list[dict]) -> list[dict]:
    """Chaque source : {"path": ..., "variete": ...}. Renvoie [{"id", "texte", "variete"}]."""
    out = []
    for src in sources:
        path = resolve(src["path"])
        default_var = src.get("variete", "?")
        if path.suffix == ".jsonl":
            for i, rec in enumerate(read_jsonl(path)):
                out.append({"id": rec.get("id", f"{path.stem}-{i}"), "texte": rec["texte"],
                            "variete": rec.get("variete", default_var)})
        else:
            for i, line in enumerate(read_lines(path)):
                out.append({"id": f"{path.stem}-{i}", "texte": line, "variete": default_var})
    return out


def gold_targets(paths: list[str]) -> set[str]:
    targets = set()
    for p in paths or []:
        if resolve(p).exists():
            for rec in read_jsonl(p):
                for ref in rec.get("references", []):
                    targets.add(unicodedata.normalize("NFC", ref))
    return targets


def generate_synthetic(cfg: dict) -> dict:
    rng = random.Random(int(cfg.get("seed", 13)))
    renderer = Renderer(cfg, rng)
    noiser = Noiser(cfg.get("noise"), cfg.get("level_factor", (0.0, 0.5, 1.0, 2.0)), rng)
    exclude = gold_targets(cfg.get("gold_exclude", []))
    scripts = cfg.get("scripts", {"lat": 0.6, "ara": 0.25, "tfng": 0.15})
    conventions = cfg.get("conventions", {"lat": {"latin_ircam": 0.2, "latin_social": 0.8}})
    levels = {int(k): v for k, v in cfg.get("levels", {0: 0.15, 1: 0.35, 2: 0.35, 3: 0.15}).items()}
    variants = int(cfg.get("variants_per_sentence", 6))
    max_chars = int(cfg.get("max_chars", 400))
    placeholder_rate = float(cfg.get("placeholder_rate", 0.0))

    records, seen = [], set()
    stats = {"sources": 0, "exclues_gold": 0, "trop_longues": 0, "paires": 0}
    for src in load_clean_sources(cfg["sources"]):
        stats["sources"] += 1
        tgt = clean(src["texte"])
        if tgt in exclude:
            stats["exclues_gold"] += 1
            continue
        if len(tgt) > max_chars:
            stats["trop_longues"] += 1
            continue
        for k in range(variants):
            script = _pick(rng, scripts)
            conv = _pick(rng, conventions[script]) if script in conventions else script
            level = int(_pick(rng, levels))
            source = noiser.apply(renderer.render(tgt, script, conv, src["variete"]), script, level)
            target = tgt
            if placeholder_rate and rng.random() < placeholder_rate:
                if rng.random() < 0.5:
                    source, target = f"{PLACEHOLDER} {source}", f"{PLACEHOLDER} {target}"
                else:
                    source, target = f"{source} {PLACEHOLDER}", f"{target} {PLACEHOLDER}"
            if (source, target) in seen:
                continue
            seen.add((source, target))
            records.append({
                "id": f"syn-{src['id']}-{k}", "entree": source, "references": [target],
                "variete": src["variete"], "ecriture": script, "registre": "synthetique",
                "niveau_bruit": level, "convention": conv, "phrase_source_id": src["id"], "source": "synthetique",
            })
    stats["paires"] = write_jsonl(resolve(cfg["output"]), records)
    return stats
