"""Lancement d'une évaluation : même code pour tous les systèmes (B0, B1, B2, modèle seul, pipeline complet)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .metrics import compute_all
from .phon import phon_analysis

CELL_KEYS = ("variete", "ecriture", "registre")
SYSTEMS = ("b0", "b1", "b2", "model", "pipeline")


def build_system(name: str, config: str | Path = "configs/pipeline.yaml"):
    from ..pipeline import Normalizer

    if name in ("b0", "b1", "b2"):
        return Normalizer.from_config(config, rules_level=int(name[1]), load_model=False)
    if name == "model":
        return Normalizer.from_config(config, use_reranker=False, fallback_rules=False)
    if name == "pipeline":
        return Normalizer.from_config(config)
    raise ValueError(f"système inconnu : {name} ({', '.join(SYSTEMS)})")


def run_system(normalizer, records: list[dict], oracle_variety: bool = True, batch_size: int = 32) -> list[str]:
    hyps: list[str] = []
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        varieties = [r.get("variete") if oracle_variety else None for r in batch]
        varieties = [v if v and v != "?" else None for v in varieties]
        res = normalizer.normalize_batch([r["entree"] for r in batch], varieties, None)
        hyps.extend(x.sortie for x in res)
    return hyps


def _group(records, hyps, key_fn) -> dict:
    groups: dict = defaultdict(lambda: ([], []))
    for rec, h in zip(records, hyps):
        g = groups[key_fn(rec)]
        g[0].append(h)
        g[1].append(rec["references"])
    return {k: compute_all(h, r) for k, (h, r) in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def evaluate(records: list[dict], hyps: list[str]) -> dict:
    refs = [r["references"] for r in records]
    result = {"global": compute_all(hyps, refs), "par": {}, "phonologie": phon_analysis(hyps, refs)}
    for key in CELL_KEYS:
        result["par"][key] = _group(records, hyps, lambda r, k=key: r.get(k, "?"))
    cells = _group(records, hyps, lambda r: " × ".join(str(r.get(k, "?")) for k in CELL_KEYS))
    result["par"]["cellule"] = cells
    if cells:
        cers = [m["cer"] for m in cells.values()]
        mean = result["global"]["cer"]
        result["equilibre"] = {"cer_min": min(cers), "cer_max": max(cers),
                               "pire_sur_moyenne": (max(cers) / mean) if mean else None}
    if any("oov" in r for r in records):
        result["par"]["oov"] = _group(records, hyps, lambda r: "hors_vocabulaire" if r.get("oov") else "connu")
    return result


def _table(rows: dict, title: str) -> list[str]:
    lines = [f"### {title}", "", "| Groupe | n | CER | WER | Exact match | chrF |", "|---|---|---|---|---|---|"]
    for k, m in rows.items():
        lines.append(f"| {k} | {m['n']} | {m['cer']:.2%} | {m['wer']:.2%} | {m['exact_match']:.2%} | {m['chrf']:.1f} |")
    return lines + [""]


def write_report(result: dict, out_dir: str | Path, system: str, data_name: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{system}_{data_name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    g = result["global"]
    lines = [f"# Évaluation : {system} sur {data_name}", "",
             f"- Phrases : {g['n']}", f"- **CER : {g['cer']:.2%}**", f"- WER : {g['wer']:.2%}",
             f"- Exact match : {g['exact_match']:.2%}", f"- chrF : {g['chrf']:.1f}", ""]
    if "equilibre" in result:
        e = result["equilibre"]
        ratio = f"{e['pire_sur_moyenne']:.2f}" if e["pire_sur_moyenne"] is not None else "n/a"
        lines += [f"Équilibre : CER par cellule de {e['cer_min']:.2%} à {e['cer_max']:.2%} "
                  f"(pire cellule / moyenne = {ratio})", ""]
    for key, rows in result["par"].items():
        lines += _table(rows, f"Par {key}")
    lines += ["### Par classe phonologique", "", "| Classe | n | Erreurs | Taux |", "|---|---|---|---|"]
    for cls, m in result["phonologie"].items():
        rate = f"{m['taux']:.2%}" if m["taux"] is not None else "n/a"
        lines.append(f"| {cls} | {m['n']} | {m['erreurs']} | {rate} |")
    path = out / f"{system}_{data_name}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
