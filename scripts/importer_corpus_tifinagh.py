"""Télécharge le Tifinagh standard d'OLDI Seed et de FLORES+ (sous-ensemble zgh_Tfng) vers data/interim/.

Prérequis, à faire vous-même une seule fois :
  1. créer un compte sur https://huggingface.co ;
  2. ouvrir https://huggingface.co/datasets/openlanguagedata/flores_plus et accepter les conditions d'accès ;
  3. pip install -e ".[data]"   puis   huggingface-cli login   (ou définir la variable HF_TOKEN).

Usage :
  python scripts/importer_corpus_tifinagh.py
  python scripts/importer_corpus_tifinagh.py --reserver-devtest   # garde FLORES+ devtest pour le Gold formel

Les deux jeux sont sous licence CC BY-SA 4.0 : citer les sources et redistribuer sous la même licence.
"""

from __future__ import annotations

import argparse
from collections import Counter

from datasets import load_dataset

from tznorm.io import REPO_ROOT, write_jsonl
from tznorm.preprocess import clean

LICENCE = "CC BY-SA 4.0"


def tifinagh_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum("ⴰ" <= c <= "⵿" for c in letters) / len(letters)


def rows(dataset: str, split: str, prefix: str):
    ds = load_dataset(dataset, "zgh_Tfng", split=split)
    for r in ds:
        yield f"{prefix}-{r['id']}", r["text"], r.get("last_updated")


def to_records(source_rows, source: str, min_ratio: float, seen: set, stats: Counter) -> list[dict]:
    out = []
    for rid, text, updated in source_rows:
        stats[f"{source}:lues"] += 1
        stats[f"{source}:mise_a_jour={updated}"] += 1
        t = clean(text or "")
        if not t:
            continue
        if tifinagh_ratio(t) < min_ratio:
            stats[f"{source}:rejetees_pas_assez_de_tifinagh"] += 1
            continue
        if t in seen:
            stats[f"{source}:doublons"] += 1
            continue
        seen.add(t)
        out.append({"id": rid, "texte": t, "variete": "?", "registre": "formel", "source": source,
                    "licence": LICENCE})
    stats[f"{source}:gardees"] += len(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO_ROOT / "data/interim/tifinagh_propre.jsonl"))
    ap.add_argument("--reserver-devtest", action="store_true",
                    help="écrire FLORES+ devtest à part (candidats pour les cellules « Tifinagh formel » du Gold)")
    ap.add_argument("--reserve", default=str(REPO_ROOT / "data/interim/flores_devtest_reserve.jsonl"))
    ap.add_argument("--min-tifinagh", type=float, default=0.9,
                    help="part minimale de lettres Tifinagh pour garder une phrase")
    a = ap.parse_args()

    seen: set[str] = set()
    stats: Counter = Counter()
    clean_records = to_records(rows("openlanguagedata/oldi_seed", "train", "seed"), "oldi_seed",
                               a.min_tifinagh, seen, stats)
    clean_records += to_records(rows("openlanguagedata/flores_plus", "dev", "flores-dev"), "flores_dev",
                                a.min_tifinagh, seen, stats)
    devtest = to_records(rows("openlanguagedata/flores_plus", "devtest", "flores-devtest"), "flores_devtest",
                         a.min_tifinagh, seen, stats)
    if a.reserver_devtest:
        write_jsonl(a.reserve, devtest)
        print(f"FLORES+ devtest réservé : {len(devtest)} phrases → {a.reserve}")
    else:
        clean_records += devtest

    n = write_jsonl(a.out, clean_records)
    print(f"{n} phrases de Tifinagh propre → {a.out}\n")
    for key in sorted(stats):
        print(f"  {key:60s} {stats[key]}")
    print("\nVérifiez les dates « mise_a_jour » : les corrections d'Öktem et al. (WMT 2025) datent de fin 2025.")


if __name__ == "__main__":
    main()
