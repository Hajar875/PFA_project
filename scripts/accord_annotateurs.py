"""Accord inter-annotateurs sur le recouvrement (20 % du pilote) : CER entre les deux annotations,
par variété, et liste des désaccords à arbitrer.

Les deux fichiers sont des JSON Lines produits par pilote_csv_vers_jsonl.py, avec les mêmes `id`.

Usage :
  python scripts/accord_annotateurs.py data/real/pilote_A1.jsonl data/real/pilote_A2.jsonl \
      --desaccords data/real/desaccords.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict

from tznorm.eval.metrics import corpus_cer, norm
from tznorm.io import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("annotateur_a")
    ap.add_argument("annotateur_b")
    ap.add_argument("--desaccords", default="data/real/desaccords.csv")
    a = ap.parse_args()

    b_by_id = {r["id"]: r for r in read_jsonl(a.annotateur_b)}
    pairs = [(r, b_by_id[r["id"]]) for r in read_jsonl(a.annotateur_a) if r["id"] in b_by_id]
    if not pairs:
        raise SystemExit("aucun identifiant commun entre les deux fichiers")

    groups = defaultdict(list)
    for ra, rb in pairs:
        groups["global"].append((ra, rb))
        groups[ra.get("variete", "?")].append((ra, rb))
    print(f"{len(pairs)} phrases annotées deux fois\n")
    print(f"{'groupe':10s} {'n':>5s} {'CER':>8s} {'identiques':>11s}")
    for name, items in groups.items():
        hyps = [ra["references"][0] for ra, _ in items]
        refs = [rb["references"] for _, rb in items]
        same = sum(norm(h) in {norm(x) for x in r} for h, r in zip(hyps, refs)) / len(items)
        print(f"{name:10s} {len(items):5d} {corpus_cer(hyps, refs):8.2%} {same:11.0%}")

    disagreements = [(ra, rb) for ra, rb in pairs
                     if norm(ra["references"][0]) not in {norm(x) for x in rb["references"]}]
    with open(a.desaccords, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["id", "variete", "entree", "annotateur_a", "annotateur_b", "decision", "regle_du_guide"])
        for ra, rb in disagreements:
            w.writerow([ra["id"], ra.get("variete", "?"), ra["entree"], ra["references"][0], rb["references"][0], "", ""])
    print(f"\n{len(disagreements)} désaccord(s) à arbitrer → {a.desaccords}")
    print("Chaque arbitrage devient une ligne de resources/guide/decisions.md.")


if __name__ == "__main__":
    main()
