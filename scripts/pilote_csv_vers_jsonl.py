"""Convertit le tableur d'annotation du pilote (CSV) au format JSON Lines du Gold, en contrôlant chaque ligne.

Colonnes attendues (voir data/real/pilote_modele.csv) :
  id, source, variete, ecriture, registre, entree, reference, reference_latin_ircam, reference_alt,
  ambigu, alternance_codes, annotateur, remarque

- `reference` : la forme normalisée en Tifinagh IRCAM.
- `reference_latin_ircam` : à défaut, la même forme en latin IRCAM avec diacritiques (ɣ ɛ ḥ ṭ ḍ ṣ ẓ ṛ ʷ, c pour ch,
  x pour kh) ; elle est convertie lettre à lettre en Tifinagh.
- `reference_alt` : autres sorties valides, séparées par « | » (cas ambigus).

Usage :
  python scripts/pilote_csv_vers_jsonl.py data/real/pilote.csv --out data/real/pilote.jsonl
"""

from __future__ import annotations

import argparse
import csv
import sys
import unicodedata

from tznorm.detect.script import text_script
from tznorm.io import write_jsonl
from tznorm.postprocess import is_valid_output
from tznorm.preprocess import preprocess
from tznorm.tifinagh import LAT2TFNG, latin_to_tfng

VARIETES = {"rif", "tzm", "shi", "?"}
ECRITURES = {"lat", "ara", "tfng"}
REGISTRES = {"formel", "informel"}
DIGRAMMES_INTERDITS = ("gh", "kh", "ch", "sh", "ou")
VRAI = {"1", "oui", "o", "true", "vrai", "x"}


def read_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        return list(csv.DictReader(f, dialect=dialect))


def latin_ircam_to_tfng(text: str) -> tuple[str, list[str]]:
    t = unicodedata.normalize("NFC", text.strip().lower())
    errors = [f"digramme « {d} » : utiliser la lettre IRCAM" for d in DIGRAMMES_INTERDITS if d in t]
    unknown = sorted({c for c in t if c.isalpha() and c not in LAT2TFNG})
    if unknown:
        errors.append(f"lettres hors latin IRCAM : {' '.join(unknown)}")
    return latin_to_tfng(t), errors


def convert_row(i: int, row: dict) -> tuple[dict | None, list[str]]:
    errs: list[str] = []
    get = lambda k: (row.get(k) or "").strip()  # noqa: E731
    entree = unicodedata.normalize("NFC", get("entree"))
    if not entree:
        return None, []  # ligne vide : ignorée

    ref = unicodedata.normalize("NFC", get("reference"))
    if not ref and get("reference_latin_ircam"):
        ref, conv_errs = latin_ircam_to_tfng(get("reference_latin_ircam"))
        errs += conv_errs
    if not ref:
        errs.append("référence manquante")
    refs = [ref] + [unicodedata.normalize("NFC", x.strip()) for x in get("reference_alt").split("|") if x.strip()]
    for r in refs:
        if r and not is_valid_output(preprocess(r).text):
            errs.append(f"référence avec des lettres non Tifinagh : {r!r}")

    variete = get("variete").lower() or "?"
    if variete not in VARIETES:
        errs.append(f"variété inconnue {variete!r} (rif, tzm, shi ou ?)")
    ecriture = get("ecriture").lower() or text_script(preprocess(entree).text)
    if ecriture not in ECRITURES:
        errs.append(f"écriture inconnue {ecriture!r} (lat, ara, tfng)")
    registre = get("registre").lower() or "informel"
    if registre not in REGISTRES:
        errs.append(f"registre inconnu {registre!r} (formel, informel)")

    rec = {
        "id": get("id") or f"pilote-{i:04d}",
        "source": get("source") or "inconnue",
        "variete": variete, "ecriture": ecriture, "registre": registre,
        "entree": entree, "references": refs,
        "ambigu": get("ambigu").lower() in VRAI or len(refs) > 1,
        "alternance_codes": get("alternance_codes").lower() in VRAI,
        "annotateurs": [x.strip() for x in get("annotateur").split(",") if x.strip()],
        "version_guide": "v0",
    }
    if get("remarque"):
        rec["remarque"] = get("remarque")
    return rec, errs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    records, n_err = [], 0
    for i, row in enumerate(read_csv(a.csv), start=2):  # ligne 1 = en-têtes
        rec, errs = convert_row(i, row)
        if errs:
            n_err += 1
            print(f"ligne {i} : " + " ; ".join(errs), file=sys.stderr)
        elif rec:
            records.append(rec)
    write_jsonl(a.out, records)
    print(f"{len(records)} phrases valides → {a.out} ; {n_err} ligne(s) à corriger")
    sys.exit(1 if n_err else 0)


if __name__ == "__main__":
    main()
