"""Ligne de commande `tznorm`. Chaque sous-commande correspond à une étape du plan de travail."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import load_yaml, read_jsonl, read_lines, save_yaml, write_jsonl


def _texts_from(path: str, field: str | None) -> list[str]:
    if path.endswith(".jsonl"):
        out = []
        for rec in read_jsonl(path):
            value = rec.get(field or "texte")
            out.extend(value if isinstance(value, list) else [value])
        return [t for t in out if t]
    return read_lines(path)


# --- inférence ---------------------------------------------------------------------------------------

def cmd_normalize(a):
    from .pipeline import Normalizer

    norm = Normalizer.from_config(a.config)
    fmt = lambda r: json.dumps(r.to_dict(), ensure_ascii=False) if a.json else r.sortie  # noqa: E731

    if not a.text and a.input in (None, "-") and sys.stdin.isatty():
        # Mode interactif : une phrase par ligne, résultat immédiat (sinon la lecture attendrait la fin de l'entrée)
        print("Tapez une phrase puis Entrée (ligne vide pour quitter).", file=sys.stderr)
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if not line.strip():
                break
            print(fmt(norm(line, a.variety, a.script)), flush=True)
        return

    if a.text:
        lines = a.text
    elif a.input in (None, "-"):
        lines = sys.stdin.read().splitlines()
    else:
        lines = read_lines(a.input)
    results = norm.normalize_batch(lines, [a.variety] * len(lines), [a.script] * len(lines))
    out = open(a.out, "w", encoding="utf-8") if a.out else sys.stdout
    try:
        for r in results:
            out.write(fmt(r) + "\n")
    finally:
        if a.out:
            out.close()


# --- évaluation --------------------------------------------------------------------------------------

def cmd_eval(a):
    from .eval.runner import build_system, evaluate, run_system, write_report

    records = read_jsonl(a.data)
    norm = build_system(a.system, a.config)
    hyps = run_system(norm, records, oracle_variety=not a.no_oracle_variety)
    name = Path(a.data).stem
    write_jsonl(Path(a.report_dir) / f"{a.system}_{name}_hyps.jsonl",
                [{"id": r.get("id"), "hyp": h} for r, h in zip(records, hyps)])
    path = write_report(evaluate(records, hyps), a.report_dir, a.system, name)
    print(f"rapport : {path}")


def cmd_compare(a):
    from .eval.bootstrap import paired_bootstrap

    records = read_jsonl(a.data)
    ha = [r["hyp"] for r in read_jsonl(a.hyps_a)]
    hb = [r["hyp"] for r in read_jsonl(a.hyps_b)]
    res = paired_bootstrap(ha, hb, [r["references"] for r in records], n_samples=a.samples)
    print(json.dumps(res, ensure_ascii=False, indent=2))


# --- données -----------------------------------------------------------------------------------------

def cmd_synth(a):
    from .synth import generate_synthetic

    print(json.dumps(generate_synthetic(load_yaml(a.config)), ensure_ascii=False, indent=2))


def cmd_calibrate(a):
    from .synth import calibrate

    res = calibrate(read_jsonl(a.pairs))
    save_yaml(res, a.out)
    print(f"{res['n_paires_latines']} paires latines analysées → {a.out}")


def cmd_split_gold(a):
    from .splits import stratified_split

    dev, test = stratified_split(read_jsonl(a.data), a.dev_ratio, a.seed)
    out = Path(a.out_dir)
    write_jsonl(out / "dev.jsonl", dev)
    write_jsonl(out / "test.jsonl", test)
    print(f"dev : {len(dev)} phrases, test : {len(test)} phrases → {out}")


def cmd_mark_oov(a):
    from .splits import mark_oov

    train = [r for p in a.train for r in read_jsonl(p)]
    gold = mark_oov(read_jsonl(a.gold), train)
    write_jsonl(a.out or a.gold, gold)
    print(f"{sum(r['oov'] for r in gold)} phrases hors vocabulaire sur {len(gold)}")


def cmd_check_leak(a):
    from .splits import check_leak

    train = [r for p in a.train for r in read_jsonl(p)]
    gold = [r for p in a.gold for r in read_jsonl(p)]
    print(json.dumps(check_leak(train, gold, a.n), ensure_ascii=False, indent=2))


# --- ressources --------------------------------------------------------------------------------------

def cmd_train_lm(a):
    from .rerank import CharNgramLM

    lm = CharNgramLM(a.order).fit(_texts_from(a.corpus, a.field))
    lm.save(a.out)
    print(f"LM caractère d'ordre {a.order} → {a.out}")


def cmd_build_lexicon(a):
    from .rerank import Lexicon

    lex = Lexicon.from_corpus(_texts_from(a.corpus, a.field), a.variety, a.min_freq)
    lex.save(a.out)
    print(f"{len(lex)} formes → {a.out}")


def cmd_train_detector(a):
    from .detect import Detector, NgramNB

    variety_clf = lang_clf = None
    if a.variety_data:
        recs = [r for r in read_jsonl(a.variety_data) if r.get("variete") not in (None, "?")]
        variety_clf = NgramNB().fit([r[a.text_field] for r in recs], [r["variete"] for r in recs])
    if a.lang_data:
        recs = read_jsonl(a.lang_data)
        lang_clf = NgramNB().fit([r["texte"] for r in recs], [r["langue"] for r in recs])
    Detector(variety_clf, lang_clf).save(a.out)
    print(f"détecteur → {a.out}")


# --- modèles -----------------------------------------------------------------------------------------

def cmd_train_byt5(a):
    from .models.byt5 import train_byt5

    print(f"modèle → {train_byt5(load_yaml(a.config), a.phase)}")


def cmd_train_char(a):
    from .models.char_transformer import train_char

    print(f"modèle → {train_char(load_yaml(a.config))}")


def cmd_tune_rerank(a):
    from .pipeline import Normalizer
    from .preprocess import preprocess
    from .rerank import TuneItem

    norm = Normalizer.from_config(a.config)
    if norm.model is None:
        sys.exit("tune-rerank demande un modèle (section model.type de la configuration)")
    records = read_jsonl(a.dev)
    items = []
    for i in range(0, len(records), 32):
        batch = records[i:i + 32]
        preps = norm.prepare([r["entree"] for r in batch], [r.get("variete") for r in batch])
        for rec, p, cands in zip(batch, preps, norm.candidates(preps)):
            refs = [preprocess(x).text for x in rec["references"]]
            items.append(TuneItem(cands, refs, p.rules_out))
    weights, cer = norm.reranker.tune(items)
    weights.save(a.out)
    print(f"poids {weights} (CER dev {cer:.2%}) → {a.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="tznorm", description="Normalisation du tamazight marocain vers le Tifinagh IRCAM")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("normalize", help="normaliser du texte (argument, fichier, entrée standard ou mode interactif)")
    p.add_argument("input", nargs="?", help="fichier texte (une phrase par ligne), ou - pour l'entrée standard ; "
                                            "sans argument dans un terminal : mode interactif")
    p.add_argument("-t", "--text", action="append", help="phrase à normaliser (option répétable)")
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--variety", choices=["rif", "tzm", "shi"])
    p.add_argument("--script", choices=["lat", "ara", "tfng"])
    p.add_argument("--out")
    p.add_argument("--json", action="store_true", help="sortie JSON détaillée")
    p.set_defaults(fn=cmd_normalize)

    p = sub.add_parser("eval", help="évaluer un système sur un jeu Gold")
    p.add_argument("--data", required=True)
    p.add_argument("--system", required=True, choices=["b0", "b1", "b2", "model", "pipeline"])
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--report-dir", default="reports")
    p.add_argument("--no-oracle-variety", action="store_true", help="ne pas fournir la variété du Gold au système")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("compare", help="bootstrap apparié entre deux fichiers d'hypothèses")
    p.add_argument("--data", required=True)
    p.add_argument("--hyps-a", required=True)
    p.add_argument("--hyps-b", required=True)
    p.add_argument("--samples", type=int, default=1000)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("synth", help="générer les paires synthétiques")
    p.add_argument("--config", default="configs/synth.yaml")
    p.set_defaults(fn=cmd_synth)

    p = sub.add_parser("calibrate", help="estimer les probabilités de bruit sur des paires réelles")
    p.add_argument("--pairs", required=True)
    p.add_argument("--out", default="configs/noise_calibrated.yaml")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("split-gold", help="découper le Gold en dev / test stratifiés")
    p.add_argument("--data", required=True)
    p.add_argument("--out-dir", default="data/gold")
    p.add_argument("--dev-ratio", type=float, default=1 / 3)
    p.add_argument("--seed", type=int, default=13)
    p.set_defaults(fn=cmd_split_gold)

    p = sub.add_parser("mark-oov", help="marquer les phrases du Gold hors vocabulaire")
    p.add_argument("--gold", required=True)
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_mark_oov)

    p = sub.add_parser("check-leak", help="contrôler les recouvrements entraînement / Gold")
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--gold", nargs="+", required=True)
    p.add_argument("--n", type=int, default=4)
    p.set_defaults(fn=cmd_check_leak)

    p = sub.add_parser("train-lm", help="entraîner le LM caractère du Tifinagh")
    p.add_argument("--corpus", required=True, help=".txt ou .jsonl")
    p.add_argument("--field", default="texte", help="champ à lire dans un .jsonl (ex. references)")
    p.add_argument("--order", type=int, default=6)
    p.add_argument("--out", default="models/charlm.pkl")
    p.set_defaults(fn=cmd_train_lm)

    p = sub.add_parser("build-lexicon", help="construire un lexique à partir d'un corpus Tifinagh")
    p.add_argument("--corpus", required=True)
    p.add_argument("--field", default="texte")
    p.add_argument("--variety", default="*")
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_build_lexicon)

    p = sub.add_parser("train-detector", help="entraîner les classifieurs de variété et de langue")
    p.add_argument("--variety-data", help="jsonl avec un champ variete")
    p.add_argument("--text-field", default="entree")
    p.add_argument("--lang-data", help="jsonl avec les champs texte et langue (zgh, ary, fr…)")
    p.add_argument("--out", default="models/detector.json")
    p.set_defaults(fn=cmd_train_detector)

    p = sub.add_parser("train-byt5", help="entraîner ByT5 (phase A ou B)")
    p.add_argument("--config", default="configs/byt5.yaml")
    p.add_argument("--phase", choices=["A", "B"], required=True)
    p.set_defaults(fn=cmd_train_byt5)

    p = sub.add_parser("train-char", help="entraîner le Transformer caractère")
    p.add_argument("--config", default="configs/char_transformer.yaml")
    p.set_defaults(fn=cmd_train_char)

    p = sub.add_parser("tune-rerank", help="régler les poids et le seuil du reclassement sur le Gold dev")
    p.add_argument("--config", default="configs/pipeline.yaml")
    p.add_argument("--dev", default="data/gold/dev.jsonl")
    p.add_argument("--out", default="models/rerank.json")
    p.set_defaults(fn=cmd_tune_rerank)

    # Sous Windows, une sortie redirigée vers un fichier utilise sinon cp1252, qui ne connaît pas le Tifinagh.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
