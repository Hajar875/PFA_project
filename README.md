# tznorm : normalisation du tamazight marocain vers le Tifinagh IRCAM

Convertit un texte tamazight marocain (tarifit, tamazight du Moyen Atlas, tachelhit), écrit en **latin**, en
**arabe** ou en **Tifinagh**, formel ou informel, vers une forme unique : le **Tifinagh IRCAM**. Le sens, le
lexique et la variété du texte sont conservés.

Ce dépôt implémente l'architecture décrite dans `Strategie_et_architecture_Tamazight.docx` :

| Sous-système | Code |
|---|---|
| A. Fabrique de données | `src/tznorm/synth/` (translittération inverse, bruiteur, calibration, génération) |
| B. Entraînement | `src/tznorm/models/` (ByT5-small M2, Transformer caractère M1), `src/tznorm/rules/` (baselines B0–B2) |
| C. Pipeline d'inférence | `src/tznorm/pipeline.py` (M0 `preprocess/`, M1 `detect/`, M2 modèle, M3 `rerank/`, M4 `postprocess.py`), `api/`, `demo/` |
| D. Banc d'évaluation | `src/tznorm/eval/` (CER, WER, exact match, chrF, classes phonologiques, bootstrap, rapports) |
| Ressources partagées | `resources/` (tables de correspondance, lexiques, guide d'annotation, journal des décisions) |

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (sous Linux/macOS : source .venv/bin/activate)
pip install -e ".[dev]"           # règles, évaluation, données synthétiques
pip install -e ".[neural]"        # + ByT5 et Transformer caractère (torch, transformers)
pip install -e ".[api,demo]"      # + API REST et démonstration
```

Le socle (règles, évaluation, fabrique de données) ne dépend que de PyYAML.

## Utilisation rapide

```python
from tznorm import Normalizer

norm = Normalizer.from_config("configs/pipeline.yaml")   # règles B2 seules tant qu'aucun modèle n'est configuré
res = norm("tanmirt s watas")
res.sortie       # 'ⵜⴰⵏⵎⵉⵔⵜ ⵙ ⵡⴰⵟⴰⵙ'
res.to_dict()    # entree, sortie, ecriture_detectee, variete, confiance, alternatives, source
```

```bash
tznorm normalize phrases.txt --json
uvicorn api.main:app --reload          # POST /normalize, POST /normalize/batch, GET /health
python demo/app.py                     # interface Gradio
```

## Déroulé du projet, phase par phase

Pour les trois premières semaines (téléchargement du Tifinagh propre, pilote, demandes d'accès aux données), suivre
`docs/guide_demarrage_donnees.md` pas à pas.

Les commandes suivent le plan de travail du document (P1 à P8).

**P1–P2 · Guide, pilote, baselines.** Annoter environ 200 phrases pilotes (`data/real/pilote.jsonl`, format de
`data/gold/README.md`), puis :

```bash
pytest                                                    # dont les 20 exemples du cadrage
tznorm eval --data data/real/pilote.jsonl --system b1
tznorm eval --data data/real/pilote.jsonl --system b2     # jalon 1 : baseline de référence
tznorm calibrate --pairs data/real/pilote.jsonl --out configs/noise_calibrated.yaml
```

Reporter les probabilités de `configs/noise_calibrated.yaml` dans `configs/synth.yaml`.

**P3 · Jeu Gold.** Une fois les 1 800 phrases annotées :

```bash
tznorm split-gold --data data/real/gold_annote.jsonl --out-dir data/gold    # jalon 2 : Gold figé
```

**P4 · Fabrique synthétique.** Placer le Tifinagh propre dans `data/interim/tifinagh_propre.jsonl`
(un objet `{"texte": ..., "variete": ...}` par ligne), puis :

```bash
tznorm synth --config configs/synth.yaml
tznorm check-leak --train data/synthetic/synth_v1.jsonl --gold data/gold/dev.jsonl data/gold/test.jsonl
tznorm build-lexicon --corpus data/interim/tifinagh_propre.jsonl --out resources/lexicon/standard.tsv
tznorm train-lm --corpus data/interim/tifinagh_propre.jsonl --out models/charlm.pkl
tznorm train-detector --variety-data data/real/train.jsonl --out models/detector.json
```

Déclarer ensuite le lexique, le LM et le détecteur dans `configs/pipeline.yaml`.

**P5 · Modèles.**

```bash
tznorm train-byt5 --phase A --config configs/byt5.yaml    # pré-entraînement synthétique (curriculum)
tznorm train-byt5 --phase B --config configs/byt5.yaml    # fine-tuning réel + rejeu
tznorm train-char --config configs/char_transformer.yaml  # modèle de contrôle M1
```

**P6 · Reclassement, évaluation, ablations.** Mettre `model.type: byt5` dans `configs/pipeline.yaml`, puis :

```bash
tznorm tune-rerank --dev data/gold/dev.jsonl --out models/rerank.json   # puis rerank.weights dans la config
tznorm eval --data data/gold/dev.jsonl --system model
tznorm eval --data data/gold/dev.jsonl --system pipeline
tznorm compare --data data/gold/dev.jsonl --hyps-a reports/pipeline_dev_hyps.jsonl --hyps-b reports/b2_dev_hyps.jsonl
tznorm eval --data data/gold/test.jsonl --system pipeline  # jalon 3 : une seule fois, à la fin
```

Les rapports (`reports/*.md` et `*.json`) donnent les scores globaux, par variété, par écriture, par registre,
par cellule, par classe phonologique et, si `tznorm mark-oov` a été lancé, hors vocabulaire.

## Formats

- **Entrée du modèle** : `<v:shi> <s:lat> texte`. `?` remplace une variété ou une écriture inconnue.
- **Éléments protégés** (URL, mentions, hashtags, emojis, nombres, segments dans une autre langue) : remplacés
  par `¤` avant le modèle, puis réinsérés dans l'ordre.
- **Paires** (Gold, réelles, synthétiques) : JSON Lines, voir `data/gold/README.md`.

## Points à valider

Les règles marquées « à valider » dans `resources/mapping/latin_informal.tsv` et les questions ouvertes de
`resources/guide/decisions.md` (finale ⵅ ou ⵖ, `rh`, affriquées du tarifit, `é`) doivent être tranchées avec des
locuteurs natifs. Les hyperparamètres des fichiers `configs/*.yaml` sont des valeurs de départ.
