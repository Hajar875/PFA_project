# Jeu d'évaluation Gold

Fichiers attendus (créés par `tznorm split-gold`) :

- `dev.jsonl` : environ 600 phrases, pour la sélection des modèles et le réglage du reclassement ;
- `test.jsonl` : environ 1 200 phrases, évaluées **une seule fois**, à la fin.

Le Gold est figé au jalon 2 et n'est jamais utilisé pour entraîner un modèle ni pour générer des données.

## Format (JSON Lines, une phrase par ligne)

```json
{
  "id": "gold-rif-lat-inf-0042",
  "source": "reseaux_sociaux",
  "variete": "rif",
  "ecriture": "lat",
  "registre": "informel",
  "entree": "thanmirth",
  "references": ["ⵜⴰⵏⵎⵉⵔⵜ"],
  "ambigu": false,
  "alternance_codes": false,
  "annotateurs": ["A1", "A2"],
  "version_guide": "v2"
}
```

- `variete` : `rif`, `tzm`, `shi` (ou `?`) ; `ecriture` : `lat`, `ara`, `tfng` ; `registre` : `formel`, `informel`.
- `references` : une ou plusieurs sorties valides ; le CER est calculé contre la plus proche.
- `oov` (ajouté par `tznorm mark-oov`) : la phrase contient un mot absent des données d'entraînement.
