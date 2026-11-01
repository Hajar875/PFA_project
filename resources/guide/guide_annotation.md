# Guide d'annotation (version 0, à compléter)

Ce guide fixe la sortie attendue pour chaque entrée. En cas de doute, il fait foi ; chaque décision nouvelle est
consignée dans `decisions.md`, puis reportée ici avec un exemple.

## 1. Cible

- Écriture : Tifinagh IRCAM (bloc Unicode U+2D30 à U+2D7F), forme NFC, sans caractère invisible.
- Les mots sont séparés par des espaces ; ponctuation et chiffres sont conservés tels quels.
- Le texte garde **sa variété** (tarifit, tamazight du Moyen Atlas, tachelhit) : on unifie l'écriture et
  l'orthographe, pas le lexique.

## 2. Ce qui se normalise / ce qui se conserve

| Phénomène | Décision | Exemple |
|---|---|---|
| Spirantisation notée (`th`, `dh`) | normalisée | `thamazighth` → ⵜⴰⵎⴰⵣⵉⵖⵜ |
| Schwa écrit (`e`) | supprimé | `fellawen` → ⴼⵍⵍⴰⵡⵏ |
| Voyelles omises | restituées | `ur ssnx` → ⵓⵔ ⵙⵙⵉⵏⴰⵅ (finale ⵅ ou ⵖ : à trancher) |
| Emphatique sans point | restituée si le mot est connu | `watas` → ⵡⴰⵟⴰⵙ |
| Forme lexicale propre à la variété | conservée | `aryaz` (tarifit) → ⴰⵔⵢⴰⵣ, pas ⴰⵔⴳⴰⵣ |
| Répétition expressive | réduite | `azuuuul` → ⴰⵣⵓⵍ |

## 3. Correspondances

La table de référence est dans `resources/mapping/` (latin informel et écriture arabe) et dans
`src/tznorm/tifinagh.py` (latin canonique ↔ Tifinagh).

Points ouverts :
- affriquées du tarifit (`tch`, `dj`) ;
- `rh` pour ⵖ (cité par le cadrage, mais peut noter r + h) ;
- `é` : ⵉ ou schwa ?

## 4. Emprunts et alternance de codes

- Emprunt intégré (mot isolé dans une phrase tamazight) : transcrit en Tifinagh.
- Segment entier en français ou en darija : conservé tel quel.
- URL, mentions, hashtags, emojis, nombres : conservés tels quels.

## 5. Ambiguïtés

Si plusieurs sorties sont valides, les noter toutes dans `references` et mettre `"ambigu": true`.

## 6. Procédure

1. Deux annotateurs par variété annotent indépendamment 20 % des phrases.
2. On mesure l'accord par le CER entre les deux annotations (`tznorm eval` sur les deux fichiers).
3. Les désaccords sont arbitrés ; chaque arbitrage devient une ligne de `decisions.md`.
