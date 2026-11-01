# Guide de démarrage : les données (semaines 1 à 3)

Trois chantiers à mener **en parallèle** dès la première semaine :

| Chantier | Durée | Résultat attendu |
|---|---|---|
| 1. Télécharger le Tifinagh propre (OLDI Seed + FLORES+) | une demi-journée | `data/interim/tifinagh_propre.jsonl`, lexique, LM caractère |
| 2. Collecter et annoter le pilote | 2 à 3 semaines | `data/real/pilote.jsonl` (≈ 200 phrases), accord inter-annotateurs, bruit calibré |
| 3. Écrire aux auteurs (DATASHI, D-LexeCan, Awal) | 1 heure + relances | accès aux données, ou un refus documenté |

Commencez par le chantier 3 (il ne prend qu'une heure et les réponses sont lentes), puis le 1, puis le 2.

---

## Chantier 1 · Le Tifinagh propre

### Ce qu'il faut savoir (vérifié le 25/10/2026)

| Jeu | Sous-ensemble | Taille | Accès | Licence |
|---|---|---|---|---|
| OLDI Seed (`openlanguagedata/oldi_seed`) | `zgh_Tfng` | 6 193 phrases, split `train` | public | CC BY-SA 4.0 |
| FLORES+ (`openlanguagedata/flores_plus`) | `zgh_Tfng` | 997 phrases `dev` + 1 012 `devtest` | **compte + acceptation des conditions** | CC BY-SA 4.0 |

- Les corrections d'Öktem et al. (WMT 2025) sont **intégrées à OLDI Seed** : commit « Tamazight corrections » du
  3 novembre 2025, avec une fiche `dataset_cards/zgh_Tfng.md` qui les décrit.
- Pour **FLORES+**, la fiche du jeu ne mentionne pas ces corrections. Il faut donc vérifier (étape 1.5) que la version
  téléchargée est bien la version corrigée.
- **CC BY-SA 4.0** : citez les sources ; tout jeu dérivé que vous publiez (Gold, données synthétiques) doit être
  diffusé sous la même licence.

### 1.1 Créer le compte et accepter les conditions

1. Créez un compte sur <https://huggingface.co> (adresse universitaire de préférence).
2. Ouvrez <https://huggingface.co/datasets/openlanguagedata/flores_plus>, lisez les conditions et acceptez-les
   (bouton en haut de la page). C'est à vous de le faire : c'est un engagement personnel.
3. Créez un jeton d'accès : *Settings → Access Tokens → New token*, type **Read**. Ne le partagez pas et ne le mettez
   jamais dans le dépôt git.

### 1.2 Installer et se connecter

```bash
pip install -e ".[data]"
huggingface-cli login
```

Collez le jeton quand il est demandé. Sans terminal interactif, vous pouvez à la place définir la variable
d'environnement `HF_TOKEN` dans votre session.

### 1.3 Télécharger et nettoyer

```bash
python scripts/importer_corpus_tifinagh.py --reserver-devtest
```

Le script :
- télécharge `zgh_Tfng` d'OLDI Seed et de FLORES+ (dev et devtest) ;
- nettoie chaque phrase (NFC, caractères invisibles) ;
- écarte les phrases contenant moins de 90 % de lettres Tifinagh et les doublons ;
- écrit `data/interim/tifinagh_propre.jsonl` (champs `id`, `texte`, `variete: "?"`, `source`, `licence`) ;
- avec `--reserver-devtest`, met **FLORES+ devtest de côté** dans `data/interim/flores_devtest_reserve.jsonl`.

**Pourquoi réserver devtest ?** Ces 1 012 phrases propres, jamais utilisées pour générer des données, serviront à
construire les cellules « formel » du Gold. On demande à des locuteurs de les recopier en latin ou en arabe, et la cible
Tifinagh est déjà connue.

### 1.4 Contrôler le résultat

À la fin, le script affiche des compteurs (`lues`, `gardees`, `doublons`, `rejetees_pas_assez_de_tifinagh`, et la
répartition des dates `mise_a_jour`). Vérifiez :

- environ 6 000 phrases gardées pour OLDI Seed et environ 1 000 pour FLORES+ dev ;
- un nombre de phrases rejetées faible (quelques dizaines au plus). Sinon, ouvrez le fichier et regardez pourquoi
  (noms propres en latin, chiffres…) ;
- en ouvrant `data/interim/tifinagh_propre.jsonl` dans VS Code : les lettres doivent s'afficher en Tifinagh (police
  *Ebrima* sous Windows).

### 1.5 Vérifier que FLORES+ est la version corrigée

1. Dans le dépôt FLORES+ sur Hugging Face, onglet *Files*, ouvrez `dataset_cards/zgh_Tfng.md` et `CHANGELOG.md` :
   cherchez une mention des corrections de 2025 (Öktem, Awal, « corrections »).
2. Comparez avec les dates `mise_a_jour` affichées par le script.
3. Si rien n'indique la correction, posez la question aux auteurs (e-mail n° 3 ci-dessous). En attendant, utilisez
   FLORES+ normalement : les phrases non corrigées restent du Tifinagh valide dans leur grande majorité.

### 1.6 Produire les ressources qui en découlent

```bash
tznorm build-lexicon --corpus data/interim/tifinagh_propre.jsonl --variety "*" --min-freq 2 --out resources/lexicon/standard.tsv
tznorm train-lm --corpus data/interim/tifinagh_propre.jsonl --out models/charlm.pkl
```

Ajoutez ensuite dans `configs/pipeline.yaml` :

```yaml
lexicon:
  - resources/lexicon/lexique_exemple.tsv
  - resources/lexicon/standard.tsv
charlm: models/charlm.pkl
```

Vous pouvez déjà lancer une **première génération synthétique** pour tester la chaîne
(`tznorm synth`), mais la version qui servira à l'entraînement viendra après la calibration du chantier 2.

---

## Chantier 2 · Le pilote (≈ 200 phrases réelles annotées)

### 2.1 Objectif et quotas

Le pilote sert à trois choses : **mesurer le bruit réel** (pour calibrer le bruiteur), **roder le guide
d'annotation** et **évaluer la baseline B2** (jalon 1). Visez cette répartition :

| Variété | Latin informel | Arabe | Tifinagh | Formel (toutes écritures) | Total |
|---|---|---|---|---|---|
| Tarifit (`rif`) | 35 | 10 | 10 | 10 | 65 |
| Moyen Atlas (`tzm`) | 35 | 10 | 10 | 10 | 65 |
| Tachelhit (`shi`) | 35 | 10 | 10 | 15 | 70 |

Le latin informel domine parce que c'est ce qu'on trouve le plus en ligne. Les cellules arabe et Tifinagh viendront
surtout de l'élicitation (2.3).

### 2.2 Collecter des textes réels

**Où chercher :** pages et groupes publics Facebook régionaux, commentaires YouTube sous des chansons, des émissions ou
des actualités en tamazight, publications publiques sur X avec des mots-dièse comme `#tamazight` ou `#ⵜⴰⵎⴰⵣⵉⵖⵜ`.
Ciblez des contenus dont la région est claire (Rif, Moyen Atlas, Souss), pour attribuer la variété sans deviner.

**Règles :**
- copiez à la main de petites quantités de textes **publics** : pas de robot d'aspiration, pas de groupe privé, pas
  de message privé ;
- ne conservez **ni nom, ni pseudonyme, ni lien de profil, ni numéro**. Notez seulement le type de source
  (`facebook`, `youtube`, `x`, `elicitation`) ;
- gardez des phrases de 3 à 30 mots, majoritairement en tamazight ;
- une phrase entièrement en darija ou en français est exclue. Un emprunt isolé est gardé, avec `alternance_codes`
  à `oui` s'il y a un segment dans une autre langue ;
- si vous hésitez sur la variété, demandez à un locuteur ou mettez `?`. Ne devinez pas.

### 2.3 Élicitation (pour les cellules rares)

1. Recrutez 2 ou 3 locuteurs natifs par variété : camarades, famille, étudiants d'un département d'études
   amazighes. Expliquez le projet en deux phrases et demandez leur accord oral pour l'usage anonyme de leurs
   écrits.
2. Séance de 15 minutes : donnez-leur 10 phrases courtes de la vie courante (à l'oral, ou en français pour le sens),
   et demandez-leur de les écrire **« comme sur WhatsApp »** dans l'écriture imposée (latin, puis arabe, puis
   Tifinagh).
3. Variante pour le registre formel : donnez des phrases de `flores_devtest_reserve.jsonl` (en Tifinagh) et demandez
   de les recopier en latin ou en arabe. La référence est alors la phrase Tifinagh d'origine.
4. Notez `source: elicitation` dans le tableur.

### 2.4 Annoter

1. Copiez `data/real/pilote_modele.csv` en `data/real/pilote.csv` et ouvrez-le dans **Google Sheets** ou
   **LibreOffice Calc**. Excel affiche parfois mal le Tifinagh à l'ouverture d'un CSV : si vous l'utilisez, passez par
   *Données → À partir d'un fichier texte*, encodage UTF-8.
2. Remplissez une ligne par phrase :

| Colonne | Contenu |
|---|---|
| `id` | identifiant unique, par ex. `pilote-0042` |
| `source` | `facebook`, `youtube`, `x`, `elicitation`… |
| `variete` | `rif`, `tzm`, `shi` ou `?` |
| `ecriture` | `lat`, `ara`, `tfng` (vide = détection automatique) |
| `registre` | `formel` ou `informel` |
| `entree` | le texte **tel qu'il a été écrit**, sans rien corriger |
| `reference` | la forme normalisée en Tifinagh IRCAM |
| `reference_latin_ircam` | à défaut de clavier Tifinagh : la forme normalisée en latin IRCAM (ɣ ɛ ḥ ṭ ḍ ṣ ẓ ṛ, `c` pour ch, `x` pour kh), convertie automatiquement |
| `reference_alt` | autres sorties valides, séparées par une barre verticale (`a \| b`) |
| `ambigu`, `alternance_codes` | `oui` / `non` |
| `annotateur` | initiales (A1, A2…) |
| `remarque` | toute hésitation : elle alimentera le guide |

3. **Clavier Tifinagh :** sous Windows, *Paramètres → Heure et langue → Langue et région → Ajouter une langue*,
   cherchez « Tamazight » et ajoutez un clavier Tifinagh ; basculez avec `Windows + Espace`. Sur téléphone, Gboard
   propose aussi un clavier Tifinagh.
4. Appliquez `resources/guide/guide_annotation.md`. Chaque hésitation va dans `remarque`, puis dans
   `resources/guide/decisions.md` une fois tranchée.
5. Exportez en CSV UTF-8 (séparateur `;`), puis :

```bash
python scripts/pilote_csv_vers_jsonl.py data/real/pilote.csv --out data/real/pilote.jsonl
```

Le script refuse les lignes problématiques et indique leur numéro : référence manquante, lettres non Tifinagh dans la
référence, `gh` ou `ch` dans la colonne latin IRCAM, variété ou écriture inconnue. Corrigez-les et relancez jusqu'à
« 0 ligne(s) à corriger ».

### 2.5 Double annotation et accord

1. Tirez 20 % des phrases (≈ 40), en équilibrant les variétés, et faites-les annoter **indépendamment** par un second
   annotateur (même tableur vide de références, mêmes `id`).
2. Convertissez les deux tableurs (`pilote_A1.jsonl`, `pilote_A2.jsonl`), puis :

```bash
python scripts/accord_annotateurs.py data/real/pilote_A1.jsonl data/real/pilote_A2.jsonl --desaccords data/real/desaccords.csv
```

3. Lecture : un CER inter-annotateurs de quelques pourcents est normal. Au-delà de 5 % environ, c'est en général une
   règle absente ou floue du guide : regardez `desaccords.csv`, tranchez avec les annotateurs et consignez chaque
   décision dans `decisions.md`.

### 2.6 Exploiter le pilote (jalon 1)

```bash
tznorm eval --data data/real/pilote.jsonl --system b1
tznorm eval --data data/real/pilote.jsonl --system b2
tznorm calibrate --pairs data/real/pilote.jsonl --out configs/noise_calibrated.yaml
```

- Les rapports `reports/b1_pilote.md` et `reports/b2_pilote.md` donnent votre **première référence chiffrée** et
  montrent les classes d'erreurs dominantes.
- Recopiez les valeurs de `configs/noise_calibrated.yaml` (sections `noise` et `latin_social`) dans
  `configs/synth.yaml`, puis régénérez les données synthétiques avec `tznorm synth`.

**Important :** le pilote sert à calibrer et à mesurer. Ses phrases ne doivent pas être réutilisées dans le Gold
final.

---

## Chantier 3 · Écrire aux auteurs

### 3.1 Qui contacter et pour quoi

| Jeu | Auteurs | Ce que vous demandez | Où trouver l'adresse |
|---|---|---|---|
| **DATASHI** (LREC 2026) | Nasser-Eddine Monir, Zakaria Baou | le sous-ensemble de 1 500 phrases standard / non standard, et sa licence | première page du PDF : arXiv `2603.21571` ou ACL Anthology `2026.lrec-1.153` |
| **D-LexeCan** (IJACSA 2026) | Amar Amakssoum, Fadwa Bouhafer, Anass El Haddadi, Abdelkhalak Bahri (ENSA d'Al Hoceima, Université Abdelmalek Essaâdi) | le corpus tarifit multi-écriture annoté | première page du PDF sur thesai.org (vol. 17, n° 3, article 72) |
| **Awal** (facultatif) | Alp Öktem, Farida Boudichat | les paires de la plateforme ; savoir si les corrections FLORES+ zgh sont publiées | site awaldigital.org, ou les PDF WMT 2025 / TICAM 2025 |

Au moment de la vérification, aucune page publique ne donnait de lien de téléchargement ni de licence pour DATASHI et
D-LexeCan : il faut donc les demander.

### 3.2 Conseils d'envoi

- Écrivez depuis votre **adresse universitaire** et mettez votre **encadrant en copie** : cela rassure sur l'usage
  qui sera fait des données.
- Un message court, une demande précise, un engagement clair : usage de recherche, pas de redistribution sans accord,
  citation de leur article.
- Sans réponse au bout de 10 jours, une relance courte. Ensuite, essayez un autre auteur de l'article ou un autre
  canal (ResearchGate, LinkedIn).
- Notez chaque envoi dans un petit tableau de suivi : date, destinataire, réponse, licence obtenue.

### 3.3 Modèles d'e-mails

Remplacez les crochets. Ces textes sont des propositions : adaptez le ton et relisez avant d'envoyer.

**E-mail 1 · DATASHI**

> **Objet :** Demande d'accès au corpus DATASHI (sous-ensemble standard / non standard) pour un projet de recherche
>
> Bonjour Monsieur Monir, bonjour Monsieur Baou,
>
> Je suis [prénom nom], étudiant(e) en [filière] à [établissement], et je réalise, sous la direction de
> [nom de l'encadrant·e] (en copie), un projet sur la normalisation automatique du tamazight marocain : convertir un
> texte écrit en latin, en arabe ou en Tifinagh, formel ou informel, vers le Tifinagh IRCAM.
>
> Votre article DATASHI (LREC 2026) est le travail le plus proche du nôtre. Le sous-ensemble de 1 500 phrases en
> versions standard et non standard nous serait très précieux comme **jeu de test externe pour le tachelhit**, afin de
> comparer nos résultats aux vôtres.
>
> Pourriez-vous nous indiquer si ce sous-ensemble est disponible, et sous quelle licence ? Nous l'utiliserions
> uniquement pour l'évaluation, sans le redistribuer, et nous citerions votre article dans toute publication.
>
> Je vous remercie par avance et reste à votre disposition pour toute précision.
>
> Cordialement,
> [prénom nom], [établissement], [adresse e-mail]

**E-mail 2 · D-LexeCan**

> **Objet :** Demande d'accès au corpus tarifit multi-écriture de D-LexeCan
>
> Bonjour Monsieur Amakssoum,
>
> Je suis [prénom nom], étudiant(e) en [filière] à [établissement]. Avec [nom de l'encadrant·e] (en copie), je
> travaille sur un normaliseur du tamazight marocain qui convertit les textes écrits en latin, en arabe ou en Tifinagh
> vers le Tifinagh IRCAM, pour le tarifit, le tamazight du Moyen Atlas et le tachelhit.
>
> Votre article D-LexeCan (IJACSA, 2026) décrit un corpus tarifit issu des réseaux sociaux, dans plusieurs écritures.
> C'est la ressource la plus proche de nos besoins pour le tarifit, qui est la variété la moins représentée dans les
> données disponibles.
>
> Serait-il possible d'accéder à ce corpus (les textes ; les annotations de sentiment ne sont pas nécessaires), et si
> oui à quelles conditions ? Il servirait à étudier les graphies réelles du tarifit et à évaluer notre système. Nous ne
> le redistribuerions pas et citerions votre article. Si cela vous intéresse, nous partagerions volontiers en retour
> nos versions normalisées en Tifinagh.
>
> En vous remerciant par avance,
> [prénom nom], [établissement], [adresse e-mail]

**E-mail 3 · Awal (facultatif)**

> **Objet :** Corrections Tamazight de FLORES+ et données Awal
>
> Bonjour Monsieur Öktem, bonjour Madame Boudichat,
>
> Je suis [prénom nom], étudiant(e) à [établissement]. Je travaille sur la normalisation du tamazight marocain vers le
> Tifinagh IRCAM et j'utilise la version corrigée d'OLDI Seed (zgh_Tfng) issue de votre article à WMT 2025 : merci
> pour ce travail.
>
> J'ai deux questions :
> 1. Les corrections de FLORES+ (dev et devtest) sont-elles intégrées au dépôt `openlanguagedata/flores_plus`, ou
>    publiées ailleurs ?
> 2. Les paires collectées sur la plateforme Awal sont-elles accessibles pour la recherche, et sous quelle licence ?
>
> Merci par avance pour votre réponse.
>
> Cordialement,
> [prénom nom], [établissement], [adresse e-mail]

---

## Liste de contrôle à la fin de la semaine 3

- [ ] E-mails envoyés (et relancés si besoin), tableau de suivi à jour
- [ ] `data/interim/tifinagh_propre.jsonl` créé, compteurs vérifiés, statut des corrections FLORES+ connu
- [ ] `resources/lexicon/standard.tsv` et `models/charlm.pkl` produits et déclarés dans `configs/pipeline.yaml`
- [ ] ≈ 200 phrases pilotes annotées, converties sans erreur
- [ ] Accord inter-annotateurs mesuré, désaccords arbitrés, `decisions.md` complété
- [ ] Rapports B1 et B2 sur le pilote (jalon 1)
- [ ] `configs/synth.yaml` mis à jour avec les valeurs calibrées

## Sources vérifiées

- FLORES+ : <https://huggingface.co/datasets/openlanguagedata/flores_plus>
- OLDI Seed : <https://huggingface.co/datasets/openlanguagedata/oldi_seed> (commit « Tamazight corrections » :
  <https://huggingface.co/datasets/openlanguagedata/oldi_seed/commit/2476cd0fc3dd6efe6f4333fbe4acf839697d2127>)
- Öktem et al., WMT 2025 : <https://aclanthology.org/2025.wmt-1.82>
- DATASHI : <https://arxiv.org/abs/2603.21571> et <https://aclanthology.org/2026.lrec-1.153/>
- D-LexeCan : <https://thesai.org/Downloads/Volume17No3/Paper_72-D_LexeCan_A_Dynamic_Lexicon_Based_Framework.pdf>
