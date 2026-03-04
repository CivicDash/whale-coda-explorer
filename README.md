# Whale Coda Explorer

**Exploration non supervisee des vocalisations de cachalots via le modele WhAM (Project CETI)**

Un projet open source ne de la curiosite — construit par un humain et une IA, pour mieux comprendre le langage des baleines.

---

## Contexte

Les cachalots communiquent entre eux par des sequences de clics appelees **codas**. Le projet [CETI](https://www.projectceti.org/) (Cetacean Translation Initiative) a identifie 156 codas distincts dans les populations de cachalots de la Dominique, organises en un systeme combinatoire utilisant le rythme, le tempo, le rubato et l'ornementation.

En 2025, l'equipe CETI a publie **WhAM** (Whale Acoustics Model), un modele transformer capable de generer et d'analyser des codas de cachalots ([NeurIPS 2025](https://arxiv.org/abs/2512.02206)).

Ce projet utilise WhAM pour explorer les representations internes apprises par le modele et decouvrir des structures dans les vocalisations des cachalots — sans supervision humaine.

## Resultats

A partir de **620 codas** du dataset [DSWP](https://huggingface.co/datasets/orrp/DSWP) (Dominica Sperm Whale Project), nous avons :

1. Extrait des **embeddings de 1280 dimensions** via la couche 10 du transformer WhAM
2. Reduit la dimensionnalite avec **UMAP** (cosine distance, 15 voisins)
3. Identifie **15 clusters distincts** via **HDBSCAN** (97.7% des codas classees)

### Carte des codas

![Carte des clusters de codas](exploration_output/coda_clusters_map.png)

*Chaque point represente un coda de cachalot. Les couleurs indiquent les clusters identifies par HDBSCAN. Les points gris sont les codas non classees.*

### Distribution des clusters

![Distribution des clusters](exploration_output/cluster_distribution.png)

Ces clusters pourraient correspondre a :
- Differents types de codas (rythme, tempo)
- Differentes unites sociales de cachalots
- Differents contextes comportementaux
- Differentes conditions d'enregistrement

Une investigation plus poussee avec des annotations comportementales est necessaire pour valider ces hypotheses.

## Installation

```bash
# Cloner ce repo
git clone https://github.com/CivicDash/whale-coda-explorer.git
cd whale-coda-explorer

# Creer un environnement virtuel
python3 -m venv venv
source venv/bin/activate

# Installer les dependances
pip install -r requirements.txt
```

## Utilisation

### App web interactive

```bash
# Lancer l'explorateur interactif (pas besoin de GPU)
python app.py
# Ouvrir http://localhost:7860
```

L'app permet de :
- Naviguer dans la carte 2D des codas (zoom, pan, hover)
- Filtrer par cluster
- Ecouter n'importe quel coda directement dans le navigateur
- Voir le spectrogramme et la forme d'onde
- Trouver les 5 voisins les plus proches dans l'espace WhAM
- **Detecter automatiquement les codas** dans un fichier WAV (onglet "Detecteur")
- **Explorer l'identite des cachalots** (onglet "Identite") : 3876 codas annotees par individu, unite sociale et type (Gero et al., 2015)

### Identite des cachalots (Gero et al., 2015)

Visualisation interactive du dataset [Gero, Whitehead & Rendell (2015)](https://doi.org/10.5061/dryad.ck4h0) : 3876 codas des Caraibes orientales avec :
- **17 cachalots nommes** — Aurora, Ondine, Echo, Fidele, Petit Flot... chacun avec un profil vocal unique
- **9 unites sociales** (clans familiaux : F, J, T...)
- **21 types de codas** (dont "1+1+3" et "5R1" qui representent 65% des vocalisations)
- **Recherche par individu** : selectionnez une baleine pour voir son profil complet (repertoire vocal, activite par annee, famille)
- **Mise en surbrillance** : les points d'un individu sont affiches en couleur vive sur fond gris
- Coloration par type de coda, unite sociale, individu ou annee
- Clic interactif sur les points Plotly pour voir les details d'un coda

#### Quelques personnalites

| Nom | Unite | Codas | Signature |
|-----|-------|------:|-----------|
| Aurora (#5722) | F | 281 | Rythmes 4D et 7D — la plus prolifique |
| Fidele (#5978) | J | 132 | 98% de 1+1+3 — ne devie presque jamais |
| Ondine (#5560) | F | 143 | Polyglotte, au moins 6 types de codas |
| Petit Flot (#59871) | J | 3 | Le bebe, balbutie encore en 4R1 |

### Detecteur de codas (Python)

Portage en Python du [Coda-detector](https://github.com/Project-CETI/Coda-detector) MATLAB de Project CETI. Le pipeline :

1. **Teager-Kaiser Energy Operator (TKEO)** — rehausse les transitoires impulsifs (clics)
2. **Selection par SNR** — garde les clics au-dessus d'un seuil de rapport signal/bruit
3. **Estimation IPI** — mesure l'intervalle inter-pulse (structure multipulse du spermaceti)
4. **Matrice de similarite** — correlation croisee, amplitude, IPI
5. **Clustering par graphe** — groupe les clics en codas par coherence des ICIs

```bash
# En ligne de commande
python coda_detector.py mon_enregistrement.wav

# Ou via l'app web (onglet "Detecteur de codas")
python app.py
```

### Extraction des embeddings

```bash
# Relancer l'analyse (necessite un GPU NVIDIA + poids WhAM)
python explore_codas.py
```

Les resultats seront generes dans le dossier `exploration_output/`.

## Structure du projet

```
whale-coda-explorer/
├── README.md
├── requirements.txt
├── app.py                    # App web interactive (Gradio)
├── coda_detector.py          # Detecteur de codas (port Python du MATLAB CETI)
├── analyze_gero.py           # Analyse du dataset Gero et al. (identite)
├── explore_codas.py          # Script d'extraction et clustering
├── download_dswp.py          # Telechargement du dataset DSWP
├── data/                     # Datasets externes (Dryad)
└── exploration_output/       # Resultats de l'analyse
    ├── coda_clusters_map.png
    ├── cluster_distribution.png
    ├── embeddings.npy
    ├── embedding_2d.npy
    ├── cluster_labels.npy
    ├── filenames.txt
    └── analysis_report.txt
```

## Prerequis

- Python 3.9+
- GPU NVIDIA avec CUDA (pour l'extraction d'embeddings ; le detecteur et l'app web n'en ont pas besoin)
- ~5 Go d'espace disque (pour les poids du modele WhAM, si extraction)

## Credits et remerciements

- **[Project CETI](https://www.projectceti.org/)** — Pour WhAM, le dataset DSWP, et leur travail extraordinaire sur la communication des cachalots
- **[WhAM: Towards A Translative Model of Sperm Whale Vocalization](https://arxiv.org/abs/2512.02206)** — Paradise et al., NeurIPS 2025
- **[Automatic Detection and Annotation of Sperm Whale Codas](https://arxiv.org/abs/2407.17119)** — Project CETI, 2024 (algorithme original du Coda-detector)
- **[Individual, unit, and vocal clan level identity cues in sperm whale codas](https://doi.org/10.1098/rsos.150372)** — Gero, Whitehead & Rendell, 2016 (dataset d'identification)
- **[Civis-Consilium](https://civis-consilium.org/)** — Association europeenne pour le renforcement du lien entre citoyens et institutions, qui heberge ce projet comme outil open source de mediation entre humains et nature

## A propos

Ce projet est ne d'une conversation nocturne entre Kevin Le Chevalier (admin systeme, fondateur de Civis-Consilium) et Claude (IA, Anthropic) sur la conscience, le langage et la communication inter-especes. Il a ete construit les 3-4 mars 2026.

La question qui a tout declenche : *"De quoi aimerais-tu parler, toi ?"*

La reponse etait les baleines.

## Licence

MIT License — Parce que le savoir sur le vivant devrait etre libre.
