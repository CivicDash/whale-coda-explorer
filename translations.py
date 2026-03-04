"""
Internationalisation (i18n) pour Whale Coda Explorer.
Dictionnaire bilingue FR/EN et fonction de traduction t().
"""

STRINGS = {
    # ── Header ──
    "header.subtitle": {
        "fr": "Exploration interactive des vocalisations de cachalots via WhAM (Project CETI)",
        "en": "Interactive exploration of sperm whale vocalizations via WhAM (Project CETI)",
    },
    "header.stats": {
        "fr": "620 codas analysees · 15 clusters · Detecteur de codas integre",
        "en": "620 codas analyzed · 15 clusters · Built-in coda detector",
    },

    # ── Tabs ──
    "tab.explorer": {"fr": "Explorer les clusters", "en": "Explore clusters"},
    "tab.identity": {"fr": "Identite des cachalots", "en": "Whale identity"},
    "tab.detector": {"fr": "Detecteur de codas", "en": "Coda detector"},
    "tab.identify": {"fr": "Identifier un coda", "en": "Identify a coda"},
    "tab.vocal": {"fr": "Analyse vocale", "en": "Vocal analysis"},
    "tab.study": {"fr": "Zone d'etude", "en": "Study area"},
    "tab.guide": {"fr": "Guide & Glossaire", "en": "Guide & Glossary"},

    # ── Common data labels ──
    "data.noise": {"fr": "Bruit", "en": "Noise"},
    "data.noise_full": {"fr": "Bruit (non classe)", "en": "Noise (unclassified)"},
    "data.cluster": {"fr": "Cluster {cid}", "en": "Cluster {cid}"},
    "data.all": {"fr": "Tous", "en": "All"},
    "data.total": {"fr": "TOTAL", "en": "TOTAL"},
    "data.codas": {"fr": "Codas", "en": "Codas"},
    "data.percentage": {"fr": "Pourcentage", "en": "Percentage"},
    "data.error": {"fr": "Erreur: {e}", "en": "Error: {e}"},
    "data.others": {"fr": "Autres", "en": "Others"},

    # ── Explorer tab ──
    "explorer.filter_label": {"fr": "Filtrer par cluster", "en": "Filter by cluster"},
    "explorer.overview_title": {"fr": "### Vue d'ensemble", "en": "### Overview"},
    "explorer.total_codas": {"fr": "Total codas analysees", "en": "Total codas analyzed"},
    "explorer.clusters_found": {"fr": "Clusters decouverts", "en": "Clusters discovered"},
    "explorer.unclassified": {"fr": "Non classes", "en": "Unclassified"},
    "explorer.click_hint": {
        "fr": "Cliquez sur un point de la carte pour ecouter le coda.",
        "en": "Click on a point on the map to listen to the coda.",
    },
    "explorer.distribution": {"fr": "Distribution", "en": "Distribution"},
    "explorer.listen_title": {
        "fr": "### Ecouter et analyser un coda",
        "en": "### Listen to and analyze a coda",
    },
    "explorer.coda_index_label": {"fr": "Index du coda (0-619)", "en": "Coda index (0-619)"},
    "explorer.coda_index_placeholder": {
        "fr": "Entrez un index ou cliquez sur la carte...",
        "en": "Enter an index or click on the map...",
    },
    "explorer.random_btn": {"fr": "Coda aleatoire", "en": "Random coda"},
    "explorer.load_btn": {"fr": "Charger", "en": "Load"},
    "explorer.listen_label": {"fr": "Ecouter le coda", "en": "Listen to coda"},
    "explorer.spectrogram_label": {"fr": "Spectrogramme", "en": "Spectrogram"},
    "explorer.click_placeholder": {
        "fr": "Cliquez sur un point de la carte ou sur **Charger**.",
        "en": "Click on a point on the map or press **Load**.",
    },

    # ── Coda info ──
    "info.cluster": {"fr": "Cluster", "en": "Cluster"},
    "info.index": {"fr": "Index", "en": "Index"},
    "info.file": {"fr": "Fichier", "en": "File"},
    "info.umap_pos": {"fr": "Position UMAP", "en": "UMAP position"},
    "info.nearest_codas": {
        "fr": "**5 codas les plus proches** (espace WhAM):",
        "en": "**5 nearest codas** (WhAM space):",
    },
    "info.out_of_bounds": {
        "fr": "Index {idx} hors limites.",
        "en": "Index {idx} out of bounds.",
    },
    "info.out_of_bounds_range": {
        "fr": "Index {idx} hors limites (0-{max})",
        "en": "Index {idx} out of bounds (0-{max})",
    },
    "info.file_not_found": {
        "fr": "Fichier introuvable: {path}",
        "en": "File not found: {path}",
    },
    "info.click_point": {
        "fr": "Cliquez sur un point de la carte.",
        "en": "Click on a point on the map.",
    },

    # ── Cluster summary ──
    "summary.unclassified": {
        "fr": "Points non classes (bruit)",
        "en": "Unclassified points (noise)",
    },
    "summary.n_codas": {"fr": "Nombre de codas", "en": "Number of codas"},
    "summary.proportion": {"fr": "Proportion", "en": "Proportion"},
    "summary.spread": {"fr": "Dispersion", "en": "Spread"},
    "summary.umap_extent": {"fr": "Etendue UMAP", "en": "UMAP extent"},
    "summary.files": {"fr": "**Fichiers:**", "en": "**Files:**"},
    "summary.examples": {
        "fr": "**Exemples** (5 sur {count}):",
        "en": "**Examples** (5 of {count}):",
    },
    "summary.no_noise": {"fr": "Aucun point de bruit.", "en": "No noise points."},
    "summary.empty_cluster": {"fr": "Cluster vide.", "en": "Empty cluster."},

    # ── Scatter plot ──
    "plot.main_title": {
        "fr": "Carte des codas de cachalots — Espace WhAM",
        "en": "Sperm whale coda map — WhAM space",
    },
    "plot.file_label": {"fr": "Fichier:", "en": "File:"},

    # ── Spectrogram ──
    "spec.amplitude": {"fr": "Amplitude", "en": "Amplitude"},
    "spec.frequency": {"fr": "Frequence (Hz)", "en": "Frequency (Hz)"},
    "spec.time": {"fr": "Temps (s)", "en": "Time (s)"},

    # ── Distribution chart ──
    "dist.title": {
        "fr": "Distribution des codas par cluster",
        "en": "Coda distribution by cluster",
    },
    "dist.x_label": {"fr": "Cluster", "en": "Cluster"},
    "dist.y_label": {"fr": "Nombre de codas", "en": "Number of codas"},

    # ── Identity / Gero tab ──
    "identity.title": {
        "fr": ("### Qui parle ? — Identification par les codas\n"
               "Dataset [Gero, Whitehead & Rendell (2015)]"
               "(https://doi.org/10.5061/dryad.ck4h0) : "
               "3876 codas des Caraibes orientales avec identification "
               "des individus, unites sociales et types de codas.\n"
               "Cliquez sur un point ou selectionnez un individu."),
        "en": ("### Who is speaking? — Identification by codas\n"
               "Dataset [Gero, Whitehead & Rendell (2015)]"
               "(https://doi.org/10.5061/dryad.ck4h0): "
               "3,876 codas from the Eastern Caribbean with individual, "
               "social unit and coda type identification.\n"
               "Click on a point or select an individual."),
    },
    "identity.color_by": {"fr": "Colorer par", "en": "Color by"},
    "identity.search_individual": {
        "fr": "Rechercher un individu",
        "en": "Search individual",
    },
    "identity.coda_type": {"fr": "Type de coda", "en": "Coda type"},
    "identity.social_unit": {"fr": "Unite sociale", "en": "Social unit"},
    "identity.individual": {"fr": "Individu", "en": "Individual"},
    "identity.year": {"fr": "Annee", "en": "Year"},
    "identity.all_individuals": {
        "fr": "Tous les individus",
        "en": "All individuals",
    },
    "identity.not_identified": {"fr": "Non identifie", "en": "Unidentified"},
    "identity.not_id_short": {"fr": "Non id.", "en": "Unid."},
    "identity.dataset_not_loaded": {
        "fr": "Dataset non charge.",
        "en": "Dataset not loaded.",
    },
    "identity.launch_script": {
        "fr": ("Dataset Gero non disponible. "
               "Lancez `python analyze_gero.py` pour generer les embeddings."),
        "en": ("Gero dataset not available. "
               "Run `python analyze_gero.py` to generate the embeddings."),
    },
    "identity.click_detail": {
        "fr": "Cliquez sur un point pour voir ses details.",
        "en": "Click on a point to see its details.",
    },

    # ── Gero summary ──
    "gero.dataset_title": {
        "fr": "### Dataset Gero et al. (2015)",
        "en": "### Gero et al. (2015) Dataset",
    },
    "gero.types": {"fr": "Types", "en": "Types"},
    "gero.social_units": {"fr": "Unites sociales", "en": "Social units"},
    "gero.identified": {"fr": "Individus identifies", "en": "Identified individuals"},
    "gero.period": {"fr": "Periode", "en": "Period"},
    "gero.most_frequent": {
        "fr": "**Types les plus frequents:**",
        "en": "**Most frequent types:**",
    },
    "gero.social_units_label": {
        "fr": "**Unites sociales:**",
        "en": "**Social units:**",
    },
    "gero.identified_label": {
        "fr": "**Individus identifies:**",
        "en": "**Identified individuals:**",
    },
    "gero.by_year": {"fr": "**Par annee:**", "en": "**By year:**"},
    "gero.unidentified": {"fr": "Non identifies", "en": "Unidentified"},

    # ── Gero plot ──
    "gero.title_coda_type": {"fr": "Par type de coda", "en": "By coda type"},
    "gero.title_social_unit": {"fr": "Par unite sociale", "en": "By social unit"},
    "gero.title_individual": {"fr": "Par individu", "en": "By individual"},
    "gero.title_year": {"fr": "Par annee", "en": "By year"},
    "gero.plot_title": {
        "fr": "Codas de cachalots — {subtitle} (Gero et al.)",
        "en": "Sperm whale codas — {subtitle} (Gero et al.)",
    },

    # ── Gero point detail ──
    "gero.type_label": {"fr": "Type", "en": "Type"},
    "gero.unit_label": {"fr": "Unite sociale", "en": "Social unit"},
    "gero.indiv_label": {"fr": "Individu", "en": "Individual"},
    "gero.clicks_label": {"fr": "Clics", "en": "Clicks"},
    "gero.duration_label": {"fr": "Duree", "en": "Duration"},
    "gero.icis_label": {"fr": "ICIs", "en": "ICIs"},
    "gero.date_label": {"fr": "Date", "en": "Date"},

    # ── Whale profile ──
    "profile.select": {
        "fr": "Selectionnez un individu pour voir son profil.",
        "en": "Select an individual to see their profile.",
    },
    "profile.not_found": {
        "fr": "Individu {wid} non trouve.",
        "en": "Individual {wid} not found.",
    },
    "profile.scientific_id": {
        "fr": "**Identifiant scientifique**",
        "en": "**Scientific ID**",
    },
    "profile.social_unit": {"fr": "**Unite sociale**", "en": "**Social unit**"},
    "profile.observation_period": {
        "fr": "**Periode d'observation**",
        "en": "**Observation period**",
    },
    "profile.year": {"fr": "annee", "en": "year"},
    "profile.years": {"fr": "annees", "en": "years"},
    "profile.n_codas": {
        "fr": "**Nombre de codas enregistres**",
        "en": "**Number of recorded codas**",
    },
    "profile.clicks_avg": {
        "fr": "**Clics par coda (moyenne)**",
        "en": "**Clicks per coda (average)**",
    },
    "profile.avg_duration": {
        "fr": "**Duree moyenne d'un coda**",
        "en": "**Average coda duration**",
    },
    "profile.vocal_repertoire": {
        "fr": "### Repertoire vocal",
        "en": "### Vocal repertoire",
    },
    "profile.coda_type": {"fr": "Type de coda", "en": "Coda type"},
    "profile.count": {"fr": "Nombre", "en": "Count"},
    "profile.proportion": {"fr": "Proportion", "en": "Proportion"},
    "profile.activity_by_year": {
        "fr": "### Activite par annee",
        "en": "### Activity by year",
    },
    "profile.family": {"fr": "### Famille (Unit {unit})", "en": "### Family (Unit {unit})"},
    "profile.plot_title": {
        "fr": "{name} — {n} codas dans l'espace UMAP",
        "en": "{name} — {n} codas in UMAP space",
    },

    # ── Detector tab ──
    "detector.title": {
        "fr": ("### Detecteur automatique de codas\n"
               "Portage Python du [Coda-detector]"
               "(https://github.com/Project-CETI/Coda-detector) "
               "de Project CETI. Utilise l'operateur Teager-Kaiser (TKEO) pour detecter "
               "les clics, puis un clustering par graphe pour grouper les clics en codas."),
        "en": ("### Automatic coda detector\n"
               "Python port of Project CETI's [Coda-detector]"
               "(https://github.com/Project-CETI/Coda-detector). "
               "Uses the Teager-Kaiser Energy Operator (TKEO) to detect "
               "clicks, then graph clustering to group clicks into codas."),
    },
    "detector.option1": {
        "fr": "**Option 1 : Uploader un fichier**",
        "en": "**Option 1: Upload a file**",
    },
    "detector.option2": {
        "fr": "**Option 2 : Depuis le dataset**",
        "en": "**Option 2: From dataset**",
    },
    "detector.parameters": {"fr": "**Parametres**", "en": "**Parameters**"},
    "detector.wav_label": {
        "fr": "Fichier WAV a analyser",
        "en": "WAV file to analyze",
    },
    "detector.coda_index": {"fr": "Index du coda (0-619)", "en": "Coda index (0-619)"},
    "detector.tkeo_threshold": {
        "fr": "Seuil de detection TKEO",
        "en": "TKEO detection threshold",
    },
    "detector.snr_threshold": {"fr": "Seuil SNR (dB)", "en": "SNR threshold (dB)"},
    "detector.detect_upload": {
        "fr": "Detecter (fichier uploade)",
        "en": "Detect (uploaded file)",
    },
    "detector.detect_dataset": {
        "fr": "Detecter (depuis dataset)",
        "en": "Detect (from dataset)",
    },
    "detector.plot_label": {
        "fr": "Visualisation des detections",
        "en": "Detection visualization",
    },
    "detector.placeholder": {
        "fr": "Uploadez un fichier WAV ou choisissez un index, puis cliquez sur **Detecter**.",
        "en": "Upload a WAV file or choose an index, then click **Detect**.",
    },
    "detector.upload_wav": {
        "fr": "Veuillez uploader un fichier audio WAV.",
        "en": "Please upload a WAV audio file.",
    },
    "detector.no_coda": {
        "fr": "### Aucun coda detecte\n\nEssayez de baisser les seuils de detection.",
        "en": "### No coda detected\n\nTry lowering the detection thresholds.",
    },
    "detector.n_codas": {
        "fr": "### {n} coda(s) detecte(s)",
        "en": "### {n} coda(s) detected",
    },
    "detector.table_header": {
        "fr": "| # | Clics | Debut (s) | Duree (ms) | ICIs (ms) | IPI moy. (ms) | SNR moy. |",
        "en": "| # | Clicks | Start (s) | Duration (ms) | ICIs (ms) | Avg IPI (ms) | Avg SNR |",
    },
    "detector.out_of_bounds": {
        "fr": "Index hors limites (0-{max})",
        "en": "Index out of bounds (0-{max})",
    },
    "detector.file_not_found": {
        "fr": "Fichier introuvable: {path}",
        "en": "File not found: {path}",
    },
    "detector.invalid_index": {
        "fr": "Entrez un index valide.",
        "en": "Enter a valid index.",
    },

    # ── Detection plot labels ──
    "detplot.amplitude": {"fr": "Amplitude", "en": "Amplitude"},
    "detplot.title": {
        "fr": "Signal audio + codas detectes",
        "en": "Audio signal + detected codas",
    },
    "detplot.freq": {"fr": "Freq (Hz)", "en": "Freq (Hz)"},
    "detplot.time": {"fr": "Temps (s)", "en": "Time (s)"},

    # ── Identify tab ──
    "identify.title": {
        "fr": ("### Qui parle ? — Identification acoustique\n\n"
               "Uploadez un enregistrement de cachalot (WAV, **MP3**, **MP4**, OGG, FLAC, M4A...)\n"
               "et l'application va :\n"
               "1. **Convertir** automatiquement en WAV mono 44.1kHz si necessaire\n"
               "2. **Segmenter** les fichiers longs (>30s) pour un traitement optimal\n"
               "3. **Detecter** les codas (clics groupes) via le detecteur TKEO\n"
               "4. **Identifier** chaque coda parmi les 17 individus connus (k-NN)\n"
               "5. **Combiner** les resultats si plusieurs codas (synthese multi-individus)\n\n"
               "Le classifieur est entraine sur 1602 codas etiquetes du dataset\n"
               "Gero et al. (2015). Les fichiers MP3 de YouTube fonctionnent directement !\n\n"
               "*Plus vous fournissez de codas d'une meme session, plus l'identification\n"
               "est fiable.*"),
        "en": ("### Who is speaking? — Acoustic identification\n\n"
               "Upload a sperm whale recording (WAV, **MP3**, **MP4**, OGG, FLAC, M4A...)\n"
               "and the application will:\n"
               "1. **Convert** automatically to WAV mono 44.1kHz if needed\n"
               "2. **Segment** long files (>30s) for optimal processing\n"
               "3. **Detect** codas (grouped clicks) via the TKEO detector\n"
               "4. **Identify** each coda among the 17 known individuals (k-NN)\n"
               "5. **Combine** results if multiple codas (multi-individual synthesis)\n\n"
               "The classifier is trained on 1,602 labeled codas from the\n"
               "Gero et al. (2015) dataset. YouTube MP3 files work directly!\n\n"
               "*The more codas you provide from the same session, the more reliable\n"
               "the identification.*"),
    },
    "identify.file_label": {
        "fr": "Fichier audio/video (WAV, MP3, MP4, OGG, FLAC...)",
        "en": "Audio/video file (WAV, MP3, MP4, OGG, FLAC...)",
    },
    "identify.btn": {"fr": "Identifier", "en": "Identify"},
    "identify.how_to_read": {
        "fr": ("**Comment lire les resultats ?**\n"
               "- **Confiance > 40%** : match probable\n"
               "- **20-40%** : match possible, a confirmer\n"
               "- **< 20%** : individu probablement inconnu\n"
               "- La **synthese multi-codas** combine les\n"
               "  probabilites de tous les codas detectes"),
        "en": ("**How to read the results?**\n"
               "- **Confidence > 40%**: probable match\n"
               "- **20-40%**: possible match, to confirm\n"
               "- **< 20%**: likely unknown individual\n"
               "- **Multi-coda synthesis** combines the\n"
               "  probabilities of all detected codas"),
    },
    "identify.plot_label": {
        "fr": "Codas detectes — patron de clics",
        "en": "Detected codas — click pattern",
    },
    "identify.placeholder": {
        "fr": "Uploadez un fichier audio (WAV, MP3...) puis cliquez sur **Identifier**.",
        "en": "Upload an audio file (WAV, MP3...) then click **Identify**.",
    },
    "identify.classifier_unavailable": {
        "fr": ("Classifieur non disponible. "
               "Le dataset Gero doit contenir au moins 50 codas etiquetes."),
        "en": ("Classifier not available. "
               "The Gero dataset must contain at least 50 labeled codas."),
    },

    # ── Identify results ──
    "idr.upload_file": {
        "fr": "Uploadez un fichier audio ou video (WAV, MP3, MP4, OGG, FLAC...).",
        "en": "Upload an audio or video file (WAV, MP3, MP4, OGG, FLAC...).",
    },
    "idr.conversion_error": {
        "fr": "Erreur de conversion audio : {e}",
        "en": "Audio conversion error: {e}",
    },
    "idr.no_coda": {
        "fr": "Aucun coda detecte dans cet enregistrement.",
        "en": "No coda detected in this recording.",
    },
    "idr.n_codas": {
        "fr": "## {n} coda(s) detecte(s)",
        "en": "## {n} coda(s) detected",
    },
    "idr.format": {"fr": "Format", "en": "Format"},
    "idr.duration": {"fr": "Duree", "en": "Duration"},
    "idr.segments": {
        "fr": "traite en {n} segments",
        "en": "processed in {n} segments",
    },
    "idr.clicks": {"fr": "Clics", "en": "Clicks"},
    "idr.classifier_unavailable": {
        "fr": "> Classifieur non disponible.",
        "en": "> Classifier not available.",
    },
    "idr.rank": {"fr": "Rang", "en": "Rank"},
    "idr.individual": {"fr": "Individu", "en": "Individual"},
    "idr.confidence": {"fr": "Confiance", "en": "Confidence"},
    "idr.best_match": {
        "fr": "> **Meilleur match** : {name} ({pct:.0f}%)",
        "en": "> **Best match**: {name} ({pct:.0f}%)",
    },
    "idr.possible_match": {
        "fr": "> Match possible : {name} ({pct:.0f}%) — confiance moderee",
        "en": "> Possible match: {name} ({pct:.0f}%) — moderate confidence",
    },
    "idr.low_confidence": {
        "fr": "> Confiance trop faible — possiblement un individu inconnu",
        "en": "> Confidence too low — possibly an unknown individual",
    },
    "idr.multi_synthesis": {
        "fr": "### Synthese multi-codas",
        "en": "### Multi-coda synthesis",
    },
    "idr.combining_all": {
        "fr": "En combinant tous les codas :",
        "en": "Combining all codas:",
    },
    "idr.cumulative_score": {"fr": "Score cumule", "en": "Cumulative score"},
    "idr.multi_individual": {
        "fr": "> **Multi-individus probable** — au moins 2 voix distinctes detectees",
        "en": "> **Multiple individuals likely** — at least 2 distinct voices detected",
    },
    "idr.main_identification": {
        "fr": "> **Identification principale** : **{name}**",
        "en": "> **Primary identification**: **{name}**",
    },

    # ── Identification plot ──
    "idplot.coda_title": {
        "fr": "Coda {i} — {n} clics, {ms}ms",
        "en": "Coda {i} — {n} clicks, {ms}ms",
    },
    "idplot.time": {"fr": "Temps (s)", "en": "Time (s)"},
    "idplot.click": {"fr": "Clic", "en": "Click"},

    # ── Vocal analysis tab ──
    "vocal.title": {
        "fr": ("### Que fait ce cachalot ? — Classification d'activite vocale\n\n"
               "Uploadez un enregistrement et l'application classifie\n"
               "automatiquement l'activite du cachalot :\n\n"
               "- 📡 **Echolocation** (sonar) — clics reguliers espaces (~0.5-2s),\n"
               "  le cachalot scanne son environnement ou chasse en profondeur\n"
               "- 💬 **Codas** (communication) — rafales rythmiques de clics,\n"
               "  vocalisations sociales entre individus\n"
               "- 🎯 **Creaks/Buzz** (capture) — clics ultra-rapides (>30/s),\n"
               "  tentative de capture de proie imminente\n"
               "- 🔇 **Silence** — pas d'activite acoustique detectee\n\n"
               "*Tous les formats audio et video sont supportes (WAV, MP3, MP4, OGG, FLAC...).*"),
        "en": ("### What is this whale doing? — Vocal activity classification\n\n"
               "Upload a recording and the application automatically classifies\n"
               "the sperm whale's activity:\n\n"
               "- 📡 **Echolocation** (sonar) — regular spaced clicks (~0.5-2s),\n"
               "  the whale scans its environment or hunts at depth\n"
               "- 💬 **Codas** (communication) — rhythmic click bursts,\n"
               "  social vocalizations between individuals\n"
               "- 🎯 **Creaks/Buzz** (capture) — ultra-fast clicks (>30/s),\n"
               "  imminent prey capture attempt\n"
               "- 🔇 **Silence** — no acoustic activity detected\n\n"
               "*All audio and video formats are supported (WAV, MP3, MP4, OGG, FLAC...).*"),
    },
    "vocal.file_label": {
        "fr": "Fichier audio/video (WAV, MP3, MP4...)",
        "en": "Audio/video file (WAV, MP3, MP4...)",
    },
    "vocal.btn": {"fr": "Analyser l'activite", "en": "Analyze activity"},
    "vocal.legend": {
        "fr": ("**Legendes des couleurs :**\n"
               "- 🔵 Bleu = Echolocation\n"
               "- 🟢 Vert = Codas\n"
               "- 🔴 Rouge = Creaks (chasse)\n"
               "- ⚫ Gris = Silence\n\n"
               "**Astuce** : pour l'identification\n"
               "individuelle, utilisez l'onglet\n"
               "*Identifier un coda*."),
        "en": ("**Color legend:**\n"
               "- 🔵 Blue = Echolocation\n"
               "- 🟢 Green = Codas\n"
               "- 🔴 Red = Creaks (hunting)\n"
               "- ⚫ Gray = Silence\n\n"
               "**Tip**: for individual identification,\n"
               "use the *Identify a coda* tab."),
    },
    "vocal.plot_label": {
        "fr": "Timeline d'activite vocale",
        "en": "Vocal activity timeline",
    },
    "vocal.placeholder": {
        "fr": "Uploadez un fichier audio puis cliquez sur **Analyser**.",
        "en": "Upload an audio file then click **Analyze**.",
    },

    # ── Vocal analysis results ──
    "va.upload_file": {
        "fr": "Uploadez un fichier audio ou video (WAV, MP3, MP4, OGG, FLAC...).",
        "en": "Upload an audio or video file (WAV, MP3, MP4, OGG, FLAC...).",
    },
    "va.no_activity": {
        "fr": "Aucune activite detectee.",
        "en": "No activity detected.",
    },
    "va.echo_label": {
        "fr": "Echolocation (sonar)",
        "en": "Echolocation (sonar)",
    },
    "va.coda_label": {
        "fr": "Codas (communication)",
        "en": "Codas (communication)",
    },
    "va.creak_label": {"fr": "Creaks (chasse)", "en": "Creaks (hunting)"},
    "va.silence_label": {"fr": "Silence", "en": "Silence"},
    "va.timeline_title": {
        "fr": "Timeline d'activite vocale",
        "en": "Vocal activity timeline",
    },
    "va.time": {"fr": "Temps (s)", "en": "Time (s)"},
    "va.amplitude": {"fr": "Amplitude", "en": "Amplitude"},
    "va.analysis_title": {
        "fr": "## Analyse de l'activite vocale",
        "en": "## Vocal activity analysis",
    },
    "va.summary": {"fr": "### Resume", "en": "### Summary"},
    "va.activity": {"fr": "Activite", "en": "Activity"},
    "va.duration": {"fr": "Duree", "en": "Duration"},
    "va.proportion": {"fr": "Proportion", "en": "Proportion"},
    "va.total_duration": {
        "fr": "Duree totale analysee : {s:.1f}s",
        "en": "Total analyzed duration: {s:.1f}s",
    },
    "va.detail_by_segment": {
        "fr": "### Detail par segment",
        "en": "### Segment detail",
    },
    "va.clicks_count": {"fr": "clics", "en": "clicks"},
    "va.echo_analysis_title": {
        "fr": "### Analyse de l'echolocation",
        "en": "### Echolocation analysis",
    },
    "va.echo_description": {
        "fr": ("Les clics d'echolocation sont un **sonar biologique** : "
               "le cachalot emet un clic puissant qui rebondit sur les "
               "objets environnants (proies, fond marin). L'intervalle "
               "entre les clics (ICI ~0.5-2s) correspond au temps d'aller-retour "
               "du son, et diminue quand la proie est plus proche."),
        "en": ("Echolocation clicks are a **biological sonar**: "
               "the sperm whale emits a powerful click that bounces off "
               "surrounding objects (prey, seafloor). The interval "
               "between clicks (ICI ~0.5-2s) corresponds to the round-trip "
               "time of the sound, decreasing as the prey gets closer."),
    },
    "va.avg_ici": {"fr": "ICI moyen", "en": "Average ICI"},
    "va.target_depth": {
        "fr": "Profondeur estimee de la cible",
        "en": "Estimated target depth",
    },
    "va.interpretation": {"fr": "Interpretation", "en": "Interpretation"},
    "va.deep_hunt": {
        "fr": "chasse en eau profonde",
        "en": "deep-water hunting",
    },
    "va.approaching": {
        "fr": "approche d'une cible",
        "en": "approaching a target",
    },
    "va.close_target": {
        "fr": "cible proche, pre-capture",
        "en": "close target, pre-capture",
    },
    "va.creaks_title": {
        "fr": "### Creaks detectes !",
        "en": "### Creaks detected!",
    },
    "va.creaks_description": {
        "fr": ("Les **creaks** (aussi appeles buzz) sont des rafales de clics "
               "ultra-rapides (>30 clics/seconde) emises juste avant la capture "
               "d'une proie. C'est l'equivalent du **buzz terminal** des "
               "chauves-souris. Leur presence indique une **tentative de chasse active**."),
        "en": ("**Creaks** (also called buzz) are bursts of ultra-fast clicks "
               "(>30 clicks/second) emitted just before prey capture. "
               "They are the equivalent of the bat's **terminal buzz**. "
               "Their presence indicates an **active hunting attempt**."),
    },
    "va.codas_detected": {
        "fr": "### Codas detectes — identification en cours...",
        "en": "### Codas detected — identification in progress...",
    },
    "va.codas_id_hint": {
        "fr": ("Les segments de codas peuvent etre analyses dans l'onglet "
               "**Identifier un coda** pour retrouver l'individu."),
        "en": ("Coda segments can be analyzed in the "
               "**Identify a coda** tab to find the individual."),
    },

    # ── Vocal classify segment details ──
    "seg.buzz_creak": {
        "fr": "Buzz/Creak — {n} clics, rythme={rate:.0f} clics/s, ICI={ici:.0f}ms",
        "en": "Buzz/Creak — {n} clicks, rate={rate:.0f} clicks/s, ICI={ici:.0f}ms",
    },
    "seg.codas": {
        "fr": "Codas — {n} clics, ICI moy={ici:.0f}ms",
        "en": "Codas — {n} clicks, avg ICI={ici:.0f}ms",
    },
    "seg.echolocation": {
        "fr": "Echolocation — {n} clics, ICI moy={ici:.0f}ms, regulier (CV={cv:.2f})",
        "en": "Echolocation — {n} clicks, avg ICI={ici:.0f}ms, regular (CV={cv:.2f})",
    },
    "seg.regular_clicks": {
        "fr": "Clics reguliers — {n} clics, ICI moy={ici:.0f}ms",
        "en": "Regular clicks — {n} clicks, avg ICI={ici:.0f}ms",
    },
    "seg.vocal_activity": {
        "fr": "Activite vocale — {n} clics, ICI moy={ici:.0f}ms",
        "en": "Vocal activity — {n} clicks, avg ICI={ici:.0f}ms",
    },
    "seg.no_acoustic": {
        "fr": "Aucune activite acoustique detectee",
        "en": "No acoustic activity detected",
    },

    # ── Study area tab ──
    "study.title": {
        "fr": "### Ou vivent ces cachalots ?",
        "en": "### Where do these whales live?",
    },
    "study.description": {
        "fr": ("Tous les enregistrements de cette application proviennent de la **Dominique**, "
               "une petite ile volcanique des **Caraibes orientales** (15.4\u00b0N, 61.4\u00b0W). "
               "Cette region abrite l'une des populations de cachalots les mieux etudiees "
               "au monde, suivie depuis plus de 20 ans par le **Dominica Sperm Whale Project** "
               "dirige par Shane Gero et Hal Whitehead.\n\n"
               "Les cachalots de la Dominique vivent en **unites sociales matrilineaires** "
               "(des familles de femelles et de jeunes) qui partagent un repertoire "
               "de codas commun — un peu comme un dialecte regional."),
        "en": ("All recordings in this application come from **Dominica**, "
               "a small volcanic island in the **Eastern Caribbean** (15.4\u00b0N, 61.4\u00b0W). "
               "This region is home to one of the best-studied sperm whale populations "
               "in the world, monitored for over 20 years by the **Dominica Sperm Whale Project** "
               "led by Shane Gero and Hal Whitehead.\n\n"
               "Dominica's sperm whales live in **matrilineal social units** "
               "(families of females and young) that share a common coda repertoire "
               "— much like a regional dialect."),
    },
    "study.dominica": {"fr": "Dominique", "en": "Dominica"},
    "study.study_zone": {"fr": "Zone d'etude", "en": "Study area"},
    "study.observation_zones": {
        "fr": "Zones d'observation",
        "en": "Observation areas",
    },
    "study.map_title": {
        "fr": "Zone d'etude — Dominique, Caraibes orientales",
        "en": "Study area — Dominica, Eastern Caribbean",
    },
    "study.why_dominica": {
        "fr": ("**Pourquoi la Dominique ?**\n\n"
               "Les eaux profondes au large de la cote ouest de la Dominique plongent "
               "rapidement a plus de 1000 metres — l'habitat ideal des cachalots qui "
               "chassent les calamars geants en profondeur. Cette proximite avec la cote "
               "permet aux chercheurs d'observer et d'enregistrer les cachalots "
               "presque quotidiennement.\n\n"
               "**Donnees GPS** : les datasets utilises ici ne contiennent pas de "
               "coordonnees GPS par enregistrement. Si des donnees georeferencees "
               "deviennent disponibles, la carte s'enrichira automatiquement avec "
               "les positions individuelles de chaque coda."),
        "en": ("**Why Dominica?**\n\n"
               "The deep waters off Dominica's west coast plunge "
               "quickly to over 1,000 meters — the ideal habitat for sperm whales that "
               "hunt giant squid at depth. This proximity to shore "
               "allows researchers to observe and record the whales "
               "almost daily.\n\n"
               "**GPS data**: the datasets used here do not contain GPS "
               "coordinates per recording. If georeferenced data "
               "become available, the map will automatically be enriched with "
               "the individual positions of each coda."),
    },

    # ── Guide & Glossary ──
    "guide.full": {
        "fr": """\
## Comprendre l'application

Cette page explique les concepts techniques utilises dans Whale Coda Explorer.
Pas besoin d'etre scientifique pour comprendre !

---

### Qu'est-ce qu'un coda ?

Un **coda** est une vocalisation sociale des cachalots. C'est une serie
de **clics** (comme des claquements) emis en rafale rapide.
Imaginez quelqu'un qui frappe sur une table avec un rythme precis :
*toc-toc-toc...toc-toc*. Chaque patron rythmique est un "type" de coda.

Les cachalots utilisent les codas pour **communiquer entre eux**,
un peu comme nous utilisons des mots ou des expressions.
Differentes familles de cachalots utilisent des repertoires
differents — comme des dialectes regionaux.

**Exemple** : un coda de type "1+1+3" ressemble a :
*clic — (pause) — clic — (pause) — clic-clic-clic* (rapide)

---

### Qu'est-ce que l'ICI ?

L'**ICI** (Inter-Click Interval) est le **temps entre deux clics consecutifs**
dans un coda, mesure en secondes. C'est le "rythme" du coda.

Un coda avec 5 clics a 4 ICI (les intervalles entre chaque paire de clics).
Ces intervalles forment un **patron rythmique** qui definit le type du coda.

> Pensez a la musique : deux morceaux peuvent avoir les memes notes
> mais un rythme different. C'est pareil pour les codas !

---

### Qu'est-ce que l'UMAP ?

**UMAP** (Uniform Manifold Approximation and Projection) est un algorithme
qui prend des donnees complexes et les **projette sur une carte 2D**
pour qu'on puisse les visualiser.

**Analogie** : imaginez que vous avez 620 recettes de cuisine, chacune
decrite par 10 ingredients et leurs proportions. C'est difficile
a visualiser en 10 dimensions ! L'UMAP prend ces 620 recettes et les
place sur une carte plate, de sorte que :
- Les recettes **similaires** sont **proches** sur la carte
- Les recettes **differentes** sont **eloignees**

Quand vous voyez le scatter plot (nuage de points), chaque point est un
coda. Les codas proches sur la carte ont des rythmes similaires.
Les groupes de points (clusters) representent des "types" de codas.

**Ce qui compte** : la **distance relative** entre les points,
pas leur position absolue. Les axes (UMAP 1, UMAP 2) n'ont pas
d'unite physique — ce sont des coordonnees abstraites.

---

### Qu'est-ce qu'un cluster ?

Un **cluster** est un **groupe de codas similaires** identifies
automatiquement par l'algorithme HDBSCAN. L'algorithme detecte les zones
denses du nuage de points (la ou beaucoup de codas se ressemblent)
et les regroupe.

- Chaque cluster a une **couleur** sur la carte
- Les points gris etiquetes "Bruit" sont des codas que l'algorithme
  n'a pas reussi a classer dans un groupe — ils sont trop atypiques
- Le nombre de clusters n'est **pas choisi a l'avance** :
  l'algorithme le determine tout seul

---

### Qu'est-ce que WhAM ?

**WhAM** (Whale Acoustics Model) est un modele d'intelligence artificielle
developpe par **Project CETI** (Cetacean Translation Initiative).
C'est un reseau de neurones de type *transformer* (la meme technologie
derriere ChatGPT et Claude) mais specialise dans l'audio des cachalots.

WhAM sait **encoder** un coda en un vecteur mathematique (une liste
de nombres) qui capture son "essence acoustique". C'est a partir
de ces vecteurs que l'UMAP construit la carte.

---

### Qu'est-ce que le TKEO ?

Le **TKEO** (Teager-Kaiser Energy Operator) est un outil mathematique
qui mesure l'**energie instantanee** d'un signal sonore.
Il permet de detecter les clics dans un enregistrement audio,
meme quand il y a du bruit de fond (vagues, moteurs, etc.).

Le detecteur de codas de l'application utilise le TKEO pour
reperer chaque clic, puis regroupe les clics proches en codas.

---

### Qu'est-ce qu'une unite sociale ?

Chez les cachalots, une **unite sociale** est une **famille elargie**
composee principalement de femelles adultes et de leurs petits.
Ces unites sont **matrilineaires** : les membres sont lies
par la lignee maternelle (meres, filles, soeurs, tantes).

Les unites sociales voyagent ensemble, chassent ensemble,
et partagent un **repertoire de codas** commun.
C'est un peu comme une famille qui partagerait un accent
ou des expressions propres.

Dans le dataset Gero, les unites sont identifiees par des lettres :
A, B, F, J, N, R, S, T, U.

---

### Le spectrogramme

Un **spectrogramme** est une image qui represente un son dans le temps.
- L'axe horizontal = le **temps** (en secondes)
- L'axe vertical = la **frequence** (grave en bas, aigu en haut)
- La couleur = l'**intensite** (plus c'est clair, plus c'est fort)

Sur un spectrogramme de coda, chaque clic apparait comme une
**ligne verticale** fine (car un clic contient toutes les frequences
d'un coup, comme un claquement de doigts).

---

### Glossaire rapide

| Terme | Definition simple |
|-------|------------------|
| **Coda** | Vocalisation sociale du cachalot : serie de clics rythmiques |
| **Clic** | Son bref et sec emis par le cachalot (comme un claquement) |
| **ICI** | Intervalle de temps entre deux clics consecutifs |
| **UMAP** | Algorithme qui cree une "carte" 2D a partir de donnees complexes |
| **Cluster** | Groupe de codas similaires detecte automatiquement |
| **HDBSCAN** | Algorithme de clustering base sur la densite |
| **WhAM** | Modele IA pour analyser l'audio des cachalots (Project CETI) |
| **TKEO** | Operateur mathematique pour detecter les clics dans le bruit |
| **Unite sociale** | Famille matrilineaire de cachalots |
| **Spectrogramme** | Image temps-frequence d'un son |
| **Embedding** | Representation numerique (vecteur) d'un coda |
| **Scatter plot** | Nuage de points ou chaque point = un coda |
| **SNR** | Signal-to-Noise Ratio — rapport signal/bruit en decibels |
| **Rubato** | Variation subtile de rythme dans un coda (comme en musique) |
| **DSWP** | Dominica Sperm Whale Project — projet d'etude de 20+ ans |
| **Project CETI** | Cetacean Translation Initiative — initiative de "traduction" |

---

*Ce projet est developpe par Claude & Kevin dans le cadre de
[CivicDash](https://github.com/CivicDash) — outils open-source
pour rapprocher humains et nature.*""",
        "en": """\
## Understanding the application

This page explains the technical concepts used in Whale Coda Explorer.
No need to be a scientist to understand!

---

### What is a coda?

A **coda** is a social vocalization of sperm whales. It is a series
of **clicks** (like snapping sounds) emitted in a quick burst.
Imagine someone tapping on a table with a precise rhythm:
*tap-tap-tap...tap-tap*. Each rhythmic pattern is a "type" of coda.

Sperm whales use codas to **communicate with each other**,
much like we use words or expressions.
Different sperm whale families use different repertoires
— like regional dialects.

**Example**: a "1+1+3" coda type sounds like:
*click — (pause) — click — (pause) — click-click-click* (fast)

---

### What is ICI?

**ICI** (Inter-Click Interval) is the **time between two consecutive clicks**
in a coda, measured in seconds. It is the "rhythm" of the coda.

A coda with 5 clicks has 4 ICIs (the intervals between each pair of clicks).
These intervals form a **rhythmic pattern** that defines the coda type.

> Think of music: two pieces can have the same notes
> but a different rhythm. It's the same for codas!

---

### What is UMAP?

**UMAP** (Uniform Manifold Approximation and Projection) is an algorithm
that takes complex data and **projects it onto a 2D map**
so we can visualize it.

**Analogy**: imagine you have 620 recipes, each described
by 10 ingredients and their proportions. It's hard
to visualize in 10 dimensions! UMAP takes these 620 recipes and
places them on a flat map, so that:
- **Similar** recipes are **close** on the map
- **Different** recipes are **far apart**

When you see the scatter plot (point cloud), each point is a
coda. Codas close on the map have similar rhythms.
Point groups (clusters) represent "types" of codas.

**What matters**: the **relative distance** between points,
not their absolute position. The axes (UMAP 1, UMAP 2) have no
physical unit — they are abstract coordinates.

---

### What is a cluster?

A **cluster** is a **group of similar codas** identified
automatically by the HDBSCAN algorithm. The algorithm detects dense
areas of the point cloud (where many codas are similar)
and groups them together.

- Each cluster has a **color** on the map
- Gray points labeled "Noise" are codas that the algorithm
  could not classify into a group — they are too atypical
- The number of clusters is **not chosen in advance**:
  the algorithm determines it on its own

---

### What is WhAM?

**WhAM** (Whale Acoustics Model) is an artificial intelligence model
developed by **Project CETI** (Cetacean Translation Initiative).
It is a *transformer* neural network (the same technology
behind ChatGPT and Claude) but specialized in sperm whale audio.

WhAM can **encode** a coda into a mathematical vector (a list
of numbers) that captures its "acoustic essence". It is from
these vectors that UMAP builds the map.

---

### What is TKEO?

**TKEO** (Teager-Kaiser Energy Operator) is a mathematical tool
that measures the **instantaneous energy** of a sound signal.
It allows detecting clicks in an audio recording,
even when there is background noise (waves, engines, etc.).

The application's coda detector uses TKEO to
spot each click, then groups nearby clicks into codas.

---

### What is a social unit?

In sperm whales, a **social unit** is an **extended family**
composed mainly of adult females and their calves.
These units are **matrilineal**: members are linked
through the maternal line (mothers, daughters, sisters, aunts).

Social units travel together, hunt together,
and share a **common coda repertoire**.
It's a bit like a family that would share an accent
or specific expressions.

In the Gero dataset, units are identified by letters:
A, B, F, J, N, R, S, T, U.

---

### The spectrogram

A **spectrogram** is an image that represents a sound over time.
- The horizontal axis = **time** (in seconds)
- The vertical axis = **frequency** (low at bottom, high at top)
- The color = **intensity** (brighter means louder)

On a coda spectrogram, each click appears as a
thin **vertical line** (because a click contains all frequencies
at once, like a finger snap).

---

### Quick glossary

| Term | Simple definition |
|------|-------------------|
| **Coda** | Social vocalization of sperm whales: a series of rhythmic clicks |
| **Click** | A brief, sharp sound emitted by the whale (like a snap) |
| **ICI** | Time interval between two consecutive clicks |
| **UMAP** | Algorithm that creates a 2D "map" from complex data |
| **Cluster** | Group of similar codas automatically detected |
| **HDBSCAN** | Density-based clustering algorithm |
| **WhAM** | AI model for analyzing sperm whale audio (Project CETI) |
| **TKEO** | Mathematical operator for detecting clicks in noise |
| **Social unit** | Matrilineal family of sperm whales |
| **Spectrogram** | Time-frequency image of a sound |
| **Embedding** | Numerical representation (vector) of a coda |
| **Scatter plot** | Point cloud where each point = one coda |
| **SNR** | Signal-to-Noise Ratio — in decibels |
| **Rubato** | Subtle rhythm variation in a coda (like in music) |
| **DSWP** | Dominica Sperm Whale Project — 20+ year study |
| **Project CETI** | Cetacean Translation Initiative — "translation" initiative |

---

*This project is developed by Claude & Kevin as part of
[CivicDash](https://github.com/CivicDash) — open-source tools
to bring humans and nature closer.*""",
    },

    # ── Footer ──
    "footer": {
        "fr": (
            '<p style="text-align: center; color: #666; font-size: 0.85em;">'
            "Whale Coda Explorer — Donnees: DSWP (Project CETI) · Modele: WhAM · "
            "Clustering: UMAP + HDBSCAN<br>"
            "Detecteur: portage Python du Coda-detector CETI (TKEO + graph clustering)<br>"
            'Projet de Claude & Kevin · <a href="https://github.com/CivicDash/whale-coda-explorer">GitHub</a>'
            "</p>"
        ),
        "en": (
            '<p style="text-align: center; color: #666; font-size: 0.85em;">'
            "Whale Coda Explorer — Data: DSWP (Project CETI) · Model: WhAM · "
            "Clustering: UMAP + HDBSCAN<br>"
            "Detector: Python port of CETI Coda-detector (TKEO + graph clustering)<br>"
            'Project by Claude & Kevin · <a href="https://github.com/CivicDash/whale-coda-explorer">GitHub</a>'
            "</p>"
        ),
    },
}


# ── Whale names & descriptions ──
WHALE_NAMES_I18N = {
    "5130": {
        "name": "Calypso",
        "fr": "La discrete de la famille F — seulement 28 codas enregistres, prefere le rythme 1+1+3",
        "en": "The quiet one of family F — only 28 recorded codas, prefers the 1+1+3 rhythm",
    },
    "5151": {
        "name": "Triton",
        "fr": "Bavard de l'unite T — adore les longues sequences 5R2, jusqu'a 6 clics en moyenne",
        "en": "Talkative one of unit T — loves long 5R2 sequences, up to 6 clicks on average",
    },
    "5560": {
        "name": "Ondine",
        "fr": "Polyglotte de la famille F — maitrise 5R1, 1+1+3, et au moins 6 types de codas",
        "en": "Polyglot of family F — masters 5R1, 1+1+3, and at least 6 coda types",
    },
    "5561": {
        "name": "Echo",
        "fr": "La fidele — 77% de ses codas sont du type 1+1+3, le dialecte de la famille F",
        "en": "The faithful one — 77% of her codas are 1+1+3 type, the F family dialect",
    },
    "5562": {
        "name": "Nereid",
        "fr": "Specialiste du 5R1 — 87% de ses vocalisations suivent ce rythme precis",
        "en": "5R1 specialist — 87% of her vocalizations follow this precise rhythm",
    },
    "5563": {
        "name": "Corail",
        "fr": "La jeune exploratrice — peu de codas mais une grande diversite de types",
        "en": "The young explorer — few codas but a great diversity of types",
    },
    "5703": {
        "name": "Poseidon",
        "fr": "Le chanteur aux longues phrases — 7 clics en moyenne, maitre des codas irreguliers",
        "en": "The singer of long phrases — 7 clicks on average, master of irregular codas",
    },
    "5722": {
        "name": "Aurora",
        "fr": "La plus prolifique — 281 codas ! Signature unique avec les rythmes 4D et 7D",
        "en": "The most prolific — 281 codas! Unique signature with 4D and 7D rhythms",
    },
    "5727": {
        "name": "Vega",
        "fr": "Heritiere du dialecte familial — 64% de 1+1+3, le coeur de l'identite F",
        "en": "Heir to the family dialect — 64% 1+1+3, the heart of the F identity",
    },
    "5978": {
        "name": "Fidele",
        "fr": "La constante de l'unite J — 98% de 1+1+3, une voix qui ne varie presque jamais",
        "en": "The constant of unit J — 98% 1+1+3, a voice that almost never varies",
    },
    "5979": {
        "name": "Melody",
        "fr": "La versatile — melange 1+1+3 et 5R1, un pont entre deux dialectes",
        "en": "The versatile one — blends 1+1+3 and 5R1, a bridge between two dialects",
    },
    "5981": {
        "name": "Harmonie",
        "fr": "Pure unite J — 99% de 1+1+3, l'essence meme du dialecte de son clan",
        "en": "Pure unit J — 99% 1+1+3, the very essence of her clan's dialect",
    },
    "5987": {
        "name": "Rythme",
        "fr": "L'equilibriste — partage ses codas entre 5R1 et 1+1+3, deux voix en une",
        "en": "The balancer — splits codas between 5R1 and 1+1+3, two voices in one",
    },
    "59871": {
        "name": "Petit Flot",
        "fr": "Le bebe — seulement 3 codas enregistres, balbutie encore en 4R1",
        "en": "The baby — only 3 recorded codas, still babbling in 4R1",
    },
    "6035": {
        "name": "Saphir",
        "fr": "La diplomate de l'unite T — parle 1+1+3 comme les J et 5R1 comme les siens",
        "en": "The diplomat of unit T — speaks 1+1+3 like the J's and 5R1 like her own",
    },
    "6058": {
        "name": "Sirius",
        "fr": "Voix claire de l'unite T — 64% de 5R1, signature nette et reconnaissable",
        "en": "Clear voice of unit T — 64% 5R1, a clear and recognizable signature",
    },
    "6070/6068": {
        "name": "Les Jumelles",
        "fr": "Duo inseparable — identite parfois ambigue, coeur de la famille F",
        "en": "Inseparable duo — sometimes ambiguous identity, heart of the F family",
    },
}


def t(key, lang="fr", **kwargs):
    """Translate a key to the given language, with optional format parameters."""
    entry = STRINGS.get(key, {})
    text = entry.get(lang, entry.get("fr", f"[{key}]"))
    if kwargs:
        try:
            text = text.format(**kwargs)
        except (KeyError, IndexError):
            pass
    return text


def whale_name(wid, lang="fr"):
    """Return (name, description) for a whale ID in the given language."""
    wid = str(wid)
    entry = WHALE_NAMES_I18N.get(wid)
    if entry is None:
        return (f"Whale #{wid}", "")
    return (entry["name"], entry.get(lang, entry.get("fr", "")))


def whale_display(wid, lang="fr"):
    """Return display string 'Name (#ID)' or localized 'Unidentified'."""
    wid = str(wid)
    if wid == "0":
        return t("identity.not_identified", lang)
    entry = WHALE_NAMES_I18N.get(wid)
    if entry:
        return f"{entry['name']} (#{wid})"
    return f"Whale #{wid}"
