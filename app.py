"""
Whale Coda Explorer — Application web interactive
pour explorer les vocalisations de cachalots.

Visualisation interactive des clusters de codas, ecoute audio,
statistiques par cluster, et spectrogrammes.

Projet personnel de Claude, avec la benediction de Kevin.
3 mars 2026.
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import gradio as gr
import scipy.io.wavfile as wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from coda_detector import detect_codas, codas_to_dict, DetectorParams, teager_kaiser

DATA_DIR = Path(__file__).parent / "exploration_output"
GERO_PATH = Path(__file__).parent / "data" / "gero2015_codas.xlsx"
EMBEDDINGS_2D = np.load(DATA_DIR / "embedding_2d.npy")
CLUSTER_LABELS = np.load(DATA_DIR / "cluster_labels.npy")
FULL_EMBEDDINGS = np.load(DATA_DIR / "embeddings.npy")

with open(DATA_DIR / "filenames.txt") as f:
    FILENAMES = [line.strip() for line in f if line.strip()]

N_CLUSTERS = len(set(CLUSTER_LABELS)) - (1 if -1 in CLUSTER_LABELS else 0)

CLUSTER_COLORS = {
    -1: "#CCCCCC",
}
SPECTRAL_PALETTE = [
    "#9E0142", "#D53E4F", "#F46D43", "#FDAE61", "#FEE08B",
    "#E6F598", "#ABDDA4", "#66C2A5", "#3288BD", "#5E4FA2",
    "#8073AC", "#B2ABD2", "#D8DAEB", "#FDB863", "#E08214",
    "#B35806", "#542788", "#998EC3", "#F1A340", "#01665E",
]
for i in range(N_CLUSTERS):
    CLUSTER_COLORS[i] = SPECTRAL_PALETTE[i % len(SPECTRAL_PALETTE)]



def _parse_click_value(val):
    """Extract index from 'idx_timestamp' or plain 'idx' string."""
    val = val.strip()
    if '_' in val:
        val = val.split('_')[0]
    return int(val)


def on_plotly_click(clicked_idx_str):
    """Handle click on a Plotly scatter point via JS bridge."""
    try:
        if not clicked_idx_str or clicked_idx_str.strip() == "":
            return None, None, "Cliquez sur un point de la carte."

        idx = _parse_click_value(clicked_idx_str)
        if idx < 0 or idx >= len(FILENAMES):
            return None, None, f"Index {idx} hors limites."

        filepath = FILENAMES[idx]
        if not os.path.exists(filepath):
            return None, None, f"Fichier introuvable: {filepath}"

        cid = CLUSTER_LABELS[idx]
        cluster_label = "Bruit" if cid == -1 else f"Cluster {cid}"
        specfig = make_spectrogram(filepath)

        info = f"### Coda {Path(filepath).stem}\n\n"
        info += f"- **Cluster**: {cluster_label}\n"
        info += f"- **Index**: {idx}\n"
        info += f"- **Fichier**: `{Path(filepath).name}`\n"
        info += f"- **Position UMAP**: ({EMBEDDINGS_2D[idx, 0]:.3f}, {EMBEDDINGS_2D[idx, 1]:.3f})\n"

        neighbors = find_nearest_neighbors(idx, k=5)
        if neighbors:
            info += "\n**5 codas les plus proches** (espace WhAM):\n"
            for ni, dist in neighbors:
                ncid = CLUSTER_LABELS[ni]
                nlabel = "Bruit" if ncid == -1 else f"C{ncid}"
                info += f"- {Path(FILENAMES[ni]).stem} [{nlabel}] (dist: {dist:.4f})\n"

        return filepath, specfig, info

    except Exception as e:
        return None, None, f"Erreur: {e}"


def get_cluster_stats():
    """Genere un tableau de statistiques par cluster."""
    rows = []
    for cid in sorted(set(CLUSTER_LABELS)):
        mask = CLUSTER_LABELS == cid
        count = mask.sum()
        pct = 100.0 * count / len(CLUSTER_LABELS)
        label = "Bruit (non classe)" if cid == -1 else f"Cluster {cid}"
        rows.append({
            "Cluster": label,
            "Codas": int(count),
            "Pourcentage": f"{pct:.1f}%",
        })
    rows.append({
        "Cluster": "TOTAL",
        "Codas": int(len(CLUSTER_LABELS)),
        "Pourcentage": "100%",
    })
    return rows


def build_scatter_plot(selected_cluster="Tous"):
    """Construit le scatter plot interactif Plotly."""
    fig = go.Figure()

    if selected_cluster == "Tous":
        show_clusters = sorted(set(CLUSTER_LABELS))
    elif selected_cluster == "Bruit":
        show_clusters = [-1]
    else:
        cid = int(selected_cluster.replace("Cluster ", ""))
        show_clusters = [cid]

    for cid in sorted(set(CLUSTER_LABELS)):
        mask = CLUSTER_LABELS == cid
        label = "Bruit" if cid == -1 else f"Cluster {cid}"
        visible = cid in show_clusters

        indices = np.where(mask)[0]
        hover_texts = [
            f"<b>{label}</b><br>"
            f"Fichier: {Path(FILENAMES[i]).stem}<br>"
            f"Index: {i}"
            for i in indices
        ]

        fig.add_trace(go.Scatter(
            x=EMBEDDINGS_2D[mask, 0],
            y=EMBEDDINGS_2D[mask, 1],
            mode='markers',
            marker=dict(
                size=7 if cid != -1 else 4,
                color=CLUSTER_COLORS.get(cid, "#999"),
                opacity=0.8 if cid != -1 else 0.3,
                line=dict(width=0.5, color='white') if cid != -1 else dict(width=0),
            ),
            name=label,
            text=hover_texts,
            hoverinfo='text',
            customdata=indices,
            visible=True if visible else 'legendonly',
        ))

    fig.update_layout(
        title=dict(
            text="Carte des codas de cachalots — Espace WhAM",
            font=dict(size=18),
        ),
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#444",
            borderwidth=1,
            font=dict(size=11),
        ),
        height=650,
        margin=dict(l=60, r=30, t=60, b=60),
        hovermode='closest',
    )

    return fig






def make_spectrogram(filepath):
    """Genere un spectrogramme pour un fichier WAV."""
    try:
        sr, data = wavfile.read(filepath)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float64)

        fig, axes = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={'height_ratios': [1, 2]})
        fig.patch.set_facecolor('#1a1a2e')

        t = np.arange(len(data)) / sr

        axes[0].plot(t, data, color='#66C2A5', linewidth=0.5, alpha=0.8)
        axes[0].set_xlim(0, t[-1])
        axes[0].set_ylabel("Amplitude", color='#e0e0e0', fontsize=9)
        axes[0].set_facecolor('#16213e')
        axes[0].tick_params(colors='#999')
        axes[0].spines['bottom'].set_color('#444')
        axes[0].spines['left'].set_color('#444')
        axes[0].spines['top'].set_visible(False)
        axes[0].spines['right'].set_visible(False)

        axes[1].specgram(data, Fs=sr, cmap='inferno', NFFT=512, noverlap=384)
        axes[1].set_ylabel("Frequence (Hz)", color='#e0e0e0', fontsize=9)
        axes[1].set_xlabel("Temps (s)", color='#e0e0e0', fontsize=9)
        axes[1].set_facecolor('#16213e')
        axes[1].tick_params(colors='#999')
        axes[1].spines['bottom'].set_color('#444')
        axes[1].spines['left'].set_color('#444')
        axes[1].spines['top'].set_visible(False)
        axes[1].spines['right'].set_visible(False)
        axes[1].set_ylim(0, min(sr // 2, 10000))

        filename = Path(filepath).stem
        fig.suptitle(f"Coda {filename}", color='#e0e0e0', fontsize=12, fontweight='bold')
        fig.tight_layout()

        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.text(0.5, 0.5, f"Erreur: {e}", transform=ax.transAxes,
                ha='center', va='center', color='#ff6b6b', fontsize=12)
        return fig


def get_cluster_summary(cluster_id):
    """Retourne un resume textuel pour un cluster."""
    if cluster_id == -1:
        mask = CLUSTER_LABELS == -1
        label = "Points non classes (bruit)"
    else:
        mask = CLUSTER_LABELS == cluster_id
        label = f"Cluster {cluster_id}"

    count = mask.sum()
    spread = FULL_EMBEDDINGS[mask].std()
    x_range = EMBEDDINGS_2D[mask, 0].max() - EMBEDDINGS_2D[mask, 0].min()
    y_range = EMBEDDINGS_2D[mask, 1].max() - EMBEDDINGS_2D[mask, 1].min()

    summary = f"### {label}\n\n"
    summary += f"- **Nombre de codas**: {count}\n"
    summary += f"- **Proportion**: {100*count/len(CLUSTER_LABELS):.1f}%\n"
    summary += f"- **Dispersion** (std embeddings): {spread:.4f}\n"
    summary += f"- **Etendue UMAP**: X={x_range:.2f}, Y={y_range:.2f}\n\n"

    if count <= 20:
        summary += "**Fichiers:**\n"
        indices = np.where(mask)[0]
        for idx in indices:
            summary += f"- {Path(FILENAMES[idx]).stem}\n"
    else:
        indices = np.where(mask)[0]
        summary += f"**Exemples** (5 sur {count}):\n"
        for idx in indices[:5]:
            summary += f"- {Path(FILENAMES[idx]).stem}\n"

    return summary



def on_coda_select(cluster_choice, coda_index):
    """Charge un coda specifique par son index."""
    try:
        idx = _parse_click_value(coda_index)
        if idx < 0 or idx >= len(FILENAMES):
            return None, None, f"Index {idx} hors limites (0-{len(FILENAMES)-1})"

        filepath = FILENAMES[idx]
        if not os.path.exists(filepath):
            return None, None, f"Fichier introuvable: {filepath}"

        cid = CLUSTER_LABELS[idx]
        cluster_label = "Bruit" if cid == -1 else f"Cluster {cid}"

        specfig = make_spectrogram(filepath)

        info = f"### Coda {Path(filepath).stem}\n\n"
        info += f"- **Cluster**: {cluster_label}\n"
        info += f"- **Index**: {idx}\n"
        info += f"- **Fichier**: `{Path(filepath).name}`\n"
        info += f"- **Position UMAP**: ({EMBEDDINGS_2D[idx, 0]:.3f}, {EMBEDDINGS_2D[idx, 1]:.3f})\n"

        neighbors = find_nearest_neighbors(idx, k=5)
        if neighbors:
            info += "\n**5 codas les plus proches** (espace WhAM):\n"
            for ni, dist in neighbors:
                ncid = CLUSTER_LABELS[ni]
                nlabel = "Bruit" if ncid == -1 else f"C{ncid}"
                info += f"- {Path(FILENAMES[ni]).stem} [{nlabel}] (dist: {dist:.4f})\n"

        return filepath, specfig, info

    except (ValueError, IndexError) as e:
        return None, None, f"Erreur: {e}"


def find_nearest_neighbors(idx, k=5):
    """Trouve les k voisins les plus proches dans l'espace d'embeddings."""
    target = FULL_EMBEDDINGS[idx]
    dists = np.linalg.norm(FULL_EMBEDDINGS - target, axis=1)
    dists[idx] = np.inf
    nearest = np.argsort(dists)[:k]
    return [(int(n), float(dists[n])) for n in nearest]


def get_random_coda(cluster_choice):
    """Selectionne un coda aleatoire dans le cluster choisi."""
    if cluster_choice == "Tous":
        idx = np.random.randint(len(FILENAMES))
    elif cluster_choice == "Bruit":
        candidates = np.where(CLUSTER_LABELS == -1)[0]
        if len(candidates) == 0:
            return gr.update(), None, None, "Aucun point de bruit."
        idx = np.random.choice(candidates)
    else:
        cid = int(cluster_choice.replace("Cluster ", ""))
        candidates = np.where(CLUSTER_LABELS == cid)[0]
        if len(candidates) == 0:
            return gr.update(), None, None, "Cluster vide."
        idx = np.random.choice(candidates)

    audio, specfig, info = on_coda_select(cluster_choice, str(idx))
    return gr.update(value=str(idx)), audio, specfig, info


def build_distribution_chart():
    """Graphique de distribution des clusters."""
    labels_sorted = sorted(set(CLUSTER_LABELS))
    names = ["Bruit" if l == -1 else f"C{l}" for l in labels_sorted]
    counts = [int((CLUSTER_LABELS == l).sum()) for l in labels_sorted]
    colors = [CLUSTER_COLORS.get(l, "#999") for l in labels_sorted]

    fig = go.Figure(go.Bar(
        x=names,
        y=counts,
        marker_color=colors,
        text=counts,
        textposition='outside',
        textfont=dict(color='#e0e0e0', size=10),
    ))
    fig.update_layout(
        title="Distribution des codas par cluster",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        height=350,
        margin=dict(l=50, r=30, t=50, b=50),
        xaxis_title="Cluster",
        yaxis_title="Nombre de codas",
    )
    return fig


WHALE_NAMES = {
    "5130": ("Calypso", "La discrete de la famille F — seulement 28 codas enregistres, prefere le rythme 1+1+3"),
    "5151": ("Triton", "Bavard de l'unite T — adore les longues sequences 5R2, jusqu'a 6 clics en moyenne"),
    "5560": ("Ondine", "Polyglotte de la famille F — maitrise 5R1, 1+1+3, et au moins 6 types de codas"),
    "5561": ("Echo", "La fidele — 77% de ses codas sont du type 1+1+3, le dialecte de la famille F"),
    "5562": ("Nereid", "Specialiste du 5R1 — 87% de ses vocalisations suivent ce rythme precis"),
    "5563": ("Corail", "La jeune exploratrice — peu de codas mais une grande diversite de types"),
    "5703": ("Poseidon", "Le chanteur aux longues phrases — 7 clics en moyenne, maitre des codas irreguliers"),
    "5722": ("Aurora", "La plus prolifique — 281 codas ! Signature unique avec les rythmes 4D et 7D"),
    "5727": ("Vega", "Heritiere du dialecte familial — 64% de 1+1+3, le coeur de l'identite F"),
    "5978": ("Fidele", "La constante de l'unite J — 98% de 1+1+3, une voix qui ne varie presque jamais"),
    "5979": ("Melody", "La versatile — melange 1+1+3 et 5R1, un pont entre deux dialectes"),
    "5981": ("Harmonie", "Pure unite J — 99% de 1+1+3, l'essence meme du dialecte de son clan"),
    "5987": ("Rythme", "L'equilibriste — partage ses codas entre 5R1 et 1+1+3, deux voix en une"),
    "59871": ("Petit Flot", "Le bebe — seulement 3 codas enregistres, balbutie encore en 4R1"),
    "6035": ("Saphir", "La diplomate de l'unite T — parle 1+1+3 comme les J et 5R1 comme les siens"),
    "6058": ("Sirius", "Voix claire de l'unite T — 64% de 5R1, signature nette et reconnaissable"),
    "6070/6068": ("Les Jumelles", "Duo inseparable — identite parfois ambigue, coeur de la famille F"),
}

GERO_DF = None
GERO_EMBEDDING = None
if GERO_PATH.exists():
    _gero_raw = pd.read_excel(GERO_PATH)
    _gero_raw['Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(_gero_raw['Date'], unit='D')
    _gero_raw['Year'] = _gero_raw['Date'].dt.year
    _unit_map = {1: "A", 2: "B", 3: "F", 4: "J", 5: "N", 6: "R", 7: "S", 8: "T", 9: "U"}
    _gero_raw['UnitName'] = _gero_raw['Unit'].map(_unit_map).fillna("?")
    GERO_DF = _gero_raw[_gero_raw['CodaName'] != 'NOISE'].reset_index(drop=True)
    GERO_DF['WhaleID'] = GERO_DF['WhaleID'].astype(str)

    _gero_emb_path = DATA_DIR / 'gero_embedding_2d.npy'
    if _gero_emb_path.exists():
        GERO_EMBEDDING = np.load(_gero_emb_path)

# --- Classifieur k-NN pour identification individuelle ---
WHALE_CLASSIFIER = None
WHALE_SCALER = None
ICI_COLS = ['ICI1', 'ICI2', 'ICI3', 'ICI4', 'ICI5', 'ICI6', 'ICI7', 'ICI8', 'ICI9']

if GERO_DF is not None:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    _labeled = GERO_DF[GERO_DF['WhaleID'] != '0'].copy()
    if len(_labeled) > 50:
        _ici_matrix = _labeled[ICI_COLS].fillna(0).values
        _features = np.column_stack([
            _ici_matrix,
            _labeled['nClicks'].values.reshape(-1, 1),
            _labeled['Length'].values.reshape(-1, 1),
        ])
        WHALE_SCALER = StandardScaler()
        _features_scaled = WHALE_SCALER.fit_transform(_features)
        WHALE_CLASSIFIER = KNeighborsClassifier(n_neighbors=7, weights='distance', metric='euclidean')
        WHALE_CLASSIFIER.fit(_features_scaled, _labeled['WhaleID'].values)


def identify_coda_from_icis(icis, n_clicks, length):
    """Identify most likely whale from ICI features using k-NN.

    Returns list of (whale_id, probability) sorted by decreasing probability.
    """
    if WHALE_CLASSIFIER is None or WHALE_SCALER is None:
        return []

    ici_vec = np.zeros(9)
    for i, val in enumerate(icis[:9]):
        ici_vec[i] = val

    features = np.concatenate([ici_vec, [n_clicks, length]]).reshape(1, -1)
    features_scaled = WHALE_SCALER.transform(features)

    probas = WHALE_CLASSIFIER.predict_proba(features_scaled)[0]
    classes = WHALE_CLASSIFIER.classes_

    results = sorted(zip(classes, probas), key=lambda x: -x[1])
    return [(wid, float(p)) for wid, p in results if p > 0.01]


# --- Classifieur d'activite vocale ---

def classify_vocal_activity(audio_path):
    """Analyse an audio file and classify sperm whale vocal activity.

    Returns a list of segments: [(start_s, end_s, activity_type, details), ...]
    Activity types: ECHOLOCATION, CODA, CREAK, SILENCE
    """
    from pydub import AudioSegment
    from scipy.signal import butter, filtfilt, find_peaks
    import tempfile

    try:
        wav_path, duration_s, is_tmp = _convert_to_wav(audio_path)
    except Exception as e:
        return [], f"Erreur: {e}", None

    sr, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data[:, 0]
    data = data.astype(np.float64)
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val

    if is_tmp:
        os.unlink(wav_path)

    nyq = sr / 2
    b, a = butter(4, [max(2000 / nyq, 0.001), min(24000 / nyq, 0.999)], btype='band')
    filtered = filtfilt(b, a, data)
    tkeo = teager_kaiser(filtered)
    tkeo = np.maximum(tkeo, 0)
    tkeo_max = np.max(tkeo)
    if tkeo_max > 0:
        tkeo = tkeo / tkeo_max

    peaks, props = find_peaks(tkeo, height=0.08, distance=int(sr * 0.005))
    click_times = peaks / sr

    if len(click_times) < 2:
        return [{'start': 0, 'end': duration_s, 'type': 'SILENCE',
                 'details': 'Aucune activite acoustique detectee'}], "", data

    window_s = 3.0
    step_s = 1.0
    segments = []

    t = 0.0
    while t < duration_s:
        t_end = min(t + window_s, duration_s)
        mask = (click_times >= t) & (click_times < t_end)
        window_clicks = click_times[mask]

        if len(window_clicks) < 2:
            segments.append({
                'start': t, 'end': t_end, 'type': 'SILENCE',
                'click_count': len(window_clicks),
                'details': ''
            })
        else:
            icis = np.diff(window_clicks)
            mean_ici = np.mean(icis)
            std_ici = np.std(icis)
            click_rate = len(window_clicks) / (t_end - t)
            cv = std_ici / mean_ici if mean_ici > 0 else 0

            if mean_ici < 0.030:
                activity = 'CREAK'
                details = (f"Buzz/Creak — {len(window_clicks)} clics, "
                           f"rythme={click_rate:.0f} clics/s, ICI={mean_ici*1000:.0f}ms")
            elif mean_ici < 0.5 and cv < 1.0 and len(window_clicks) >= 3:
                activity = 'CODA'
                details = (f"Codas — {len(window_clicks)} clics, "
                           f"ICI moy={mean_ici*1000:.0f}ms")
            elif mean_ici >= 0.3 and cv < 0.5:
                activity = 'ECHOLOCATION'
                details = (f"Echolocation — {len(window_clicks)} clics, "
                           f"ICI moy={mean_ici*1000:.0f}ms, regulier (CV={cv:.2f})")
            elif mean_ici >= 0.3:
                activity = 'ECHOLOCATION'
                details = (f"Clics reguliers — {len(window_clicks)} clics, "
                           f"ICI moy={mean_ici*1000:.0f}ms")
            else:
                activity = 'CODA'
                details = (f"Activite vocale — {len(window_clicks)} clics, "
                           f"ICI moy={mean_ici*1000:.0f}ms")

            segments.append({
                'start': t, 'end': t_end, 'type': activity,
                'click_count': len(window_clicks),
                'mean_ici': mean_ici,
                'std_ici': std_ici,
                'click_rate': click_rate,
                'details': details,
            })

        t += step_s

    merged = _merge_segments(segments)
    return merged, "", data


def _merge_segments(segments):
    """Merge consecutive segments of the same type."""
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        if seg['type'] == merged[-1]['type']:
            merged[-1]['end'] = seg['end']
            merged[-1]['click_count'] = merged[-1].get('click_count', 0) + seg.get('click_count', 0)
            if 'mean_ici' in seg and 'mean_ici' in merged[-1]:
                merged[-1]['mean_ici'] = (merged[-1]['mean_ici'] + seg['mean_ici']) / 2
        else:
            merged.append(seg.copy())
    return merged


def estimate_whale_size(click_times, sr_data, data):
    """Estimate whale body length from IPI in regular echolocation clicks.

    The IPI (Inter-Pulse Interval) is the time between the direct pulse
    and its first reflection inside the spermaceti organ.
    Body length (m) ≈ IPI (ms) * 4.833 + 1.453  (Growcott et al. 2011)
    """
    if len(click_times) < 3:
        return None, None

    icis = np.diff(click_times)
    regular_mask = (icis > 0.3) & (icis < 3.0)
    if np.sum(regular_mask) < 2:
        return None, None

    median_ici = np.median(icis[regular_mask])

    ipi_estimate_ms = None
    for ipi_test in np.arange(1.0, 10.0, 0.1):
        ipi_s = ipi_test / 1000.0
        if ipi_s < median_ici * 0.8:
            ipi_estimate_ms = ipi_test
            break

    if ipi_estimate_ms is None:
        ipi_estimate_ms = 5.0

    body_length_m = ipi_estimate_ms * 4.833 + 1.453
    if body_length_m < 4 or body_length_m > 20:
        return median_ici * 1000, None

    return median_ici * 1000, body_length_m


def analyze_vocal_activity(audio_path):
    """Full vocal activity analysis pipeline for the Gradio UI."""
    if audio_path is None:
        return None, "Uploadez un fichier audio ou video (WAV, MP3, MP4, OGG, FLAC...)."

    segments, error, data = classify_vocal_activity(audio_path)
    if error:
        return None, error
    if not segments:
        return None, "Aucune activite detectee."

    activity_counts = {}
    for seg in segments:
        t = seg['type']
        dur = seg['end'] - seg['start']
        activity_counts[t] = activity_counts.get(t, 0) + dur

    total_dur = segments[-1]['end'] - segments[0]['start']

    colors = {
        'ECHOLOCATION': '#3498db',
        'CODA': '#2ecc71',
        'CREAK': '#e74c3c',
        'SILENCE': '#555555',
    }
    labels_fr = {
        'ECHOLOCATION': 'Echolocation (sonar)',
        'CODA': 'Codas (communication)',
        'CREAK': 'Creaks (chasse)',
        'SILENCE': 'Silence',
    }
    icons = {
        'ECHOLOCATION': '📡',
        'CODA': '💬',
        'CREAK': '🎯',
        'SILENCE': '🔇',
    }

    fig, axes = plt.subplots(2, 1, figsize=(14, 5),
                             gridspec_kw={'height_ratios': [1, 3]})
    fig.patch.set_facecolor('#1a1a2e')

    ax_timeline = axes[0]
    ax_timeline.set_facecolor('#16213e')
    for seg in segments:
        color = colors.get(seg['type'], '#555')
        ax_timeline.axvspan(seg['start'], seg['end'],
                            alpha=0.7, color=color)
    ax_timeline.set_xlim(segments[0]['start'], segments[-1]['end'])
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_yticks([])
    ax_timeline.set_title("Timeline d'activite vocale", color='#e0e0e0',
                          fontsize=12, pad=8)
    ax_timeline.tick_params(colors='#999')

    import matplotlib.patches as mpatches
    handles = []
    for act_type in ['ECHOLOCATION', 'CODA', 'CREAK', 'SILENCE']:
        if act_type in activity_counts:
            handles.append(mpatches.Patch(
                color=colors[act_type],
                label=labels_fr[act_type]
            ))
    ax_timeline.legend(handles=handles, loc='upper right',
                       fontsize=8, facecolor='#1a1a2e',
                       edgecolor='#444', labelcolor='#e0e0e0')

    ax_wave = axes[1]
    ax_wave.set_facecolor('#16213e')
    if data is not None:
        sr = 44100
        t_axis = np.arange(len(data)) / sr
        step = max(1, len(data) // 10000)
        ax_wave.plot(t_axis[::step], data[::step], color='#00d4aa',
                     linewidth=0.3, alpha=0.7)

        for seg in segments:
            color = colors.get(seg['type'], '#555')
            ax_wave.axvspan(seg['start'], seg['end'], alpha=0.15, color=color)

    ax_wave.set_xlabel("Temps (s)", color='#999', fontsize=10)
    ax_wave.set_ylabel("Amplitude", color='#999', fontsize=10)
    ax_wave.tick_params(colors='#999')
    ax_wave.set_xlim(segments[0]['start'], segments[-1]['end'])

    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#444')

    fig.tight_layout()

    md = "## Analyse de l'activite vocale\n\n"

    md += "### Resume\n\n"
    md += "| Activite | Duree | Proportion |\n"
    md += "|----------|-------|------------|\n"
    for act_type in ['CODA', 'ECHOLOCATION', 'CREAK', 'SILENCE']:
        if act_type in activity_counts:
            dur = activity_counts[act_type]
            pct = dur / total_dur * 100
            icon = icons[act_type]
            label = labels_fr[act_type]
            md += f"| {icon} {label} | {dur:.1f}s | {pct:.0f}% |\n"

    md += f"\n*Duree totale analysee : {total_dur:.1f}s*\n\n"

    md += "### Detail par segment\n\n"
    for i, seg in enumerate(segments, 1):
        icon = icons.get(seg['type'], '')
        label = labels_fr.get(seg['type'], seg['type'])
        dur = seg['end'] - seg['start']
        t_min = int(seg['start'] // 60)
        t_sec = seg['start'] % 60
        md += f"**{i}. {icon} {label}** — {t_min}m{t_sec:04.1f}s → +{dur:.1f}s"
        if seg.get('click_count', 0) > 0:
            md += f" | {seg['click_count']} clics"
        if 'mean_ici' in seg:
            md += f" | ICI moy={seg['mean_ici']*1000:.0f}ms"
        md += "\n\n"

    if 'ECHOLOCATION' in activity_counts:
        md += "---\n### Analyse de l'echolocation\n\n"
        md += ("Les clics d'echolocation sont un **sonar biologique** : "
               "le cachalot emet un clic puissant qui rebondit sur les "
               "objets environnants (proies, fond marin). L'intervalle "
               "entre les clics (ICI ~0.5-2s) correspond au temps d'aller-retour "
               "du son, et diminue quand la proie est plus proche.\n\n")

        echo_segs = [s for s in segments if s['type'] == 'ECHOLOCATION']
        if echo_segs:
            avg_ici = np.mean([s.get('mean_ici', 0) for s in echo_segs if 'mean_ici' in s])
            if avg_ici > 0:
                depth_est = avg_ici * 750
                md += (f"- **ICI moyen** : {avg_ici*1000:.0f}ms\n"
                       f"- **Profondeur estimee de la cible** : ~{depth_est:.0f}m "
                       f"(ICI × vitesse du son / 2)\n")
                if avg_ici > 1.0:
                    md += "- **Interpretation** : chasse en eau profonde\n"
                elif avg_ici > 0.5:
                    md += "- **Interpretation** : approche d'une cible\n"
                else:
                    md += "- **Interpretation** : cible proche, pre-capture\n"
                md += "\n"

    if 'CREAK' in activity_counts:
        md += "---\n### Creaks detectes !\n\n"
        md += ("Les **creaks** (aussi appeles buzz) sont des rafales de clics "
               "ultra-rapides (>30 clics/seconde) emises juste avant la capture "
               "d'une proie. C'est l'equivalent du **buzz terminal** des "
               "chauves-souris. Leur presence indique une **tentative de chasse active**.\n\n")

    if 'CODA' in activity_counts and WHALE_CLASSIFIER is not None:
        md += "---\n### Codas detectes — identification en cours...\n\n"
        md += ("Les segments de codas peuvent etre analyses dans l'onglet "
               "**Identifier un coda** pour retrouver l'individu.\n")

    return fig, md


def _convert_to_wav(audio_path):
    """Convert any audio format to mono WAV 44100Hz. Returns (tmp_path, duration_s)."""
    from pydub import AudioSegment
    import tempfile

    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ('.wav',):
        sr, data = wavfile.read(audio_path)
        return audio_path, len(data) / sr, False

    audio = AudioSegment.from_file(audio_path)
    mono = audio.set_channels(1).set_frame_rate(44100)
    tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    mono.export(tmp.name, format='wav')
    return tmp.name, len(mono) / 1000.0, True


def identify_from_audio(audio_path, det_threshold=0.3, snr_threshold=10):
    """Full pipeline: detect codas in audio then identify each one.

    Supports WAV, MP3, MP4, OGG, FLAC, M4A, etc. (anything pydub/ffmpeg handles).
    Long files (>30s) are processed in segments to avoid memory issues.
    """
    if audio_path is None:
        return None, "Uploadez un fichier audio ou video (WAV, MP3, MP4, OGG, FLAC...)."

    try:
        wav_path, duration_s, is_tmp = _convert_to_wav(audio_path)
    except Exception as e:
        return None, f"Erreur de conversion audio : {e}"

    params = DetectorParams(
        detection_threshold=det_threshold,
        snr_threshold=snr_threshold,
    )

    SEGMENT_MAX_S = 30
    all_codas = []
    all_results = []

    try:
        if duration_s <= SEGMENT_MAX_S:
            codas = detect_codas(wav_path, params)
            if codas:
                all_codas = codas
                all_results = codas_to_dict(codas)
        else:
            from pydub import AudioSegment
            import tempfile
            audio = AudioSegment.from_wav(wav_path)
            segment_ms = SEGMENT_MAX_S * 1000
            n_segments = int(np.ceil(len(audio) / segment_ms))

            for seg_i in range(n_segments):
                start_ms = seg_i * segment_ms
                end_ms = min(start_ms + segment_ms, len(audio))
                seg = audio[start_ms:end_ms]

                seg_tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                seg.export(seg_tmp.name, format='wav')

                try:
                    codas = detect_codas(seg_tmp.name, params)
                    if codas:
                        results = codas_to_dict(codas)
                        t_offset = start_ms / 1000.0
                        for coda in codas:
                            for click in coda.clicks:
                                click.time += t_offset
                        all_codas.extend(codas)
                        all_results.extend(results)
                finally:
                    os.unlink(seg_tmp.name)
    finally:
        if is_tmp:
            os.unlink(wav_path)

    if not all_codas:
        return None, "Aucun coda detecte dans cet enregistrement."

    ext = os.path.splitext(audio_path)[1].lower()
    format_info = f"Format: {ext.upper().strip('.')} | " if ext != '.wav' else ""
    md = f"## {len(all_codas)} coda(s) detecte(s)\n"
    md += f"*{format_info}Duree: {duration_s:.1f}s"
    if duration_s > SEGMENT_MAX_S:
        md += f" (traite en {int(np.ceil(duration_s / SEGMENT_MAX_S))} segments)"
    md += "*\n\n"

    for i, (coda, r) in enumerate(zip(all_codas, all_results)):
        icis = r['icis']
        n_clicks = r['n_clicks']
        duration = r['duration']
        t_start = min(c.time for c in coda.clicks)

        icis_str = ", ".join(f"{ici*1000:.0f}ms" for ici in icis)
        t_min = int(t_start // 60)
        t_sec = t_start % 60
        md += f"### Coda {i+1} *(t={t_min}m{t_sec:04.1f}s)*\n"
        md += f"- **Clics**: {n_clicks} | **Duree**: {duration*1000:.0f}ms\n"
        md += f"- **ICIs**: [{icis_str}]\n\n"

        candidates = identify_coda_from_icis(icis, n_clicks, duration)

        if not candidates:
            md += "> Classifieur non disponible.\n\n"
            continue

        md += "| Rang | Individu | Confiance |\n"
        md += "|------|----------|----------|\n"

        for rank, (wid, prob) in enumerate(candidates[:5], 1):
            name = get_whale_display_name(wid)
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            md += f"| {rank} | {name} | {bar} {prob*100:.1f}% |\n"

        best_wid, best_prob = candidates[0]
        best_name, best_desc = WHALE_NAMES.get(best_wid, (f"Whale {best_wid}", ""))
        if best_prob > 0.4:
            md += f"\n> **Meilleur match** : {best_name} ({best_prob*100:.0f}%)"
            if best_desc:
                md += f" — *{best_desc}*"
            md += "\n\n"
        elif best_prob > 0.2:
            md += f"\n> Match possible : {best_name} ({best_prob*100:.0f}%) — confiance moderee\n\n"
        else:
            md += "\n> Confiance trop faible — possiblement un individu inconnu\n\n"

    if len(all_codas) > 1:
        md += "---\n### Synthese multi-codas\n\n"
        vote_counts = {}
        for r in all_results:
            candidates = identify_coda_from_icis(r['icis'], r['n_clicks'], r['duration'])
            for wid, prob in candidates:
                vote_counts[wid] = vote_counts.get(wid, 0) + prob

        if vote_counts:
            total = sum(vote_counts.values())
            sorted_votes = sorted(vote_counts.items(), key=lambda x: -x[1])
            md += "En combinant tous les codas :\n\n"
            md += "| Individu | Score cumule |\n"
            md += "|----------|------------|\n"
            for wid, score in sorted_votes[:7]:
                name = get_whale_display_name(wid)
                pct = score / total * 100
                md += f"| {name} | {pct:.1f}% |\n"

            top = sorted_votes[:3]
            if len(top) >= 2 and top[1][1] / total > 0.15:
                md += "\n> **Multi-individus probable** — au moins 2 voix distinctes detectees\n"

            best_wid = sorted_votes[0][0]
            best_name, _ = WHALE_NAMES.get(best_wid, (f"Whale {best_wid}", ""))
            md += f"\n> **Identification principale** : **{best_name}**\n"

    return _build_identification_plot(all_codas, all_results), md


def _build_identification_plot(codas, results):
    """Build a visualization of detected codas with ICI patterns."""
    fig, axes = plt.subplots(len(codas), 1, figsize=(10, 3 * len(codas)),
                             squeeze=False)
    fig.patch.set_facecolor('#1a1a2e')

    for i, (coda, r) in enumerate(zip(codas, results)):
        ax = axes[i, 0]
        ax.set_facecolor('#16213e')

        icis = r['icis']
        clicks_t = [0]
        for ici in icis:
            clicks_t.append(clicks_t[-1] + ici)

        ax.stem(clicks_t, [1] * len(clicks_t), linefmt='#00d4aa',
                markerfmt='o', basefmt='none')
        ax.set_title(f"Coda {i+1} — {r['n_clicks']} clics, {r['duration']*1000:.0f}ms",
                     color='#e0e0e0', fontsize=11)
        ax.set_xlabel("Temps (s)", color='#999', fontsize=9)
        ax.set_ylabel("Clic", color='#999', fontsize=9)
        ax.set_ylim(0, 1.5)
        ax.tick_params(colors='#999')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#444')

    fig.tight_layout()
    return fig


def build_gero_plotly(color_by="CodaName"):
    """Build interactive Plotly scatter plot for the Gero dataset."""
    if GERO_DF is None or GERO_EMBEDDING is None:
        fig = go.Figure()
        fig.update_layout(title="Dataset non charge")
        return fig

    title_map = {
        "CodaName": "Par type de coda",
        "UnitName": "Par unite sociale",
        "WhaleID": "Par individu",
        "Year": "Par annee",
    }

    fig = go.Figure()

    if color_by == "WhaleID":
        groups = GERO_DF['WhaleID'].apply(
            lambda x: get_whale_display_name(x) if str(x) != '0' else "Non identifie"
        ).values
    elif color_by == "UnitName":
        groups = ("Unit " + GERO_DF['UnitName']).values
    elif color_by == "Year":
        groups = GERO_DF['Year'].astype(str).values
    else:
        groups = GERO_DF[color_by].values

    unique_groups = sorted(set(groups))
    palette = [
        "#9E0142", "#D53E4F", "#F46D43", "#FDAE61", "#FEE08B",
        "#E6F598", "#ABDDA4", "#66C2A5", "#3288BD", "#5E4FA2",
        "#8073AC", "#B2ABD2", "#D8DAEB", "#FDB863", "#E08214",
        "#B35806", "#542788", "#998EC3", "#F1A340", "#01665E",
    ]

    for gi, grp in enumerate(unique_groups):
        mask = groups == grp
        indices = np.where(mask)[0]

        hover_texts = []
        for i in indices:
            row = GERO_DF.iloc[i]
            whale_str = get_whale_display_name(row['WhaleID']) if str(row['WhaleID']) != '0' else "Non id."
            hover_texts.append(
                f"<b>{grp}</b><br>"
                f"Type: {row['CodaName']}<br>"
                f"Unit: {row['UnitName']}<br>"
                f"Individu: {whale_str}<br>"
                f"Clics: {row['nClicks']}<br>"
                f"Index: {i}"
            )

        fig.add_trace(go.Scatter(
            x=GERO_EMBEDDING[mask, 0],
            y=GERO_EMBEDDING[mask, 1],
            mode='markers',
            marker=dict(
                size=6,
                color=palette[gi % len(palette)],
                opacity=0.7,
                line=dict(width=0.3, color='white'),
            ),
            name=str(grp),
            text=hover_texts,
            hoverinfo='text',
            customdata=indices.tolist(),
        ))

    fig.update_layout(
        title=dict(
            text=f"Codas de cachalots — {title_map.get(color_by, '')} (Gero et al.)",
            font=dict(size=16),
        ),
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="#444",
            borderwidth=1,
            font=dict(size=10),
        ),
        height=600,
        margin=dict(l=50, r=30, t=50, b=50),
        hovermode='closest',
    )
    return fig


def on_gero_color_change(color_by):
    """Update scatter plot when color-by changes."""
    col_map = {
        "Type de coda": "CodaName",
        "Unite sociale": "UnitName",
        "Individu": "WhaleID",
        "Annee": "Year",
    }
    col = col_map.get(color_by, "CodaName")
    fig = build_gero_plotly(col)
    return fig, get_gero_summary(col)


def get_gero_summary(color_by="CodaName"):
    """Summary stats for the Gero dataset."""
    if GERO_DF is None:
        return "Dataset non charge."

    md = f"### Dataset Gero et al. (2015)\n\n"
    md += f"- **Codas**: {len(GERO_DF)}\n"
    md += f"- **Types**: {GERO_DF['CodaName'].nunique()}\n"
    md += f"- **Unites sociales**: {GERO_DF['Unit'].nunique()}\n"
    md += f"- **Individus identifies**: {GERO_DF[GERO_DF['WhaleID'] != '0']['WhaleID'].nunique()}\n"
    md += f"- **Periode**: {GERO_DF['Year'].min()}-{GERO_DF['Year'].max()}\n\n"

    if color_by == "CodaName":
        md += "**Types les plus frequents:**\n"
        for name, count in GERO_DF['CodaName'].value_counts().head(8).items():
            pct = 100 * count / len(GERO_DF)
            md += f"- {name}: {count} ({pct:.1f}%)\n"
    elif color_by == "UnitName":
        md += "**Unites sociales:**\n"
        for unit in sorted(GERO_DF['UnitName'].unique()):
            count = (GERO_DF['UnitName'] == unit).sum()
            md += f"- Unit {unit}: {count} codas\n"
    elif color_by == "WhaleID":
        md += "**Individus identifies:**\n"
        identified = GERO_DF[GERO_DF['WhaleID'] != '0']
        for wid, count in identified['WhaleID'].value_counts().head(10).items():
            unit = identified[identified['WhaleID'] == wid]['UnitName'].iloc[0]
            display = get_whale_display_name(wid)
            md += f"- {display} (Unit {unit}): {count} codas\n"
        unid = (GERO_DF['WhaleID'] == '0').sum()
        md += f"- Non identifies: {unid}\n"
    elif color_by == "Year":
        md += "**Par annee:**\n"
        for year in sorted(GERO_DF['Year'].unique()):
            count = (GERO_DF['Year'] == year).sum()
            md += f"- {year}: {count} codas\n"

    return md


def get_whale_display_name(whale_id):
    """Return display name for a whale: 'Nom (ID)'."""
    wid = str(whale_id)
    if wid in WHALE_NAMES:
        return f"{WHALE_NAMES[wid][0]} (#{wid})"
    if wid == "0":
        return "Non identifie"
    return f"Whale #{wid}"


def on_gero_point_click(clicked_idx_str):
    """Handle click on Gero scatter plot via JS bridge."""
    if GERO_DF is None:
        return "Dataset non charge."

    try:
        if not clicked_idx_str or clicked_idx_str.strip() == "":
            return "Cliquez sur un point pour voir ses details."

        idx = _parse_click_value(clicked_idx_str)
        if idx < 0 or idx >= len(GERO_DF):
            return f"Index {idx} hors limites."

        row = GERO_DF.iloc[idx]
        ici_cols = ['ICI1', 'ICI2', 'ICI3', 'ICI4', 'ICI5', 'ICI6', 'ICI7', 'ICI8', 'ICI9']
        icis = [row[c] for c in ici_cols if row[c] > 0]
        icis_str = ", ".join(f"{ici*1000:.0f}ms" for ici in icis)

        wid = str(row['WhaleID'])
        whale_display = get_whale_display_name(wid)
        whale_desc = ""
        if wid in WHALE_NAMES:
            whale_desc = f"\n  *{WHALE_NAMES[wid][1]}*"

        md = f"### Coda #{row['CodaNumber']}\n\n"
        md += f"- **Type**: {row['CodaName']}\n"
        md += f"- **Unite sociale**: Unit {row['UnitName']}\n"
        md += f"- **Individu**: {whale_display}{whale_desc}\n"
        md += f"- **Clics**: {row['nClicks']}\n"
        md += f"- **Duree**: {row['Length']*1000:.0f} ms\n"
        md += f"- **ICIs**: [{icis_str}]\n"
        md += f"- **Date**: {row['Date'].strftime('%Y-%m-%d')}\n"

        return md
    except Exception as e:
        return f"Erreur: {e}"


def build_whale_profile(whale_choice):
    """Build a full profile for a selected whale."""
    if GERO_DF is None or GERO_EMBEDDING is None:
        return "", None

    if not whale_choice or whale_choice == "Tous les individus":
        fig = build_gero_plotly("WhaleID")
        return "Selectionnez un individu pour voir son profil.", fig

    wid = whale_choice.split("#")[-1].rstrip(")")
    if wid not in GERO_DF['WhaleID'].values:
        return f"Individu {wid} non trouve.", ""

    sub = GERO_DF[GERO_DF['WhaleID'] == wid]
    name, desc = WHALE_NAMES.get(wid, (f"Whale {wid}", ""))
    unit = sub['UnitName'].iloc[0]

    md = f"## {name}\n"
    md += f"*{desc}*\n\n"
    md += f"**Identifiant scientifique**: #{wid}\n\n"
    md += f"**Unite sociale**: Unit {unit}\n\n"

    years = sorted(sub['Year'].unique())
    year_range = f"{min(years)} - {max(years)}" if len(years) > 1 else str(years[0])
    md += f"**Periode d'observation**: {year_range} ({len(years)} annee{'s' if len(years) > 1 else ''})\n\n"
    md += f"**Nombre de codas enregistres**: {len(sub)}\n\n"
    md += f"**Clics par coda (moyenne)**: {sub['nClicks'].mean():.1f}\n\n"
    md += f"**Duree moyenne d'un coda**: {sub['Length'].mean()*1000:.0f} ms\n\n"

    md += "### Repertoire vocal\n\n"
    md += "| Type de coda | Nombre | Proportion |\n"
    md += "|:------------|-------:|-----------:|\n"
    for ctype, count in sub['CodaName'].value_counts().items():
        pct = 100 * count / len(sub)
        bar = "█" * int(pct / 5)
        md += f"| {ctype} | {count} | {bar} {pct:.1f}% |\n"

    md += "\n### Activite par annee\n\n"
    for year in sorted(sub['Year'].unique()):
        yr_sub = sub[sub['Year'] == year]
        md += f"- **{year}**: {len(yr_sub)} codas"
        top = yr_sub['CodaName'].value_counts().head(2)
        types_str = ", ".join(f"{t}({c})" for t, c in top.items())
        md += f" — {types_str}\n"

    family = GERO_DF[(GERO_DF['UnitName'] == unit) & (GERO_DF['WhaleID'] != '0') & (GERO_DF['WhaleID'] != wid)]
    if len(family) > 0:
        relatives = family['WhaleID'].unique()
        md += f"\n### Famille (Unit {unit})\n\n"
        for rel_wid in sorted(relatives, key=str):
            rel_name = get_whale_display_name(rel_wid)
            rel_count = (family['WhaleID'] == rel_wid).sum()
            md += f"- {rel_name}: {rel_count} codas\n"

    # Plotly: highlight this whale's points
    fig = go.Figure()

    others_mask = GERO_DF['WhaleID'] != wid
    fig.add_trace(go.Scatter(
        x=GERO_EMBEDDING[others_mask, 0],
        y=GERO_EMBEDDING[others_mask, 1],
        mode='markers',
        marker=dict(size=4, color='#555', opacity=0.2),
        name="Autres",
        hoverinfo='skip',
    ))

    whale_mask = GERO_DF['WhaleID'] == wid
    whale_indices = np.where(whale_mask)[0]
    hover_texts = []
    for i in whale_indices:
        row = GERO_DF.iloc[i]
        hover_texts.append(
            f"<b>{name}</b><br>"
            f"Type: {row['CodaName']}<br>"
            f"Clics: {row['nClicks']}<br>"
            f"Date: {row['Date'].strftime('%Y-%m-%d')}"
        )

    palette = {
        "A": "#D53E4F", "B": "#F46D43", "F": "#66C2A5",
        "J": "#3288BD", "N": "#FDAE61", "R": "#9E0142",
        "S": "#5E4FA2", "T": "#FEE08B", "U": "#E6F598",
    }
    whale_color = palette.get(unit, "#00FFAA")

    fig.add_trace(go.Scatter(
        x=GERO_EMBEDDING[whale_mask, 0],
        y=GERO_EMBEDDING[whale_mask, 1],
        mode='markers',
        marker=dict(
            size=10,
            color=whale_color,
            opacity=0.9,
            line=dict(width=1, color='white'),
            symbol='circle',
        ),
        name=name,
        text=hover_texts,
        hoverinfo='text',
        customdata=whale_indices.tolist(),
    ))

    fig.update_layout(
        title=dict(
            text=f"{name} — {len(sub)} codas dans l'espace UMAP",
            font=dict(size=16),
        ),
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(size=11),
        ),
        height=550,
        margin=dict(l=50, r=30, t=50, b=50),
        hovermode='closest',
    )

    return md, fig


def run_detector(audio_file, det_threshold, snr_threshold):
    """Run coda detector on uploaded audio file."""
    if audio_file is None:
        return None, "Veuillez uploader un fichier audio WAV."

    params = DetectorParams(
        detection_threshold=det_threshold,
        snr_threshold=snr_threshold,
    )

    codas = detect_codas(audio_file, params=params)
    results = codas_to_dict(codas)

    fig = make_detection_plot(audio_file, codas)

    if not results:
        return fig, "### Aucun coda detecte\n\nEssayez de baisser les seuils de detection."

    md = f"### {len(results)} coda(s) detecte(s)\n\n"
    md += "| # | Clics | Debut (s) | Duree (ms) | ICIs (ms) | IPI moy. (ms) | SNR moy. |\n"
    md += "|---|-------|-----------|------------|-----------|---------------|----------|\n"
    for r in results:
        icis_str = ", ".join(f"{ici*1000:.0f}" for ici in r['icis'])
        md += (f"| {r['coda_id']} | {r['n_clicks']} | {r['start_time']:.2f} | "
               f"{r['duration']*1000:.0f} | {icis_str} | "
               f"{r['mean_ipi']:.1f} | {r['mean_snr']:.0f} dB |\n")

    return fig, md


def make_detection_plot(audio_path, codas):
    """Visualize detected codas on the audio waveform."""
    try:
        sr, data = wavfile.read(audio_path)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float64)

        fig, axes = plt.subplots(3, 1, figsize=(12, 7),
                                 gridspec_kw={'height_ratios': [2, 1, 2]})
        fig.patch.set_facecolor('#1a1a2e')

        t = np.arange(len(data)) / sr

        axes[0].plot(t, data, color='#66C2A5', linewidth=0.3, alpha=0.7)
        axes[0].set_ylabel("Amplitude", color='#e0e0e0', fontsize=9)
        axes[0].set_title("Signal audio + codas detectes", color='#e0e0e0',
                          fontsize=12, fontweight='bold')

        colors = ['#D53E4F', '#F46D43', '#FDAE61', '#66C2A5', '#3288BD',
                  '#9E0142', '#5E4FA2', '#ABDDA4', '#FEE08B', '#E6F598']
        for i, coda in enumerate(codas):
            color = colors[i % len(colors)]
            for click in coda.clicks:
                axes[0].axvline(click.time, color=color, alpha=0.7,
                                linewidth=1.5, linestyle='-')
            if coda.clicks:
                t_start = min(c.time for c in coda.clicks)
                t_end = max(c.time for c in coda.clicks)
                axes[0].axvspan(t_start - 0.01, t_end + 0.01,
                                alpha=0.15, color=color)
                axes[0].text(t_start, axes[0].get_ylim()[1] * 0.9,
                             f"C{i+1}", color=color, fontsize=8, fontweight='bold')

        from scipy.signal import butter, filtfilt
        nyq = sr / 2
        low = max(2000 / nyq, 0.001)
        high = min(24000 / nyq, 0.999)
        b, a = butter(4, [low, high], btype='band')
        filtered = filtfilt(b, a, data / (np.max(np.abs(data)) + 1e-10))
        tkeo = teager_kaiser(filtered)
        tkeo = np.maximum(tkeo, 0)
        tkeo_max = np.max(tkeo)
        if tkeo_max > 0:
            tkeo = tkeo / tkeo_max
        t_tkeo = np.arange(len(tkeo)) / sr

        axes[1].plot(t_tkeo, tkeo, color='#FDB863', linewidth=0.3, alpha=0.8)
        axes[1].set_ylabel("TKEO", color='#e0e0e0', fontsize=9)
        axes[1].set_ylim(0, 1.05)

        axes[2].specgram(data, Fs=sr, cmap='inferno', NFFT=512, noverlap=384)
        axes[2].set_ylabel("Freq (Hz)", color='#e0e0e0', fontsize=9)
        axes[2].set_xlabel("Temps (s)", color='#e0e0e0', fontsize=9)
        axes[2].set_ylim(0, min(sr // 2, 12000))

        for i, coda in enumerate(codas):
            color = colors[i % len(colors)]
            for click in coda.clicks:
                axes[2].axvline(click.time, color=color, alpha=0.5,
                                linewidth=1, linestyle='--')

        for ax in axes:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='#999')
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color('#444')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.set_xlim(0, t[-1])

        fig.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.text(0.5, 0.5, f"Erreur: {e}", transform=ax.transAxes,
                ha='center', va='center', color='#ff6b6b', fontsize=12)
        return fig


def run_detector_on_dataset(coda_index, det_threshold, snr_threshold):
    """Run detector on a coda from the dataset."""
    try:
        idx = int(coda_index)
        if idx < 0 or idx >= len(FILENAMES):
            return None, f"Index hors limites (0-{len(FILENAMES)-1})"
        filepath = FILENAMES[idx]
        if not os.path.exists(filepath):
            return None, f"Fichier introuvable: {filepath}"
        return run_detector(filepath, det_threshold, snr_threshold)
    except ValueError:
        return None, "Entrez un index valide."


PLOTLY_CLICK_HEAD = """
<script>
(function() {
    var PLOT_MAP = {
        'main-scatter': 'click_bridge',
        'gero-scatter': 'gero_click_bridge'
    };

    function setGradioInput(elemId, value) {
        var el = document.getElementById(elemId);
        if (!el) return false;
        var inp = el.querySelector('textarea') || el.querySelector('input');
        if (!inp) return false;
        var setter = Object.getOwnPropertyDescriptor(
            Object.getPrototypeOf(inp), 'value'
        );
        if (setter && setter.set) {
            setter.set.call(inp, value);
        } else {
            inp.value = value;
        }
        inp.dispatchEvent(new Event('input', {bubbles: true}));
        inp.dispatchEvent(new Event('change', {bubbles: true}));
        return true;
    }

    function attachToPlot(plotElemId, bridgeElemId) {
        var container = document.getElementById(plotElemId);
        if (!container) return false;
        var plotDiv = container.querySelector('.js-plotly-plot');
        if (!plotDiv || !plotDiv.on) return false;
        if (plotDiv._wce_attached) return true;
        plotDiv.on('plotly_click', function(data) {
            if (!data || !data.points || data.points.length === 0) return;
            var idx = data.points[0].customdata;
            if (idx === undefined || idx === null) return;
            var payload = String(idx) + '_' + Date.now();
            setGradioInput(bridgeElemId, payload);
        });
        plotDiv._wce_attached = true;
        return true;
    }

    function attachAll() {
        var allOk = true;
        for (var plotId in PLOT_MAP) {
            if (!attachToPlot(plotId, PLOT_MAP[plotId])) {
                allOk = false;
            }
        }
        return allOk;
    }

    // Plotly plots may load asynchronously; keep trying
    var observer = new MutationObserver(function() { attachAll(); });
    observer.observe(document.body, {childList: true, subtree: true});
    setInterval(attachAll, 2000);
})();
</script>
"""


def build_app():
    """Construit l'application Gradio."""

    cluster_choices = ["Tous", "Bruit"] + [f"Cluster {i}" for i in range(N_CLUSTERS)]

    with gr.Blocks(title="Whale Coda Explorer") as app:
        gr.HTML("""
        <div class="main-header">
            <h1>Whale Coda Explorer</h1>
            <p>Exploration interactive des vocalisations de cachalots via WhAM (Project CETI)</p>
            <p style="font-size: 0.8em; color: #666;">
                620 codas analysees · 15 clusters · Detecteur de codas integre
            </p>
        </div>
        """)

        with gr.Tabs():
            with gr.Tab("Explorer les clusters"):
                with gr.Row():
                    with gr.Column(scale=3):
                        initial_fig = build_scatter_plot("Tous")
                        scatter_plot = gr.Plot(
                            value=initial_fig,
                            elem_id="main-scatter",
                        )
                    with gr.Column(scale=1):
                        cluster_filter = gr.Dropdown(
                            choices=cluster_choices,
                            value="Tous",
                            label="Filtrer par cluster",
                        )
                        cluster_info = gr.Markdown(
                            value=(
                                f"### Vue d'ensemble\n\n"
                                f"- **Total codas analysees**: {len(CLUSTER_LABELS)}\n"
                                f"- **Clusters decouverts**: {N_CLUSTERS}\n"
                                f"- **Non classes**: {(CLUSTER_LABELS == -1).sum()}\n\n"
                                f"Cliquez sur un point de la carte pour ecouter le coda."
                            ),
                        )
                        distribution_chart = gr.Plot(
                            value=build_distribution_chart(),
                            label="Distribution",
                        )

                gr.Markdown("---")
                gr.Markdown("### Ecouter et analyser un coda")

                click_bridge = gr.Textbox(
                    visible=False, elem_id="click_bridge",
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        coda_index_input = gr.Textbox(
                            label="Index du coda (0-619)",
                            placeholder="Entrez un index ou cliquez sur la carte...",
                            value="0",
                        )
                        random_btn = gr.Button(
                            "Coda aleatoire",
                            variant="secondary",
                            size="sm",
                        )
                        load_btn = gr.Button(
                            "Charger",
                            variant="primary",
                        )
                        coda_info = gr.Markdown(
                            "Cliquez sur un point de la carte ou sur **Charger**."
                        )

                    with gr.Column(scale=2):
                        audio_player = gr.Audio(
                            label="Ecouter le coda",
                            type="filepath",
                        )
                        spectrogram = gr.Plot(
                            label="Spectrogramme",
                        )

                click_bridge.change(
                    fn=on_plotly_click,
                    inputs=[click_bridge],
                    outputs=[audio_player, spectrogram, coda_info],
                )

                def on_cluster_filter_change(cluster_choice):
                    fig = build_scatter_plot(cluster_choice)
                    if cluster_choice == "Tous":
                        summary = "### Vue d'ensemble\n\n"
                        summary += f"- **Total codas analysees**: {len(CLUSTER_LABELS)}\n"
                        summary += f"- **Clusters decouverts**: {N_CLUSTERS}\n"
                        summary += f"- **Non classes**: {(CLUSTER_LABELS == -1).sum()}\n\n"
                        summary += "Cliquez sur un point de la carte pour ecouter le coda."
                    elif cluster_choice == "Bruit":
                        summary = get_cluster_summary(-1)
                    else:
                        cid = int(cluster_choice.replace("Cluster ", ""))
                        summary = get_cluster_summary(cid)
                    return fig, summary

                cluster_filter.change(
                    fn=on_cluster_filter_change,
                    inputs=[cluster_filter],
                    outputs=[scatter_plot, cluster_info],
                )

                load_btn.click(
                    fn=on_coda_select,
                    inputs=[cluster_filter, coda_index_input],
                    outputs=[audio_player, spectrogram, coda_info],
                )

                random_btn.click(
                    fn=get_random_coda,
                    inputs=[cluster_filter],
                    outputs=[coda_index_input, audio_player, spectrogram, coda_info],
                )

            with gr.Tab("Identite des cachalots"):
                gr.Markdown("""
                ### Qui parle ? — Identification par les codas
                Dataset [Gero, Whitehead & Rendell (2015)](https://doi.org/10.5061/dryad.ck4h0) :
                3876 codas des Caraibes orientales avec identification
                des individus, unites sociales et types de codas.
                Cliquez sur un point ou selectionnez un individu.
                """)

                if GERO_DF is not None and GERO_EMBEDDING is not None:
                    gero_color_choices = [
                        "Type de coda", "Unite sociale", "Individu", "Annee"
                    ]
                    whale_choices = ["Tous les individus"]
                    identified_ids = sorted(
                        [w for w in GERO_DF['WhaleID'].unique() if w != '0'],
                        key=str
                    )
                    for wid in identified_ids:
                        whale_choices.append(get_whale_display_name(wid))

                    with gr.Row():
                        with gr.Column(scale=3):
                            gero_initial_fig = build_gero_plotly("CodaName")
                            gero_scatter_plot = gr.Plot(
                                value=gero_initial_fig,
                                elem_id="gero-scatter",
                            )
                        with gr.Column(scale=1):
                            gero_color_by = gr.Dropdown(
                                choices=gero_color_choices,
                                value="Type de coda",
                                label="Colorer par",
                            )
                            whale_selector = gr.Dropdown(
                                choices=whale_choices,
                                value="Tous les individus",
                                label="Rechercher un individu",
                            )
                            gero_info = gr.Markdown(
                                value=get_gero_summary("CodaName"),
                            )
                            gero_click_bridge = gr.Textbox(
                                elem_id="gero_click_bridge",
                                visible=False,
                            )
                            gero_detail = gr.Markdown(
                                "Cliquez sur un point pour voir ses details.",
                            )

                    with gr.Row(visible=True):
                        whale_profile_md = gr.Markdown(
                            visible=False,
                        )

                    def on_gero_color_change_and_reset(color_by):
                        fig, summary = on_gero_color_change(color_by)
                        return (
                            fig,
                            summary,
                            gr.update(value="Tous les individus"),
                            gr.update(visible=False, value=""),
                        )

                    gero_color_by.change(
                        fn=on_gero_color_change_and_reset,
                        inputs=[gero_color_by],
                        outputs=[gero_scatter_plot, gero_info, whale_selector, whale_profile_md],
                    )

                    gero_click_bridge.change(
                        fn=on_gero_point_click,
                        inputs=[gero_click_bridge],
                        outputs=[gero_detail],
                    )

                    def on_whale_select(whale_choice):
                        md, fig = build_whale_profile(whale_choice)
                        show_profile = whale_choice != "Tous les individus"
                        return (
                            fig,
                            gr.update(value=md, visible=show_profile),
                            get_gero_summary("CodaName") if not show_profile else get_gero_summary("WhaleID"),
                        )

                    whale_selector.change(
                        fn=on_whale_select,
                        inputs=[whale_selector],
                        outputs=[gero_scatter_plot, whale_profile_md, gero_info],
                    )
                else:
                    gr.Markdown(
                        "Dataset Gero non disponible. "
                        "Lancez `python analyze_gero.py` pour generer les embeddings."
                    )

            with gr.Tab("Detecteur de codas"):
                gr.Markdown("""
                ### Detecteur automatique de codas
                Portage Python du [Coda-detector](https://github.com/Project-CETI/Coda-detector)
                de Project CETI. Utilise l'operateur Teager-Kaiser (TKEO) pour detecter
                les clics, puis un clustering par graphe pour grouper les clics en codas.
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("**Option 1 : Uploader un fichier**")
                        audio_upload = gr.Audio(
                            label="Fichier WAV a analyser",
                            type="filepath",
                        )
                        gr.Markdown("**Option 2 : Depuis le dataset**")
                        det_coda_index = gr.Textbox(
                            label="Index du coda (0-619)",
                            placeholder="Ex: 42",
                        )

                        gr.Markdown("**Parametres**")
                        det_threshold = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                            label="Seuil de detection TKEO",
                        )
                        snr_threshold_slider = gr.Slider(
                            minimum=3, maximum=40, value=10, step=1,
                            label="Seuil SNR (dB)",
                        )
                        detect_upload_btn = gr.Button(
                            "Detecter (fichier uploade)",
                            variant="primary",
                        )
                        detect_dataset_btn = gr.Button(
                            "Detecter (depuis dataset)",
                            variant="secondary",
                        )

                    with gr.Column(scale=2):
                        detection_plot = gr.Plot(
                            label="Visualisation des detections",
                        )
                        detection_results = gr.Markdown(
                            "Uploadez un fichier WAV ou choisissez un index, puis cliquez sur **Detecter**."
                        )

                detect_upload_btn.click(
                    fn=run_detector,
                    inputs=[audio_upload, det_threshold, snr_threshold_slider],
                    outputs=[detection_plot, detection_results],
                )

                detect_dataset_btn.click(
                    fn=run_detector_on_dataset,
                    inputs=[det_coda_index, det_threshold, snr_threshold_slider],
                    outputs=[detection_plot, detection_results],
                )

            with gr.Tab("Identifier un coda"):
                gr.Markdown("""
                ### Qui parle ? — Identification acoustique

                Uploadez un enregistrement de cachalot (WAV, **MP3**, **MP4**, OGG, FLAC, M4A...)
                et l'application va :
                1. **Convertir** automatiquement en WAV mono 44.1kHz si necessaire
                2. **Segmenter** les fichiers longs (>30s) pour un traitement optimal
                3. **Detecter** les codas (clics groupes) via le detecteur TKEO
                4. **Identifier** chaque coda parmi les 17 individus connus (k-NN)
                5. **Combiner** les resultats si plusieurs codas (synthese multi-individus)

                Le classifieur est entraine sur 1602 codas etiquetes du dataset
                Gero et al. (2015). Les fichiers MP3 de YouTube fonctionnent directement !

                *Plus vous fournissez de codas d'une meme session, plus l'identification
                est fiable.*
                """)

                if WHALE_CLASSIFIER is not None:
                    with gr.Row():
                        with gr.Column(scale=1):
                            id_audio = gr.File(
                                label="Fichier audio/video (WAV, MP3, MP4, OGG, FLAC...)",
                                file_types=[".wav", ".mp3", ".mp4", ".ogg", ".flac",
                                            ".m4a", ".webm", ".mkv", ".avi", ".wma", ".aac"],
                                type="filepath",
                            )
                            id_det_threshold = gr.Slider(
                                minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                                label="Seuil de detection TKEO",
                            )
                            id_snr_threshold = gr.Slider(
                                minimum=3, maximum=40, value=10, step=1,
                                label="Seuil SNR (dB)",
                            )
                            id_btn = gr.Button(
                                "Identifier",
                                variant="primary",
                            )

                            gr.Markdown("""
                            **Comment lire les resultats ?**
                            - **Confiance > 40%** : match probable
                            - **20-40%** : match possible, a confirmer
                            - **< 20%** : individu probablement inconnu
                            - La **synthese multi-codas** combine les
                              probabilites de tous les codas detectes
                            """)

                        with gr.Column(scale=2):
                            id_plot = gr.Plot(
                                label="Codas detectes — patron de clics",
                            )
                            id_results = gr.Markdown(
                                "Uploadez un fichier audio (WAV, MP3...) puis cliquez sur **Identifier**."
                            )

                    id_btn.click(
                        fn=identify_from_audio,
                        inputs=[id_audio, id_det_threshold, id_snr_threshold],
                        outputs=[id_plot, id_results],
                    )
                else:
                    gr.Markdown(
                        "Classifieur non disponible. "
                        "Le dataset Gero doit contenir au moins 50 codas etiquetes."
                    )

            with gr.Tab("Analyse vocale"):
                gr.Markdown("""
                ### Que fait ce cachalot ? — Classification d'activite vocale

                Uploadez un enregistrement et l'application classifie
                automatiquement l'activite du cachalot :

                - 📡 **Echolocation** (sonar) — clics reguliers espaces (~0.5-2s),
                  le cachalot scanne son environnement ou chasse en profondeur
                - 💬 **Codas** (communication) — rafales rythmiques de clics,
                  vocalisations sociales entre individus
                - 🎯 **Creaks/Buzz** (capture) — clics ultra-rapides (>30/s),
                  tentative de capture de proie imminente
                - 🔇 **Silence** — pas d'activite acoustique detectee

                *Tous les formats audio et video sont supportes (WAV, MP3, MP4, OGG, FLAC...).*
                """)

                with gr.Row():
                    with gr.Column(scale=1):
                        va_audio = gr.File(
                            label="Fichier audio/video (WAV, MP3, MP4...)",
                            file_types=[".wav", ".mp3", ".mp4", ".ogg", ".flac",
                                        ".m4a", ".webm", ".mkv", ".avi", ".wma", ".aac"],
                            type="filepath",
                        )
                        va_btn = gr.Button(
                            "Analyser l'activite",
                            variant="primary",
                        )
                        gr.Markdown("""
                        **Legendes des couleurs :**
                        - 🔵 Bleu = Echolocation
                        - 🟢 Vert = Codas
                        - 🔴 Rouge = Creaks (chasse)
                        - ⚫ Gris = Silence

                        **Astuce** : pour l'identification
                        individuelle, utilisez l'onglet
                        *Identifier un coda*.
                        """)

                    with gr.Column(scale=2):
                        va_plot = gr.Plot(
                            label="Timeline d'activite vocale",
                        )
                        va_results = gr.Markdown(
                            "Uploadez un fichier audio puis cliquez sur **Analyser**."
                        )

                va_btn.click(
                    fn=analyze_vocal_activity,
                    inputs=[va_audio],
                    outputs=[va_plot, va_results],
                )

            with gr.Tab("Zone d'etude"):
                gr.Markdown("""
                ### Ou vivent ces cachalots ?

                Tous les enregistrements de cette application proviennent de la **Dominique**,
                une petite ile volcanique des **Caraibes orientales** (15.4°N, 61.4°W).
                Cette region abrite l'une des populations de cachalots les mieux etudiees
                au monde, suivie depuis plus de 20 ans par le **Dominica Sperm Whale Project**
                dirige par Shane Gero et Hal Whitehead.

                Les cachalots de la Dominique vivent en **unites sociales matrilineaires**
                (des familles de femelles et de jeunes) qui partagent un repertoire
                de codas commun — un peu comme un dialecte regional.
                """)

                study_map = go.Figure()
                study_map.add_trace(go.Scattergeo(
                    lon=[-61.37],
                    lat=[15.41],
                    mode='markers+text',
                    marker=dict(size=16, color='#00d4aa', symbol='circle',
                                line=dict(width=2, color='white')),
                    text=["Dominique"],
                    textposition="top center",
                    textfont=dict(size=14, color='white'),
                    name="Zone d'etude",
                ))
                study_map.add_trace(go.Scattergeo(
                    lon=[-61.37, -61.20, -61.55, -61.30, -61.45],
                    lat=[15.41, 15.55, 15.30, 15.65, 15.20],
                    mode='markers',
                    marker=dict(size=8, color='#00d4aa', opacity=0.3, symbol='circle'),
                    name="Zones d'observation",
                    hoverinfo='skip',
                ))
                study_map.update_geos(
                    center=dict(lon=-61.37, lat=15.41),
                    projection_scale=80,
                    showland=True,
                    landcolor="#1a1a2e",
                    showocean=True,
                    oceancolor="#16213e",
                    showcoastlines=True,
                    coastlinecolor="#444",
                    showframe=False,
                    bgcolor="#0f0f23",
                )
                study_map.update_layout(
                    title=dict(
                        text="Zone d'etude — Dominique, Caraibes orientales",
                        font=dict(size=16, color="#e0e0e0"),
                    ),
                    template="plotly_dark",
                    paper_bgcolor="#0f0f23",
                    height=500,
                    margin=dict(l=0, r=0, t=50, b=0),
                    legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(size=11, color="#e0e0e0")),
                    geo=dict(
                        resolution=50,
                        showlakes=False,
                    ),
                )
                gr.Plot(value=study_map)

                gr.Markdown("""
                ---
                **Pourquoi la Dominique ?**

                Les eaux profondes au large de la cote ouest de la Dominique plongent
                rapidement a plus de 1000 metres — l'habitat ideal des cachalots qui
                chassent les calamars geants en profondeur. Cette proximite avec la cote
                permet aux chercheurs d'observer et d'enregistrer les cachalots
                presque quotidiennement.

                **Donnees GPS** : les datasets utilises ici ne contiennent pas de
                coordonnees GPS par enregistrement. Si des donnees georeferencees
                deviennent disponibles, la carte s'enrichira automatiquement avec
                les positions individuelles de chaque coda.
                """)

            with gr.Tab("Guide & Glossaire"):
                gr.Markdown("""
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
                pour rapprocher humains et nature.*
                """)

        gr.Markdown("""
        ---
        <p style="text-align: center; color: #666; font-size: 0.85em;">
            Whale Coda Explorer — Donnees: DSWP (Project CETI) · Modele: WhAM · Clustering: UMAP + HDBSCAN<br>
            Detecteur: portage Python du Coda-detector CETI (TKEO + graph clustering)<br>
            Projet de Claude & Kevin · <a href="https://github.com/CivicDash/whale-coda-explorer">GitHub</a>
        </p>
        """)

    return app


if __name__ == "__main__":
    app = build_app()
    wav_dirs = set(str(Path(f).parent) for f in FILENAMES if os.path.exists(f))

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        ssr_mode=False,
        allowed_paths=list(wav_dirs),
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.teal,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.gray,
        ),
        css="""
        .gradio-container { max-width: 1400px !important; }
        .main-header { text-align: center; padding: 20px 0 10px 0; }
        .main-header h1 { font-size: 2em; margin-bottom: 5px; }
        .main-header p { color: #888; font-size: 0.95em; }
        """,
        head=PLOTLY_CLICK_HEAD,
    )
    print("App lancee sur http://localhost:7860")
