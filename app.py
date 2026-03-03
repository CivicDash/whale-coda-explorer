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
from io import BytesIO
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

CODA_DF = pd.DataFrame({
    'umap_1': EMBEDDINGS_2D[:, 0],
    'umap_2': EMBEDDINGS_2D[:, 1],
    'cluster': [f"Cluster {c}" if c != -1 else "Bruit" for c in CLUSTER_LABELS],
    'idx': list(range(len(CLUSTER_LABELS))),
    'fichier': [Path(f).stem for f in FILENAMES],
})

CLUSTER_COLOR_MAP = {"Bruit": "#CCCCCC"}
for i in range(N_CLUSTERS):
    CLUSTER_COLOR_MAP[f"Cluster {i}"] = SPECTRAL_PALETTE[i % len(SPECTRAL_PALETTE)]


def get_filtered_df(selected_cluster="Tous"):
    """Return filtered DataFrame for the scatter plot."""
    if selected_cluster == "Tous":
        return CODA_DF.reset_index(drop=True)
    return CODA_DF[CODA_DF['cluster'] == selected_cluster].reset_index(drop=True)


def on_point_click(cluster_choice, evt: gr.SelectData):
    """Handle click on a scatter point — load audio + spectrogram."""
    try:
        idx = None
        current_df = get_filtered_df(cluster_choice)

        if hasattr(evt, 'index') and evt.index is not None:
            row_pos = evt.index
            if isinstance(row_pos, (list, tuple)):
                row_pos = row_pos[0]
            row_pos = int(row_pos)
            if 0 <= row_pos < len(current_df):
                idx = int(current_df.iloc[row_pos]['idx'])

        if idx is None and hasattr(evt, 'value'):
            val = evt.value
            if isinstance(val, dict) and 'idx' in val:
                idx = int(val['idx'])
            elif isinstance(val, (list, tuple)) and len(val) >= 2:
                x_val, y_val = float(val[0]), float(val[1])
                dists = (EMBEDDINGS_2D[:, 0] - x_val)**2 + (EMBEDDINGS_2D[:, 1] - y_val)**2
                idx = int(np.argmin(dists))

        if idx is None:
            return gr.update(), None, None, f"Selection non reconnue: {evt.value}"

        if idx < 0 or idx >= len(FILENAMES):
            return gr.update(), None, None, f"Index {idx} hors limites."

        filepath = FILENAMES[idx]
        if not os.path.exists(filepath):
            return gr.update(), None, None, f"Fichier introuvable: {filepath}"

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

        return gr.update(value=str(idx)), filepath, specfig, info

    except Exception as e:
        return gr.update(), None, None, f"Erreur lors du clic: {e}"


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
    centroid = FULL_EMBEDDINGS[mask].mean(axis=0)
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


def on_cluster_select(cluster_choice):
    """Callback quand l'utilisateur choisit un filtre cluster."""
    filtered_df = get_filtered_df(cluster_choice)

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

    return filtered_df, summary


def on_coda_select(cluster_choice, coda_index):
    """Charge un coda specifique par son index."""
    try:
        idx = int(coda_index)
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


GERO_DF = None
GERO_EMBEDDING = None
if GERO_PATH.exists():
    _gero_raw = pd.read_excel(GERO_PATH)
    _gero_raw['Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(_gero_raw['Date'], unit='D')
    _gero_raw['Year'] = _gero_raw['Date'].dt.year
    _unit_map = {1: "A", 2: "B", 3: "F", 4: "J", 5: "N", 6: "R", 7: "S", 8: "T", 9: "U"}
    _gero_raw['UnitName'] = _gero_raw['Unit'].map(_unit_map).fillna("?")
    GERO_DF = _gero_raw[_gero_raw['CodaName'] != 'NOISE'].reset_index(drop=True)

    _gero_emb_path = DATA_DIR / 'gero_embedding_2d.npy'
    if _gero_emb_path.exists():
        GERO_EMBEDDING = np.load(_gero_emb_path)


def build_gero_scatter_df(color_by="CodaName"):
    """Build DataFrame for Gero dataset scatter plot."""
    if GERO_DF is None or GERO_EMBEDDING is None:
        return pd.DataFrame()

    df = pd.DataFrame({
        'umap_1': GERO_EMBEDDING[:, 0],
        'umap_2': GERO_EMBEDDING[:, 1],
        'idx': range(len(GERO_DF)),
    })

    if color_by == "CodaName":
        df['group'] = GERO_DF['CodaName'].values
    elif color_by == "UnitName":
        df['group'] = ("Unit " + GERO_DF['UnitName']).values
    elif color_by == "WhaleID":
        df['group'] = GERO_DF['WhaleID'].apply(
            lambda x: f"Whale {x}" if x != 0 else "Non identifie"
        ).values
    elif color_by == "Year":
        df['group'] = GERO_DF['Year'].astype(str).values
    else:
        df['group'] = GERO_DF['CodaName'].values

    df['fichier'] = GERO_DF['CodaName'].values
    return df


def on_gero_color_change(color_by):
    """Update scatter plot when color-by changes."""
    color_map = {
        "Type de coda": "CodaName",
        "Unite sociale": "UnitName",
        "Individu": "WhaleID",
        "Annee": "Year",
    }
    col = color_map.get(color_by, "CodaName")
    df = build_gero_scatter_df(col)
    return df, get_gero_summary(col)


def get_gero_summary(color_by="CodaName"):
    """Summary stats for the Gero dataset."""
    if GERO_DF is None:
        return "Dataset non charge."

    md = f"### Dataset Gero et al. (2015)\n\n"
    md += f"- **Codas**: {len(GERO_DF)}\n"
    md += f"- **Types**: {GERO_DF['CodaName'].nunique()}\n"
    md += f"- **Unites sociales**: {GERO_DF['Unit'].nunique()}\n"
    md += f"- **Individus identifies**: {GERO_DF[GERO_DF['WhaleID'] != 0]['WhaleID'].nunique()}\n"
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
        identified = GERO_DF[GERO_DF['WhaleID'] != 0]
        for wid, count in identified['WhaleID'].value_counts().head(10).items():
            unit = identified[identified['WhaleID'] == wid]['UnitName'].iloc[0]
            md += f"- Whale {wid} (Unit {unit}): {count} codas\n"
        unid = (GERO_DF['WhaleID'] == 0).sum()
        md += f"- Non identifies: {unid}\n"
    elif color_by == "Year":
        md += "**Par annee:**\n"
        for year in sorted(GERO_DF['Year'].unique()):
            count = (GERO_DF['Year'] == year).sum()
            md += f"- {year}: {count} codas\n"

    return md


def on_gero_point_click(color_choice, evt: gr.SelectData):
    """Handle click on Gero scatter plot."""
    if GERO_DF is None:
        return "Dataset non charge."

    try:
        color_map = {
            "Type de coda": "CodaName",
            "Unite sociale": "UnitName",
            "Individu": "WhaleID",
            "Annee": "Year",
        }
        col = color_map.get(color_choice, "CodaName")
        current_df = build_gero_scatter_df(col)

        idx = None
        if hasattr(evt, 'index') and evt.index is not None:
            row_pos = evt.index
            if isinstance(row_pos, (list, tuple)):
                row_pos = row_pos[0]
            row_pos = int(row_pos)
            if 0 <= row_pos < len(current_df):
                idx = int(current_df.iloc[row_pos]['idx'])

        if idx is None:
            return "Selection non reconnue."

        row = GERO_DF.iloc[idx]
        ici_cols = ['ICI1', 'ICI2', 'ICI3', 'ICI4', 'ICI5', 'ICI6', 'ICI7', 'ICI8', 'ICI9']
        icis = [row[c] for c in ici_cols if row[c] > 0]
        icis_str = ", ".join(f"{ici*1000:.0f}ms" for ici in icis)

        whale_str = f"Whale {row['WhaleID']}" if row['WhaleID'] != 0 else "Non identifie"

        md = f"### Coda #{row['CodaNumber']}\n\n"
        md += f"- **Type**: {row['CodaName']}\n"
        md += f"- **Unite sociale**: Unit {row['UnitName']}\n"
        md += f"- **Individu**: {whale_str}\n"
        md += f"- **Clics**: {row['nClicks']}\n"
        md += f"- **Duree**: {row['Length']*1000:.0f} ms\n"
        md += f"- **ICIs**: [{icis_str}]\n"
        md += f"- **Date**: {row['Date'].strftime('%Y-%m-%d')}\n"

        return md
    except Exception as e:
        return f"Erreur: {e}"


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
                        scatter_plot = gr.ScatterPlot(
                            value=CODA_DF,
                            x="umap_1",
                            y="umap_2",
                            color="cluster",
                            color_map=CLUSTER_COLOR_MAP,
                            title="Carte des codas de cachalots — Espace WhAM",
                            x_title="UMAP 1",
                            y_title="UMAP 2",
                            tooltip="all",
                            height=650,
                            label="Carte des codas (cliquez sur un point)",
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

                with gr.Row():
                    with gr.Column(scale=1):
                        coda_index = gr.Textbox(
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

                scatter_plot.select(
                    fn=on_point_click,
                    inputs=[cluster_filter],
                    outputs=[coda_index, audio_player, spectrogram, coda_info],
                )

                cluster_filter.change(
                    fn=on_cluster_select,
                    inputs=[cluster_filter],
                    outputs=[scatter_plot, cluster_info],
                )

                load_btn.click(
                    fn=on_coda_select,
                    inputs=[cluster_filter, coda_index],
                    outputs=[audio_player, spectrogram, coda_info],
                )

                random_btn.click(
                    fn=get_random_coda,
                    inputs=[cluster_filter],
                    outputs=[coda_index, audio_player, spectrogram, coda_info],
                )

            with gr.Tab("Identite des cachalots"):
                gr.Markdown("""
                ### Qui parle ? — Identification par les codas
                Dataset [Gero, Whitehead & Rendell (2015)](https://doi.org/10.5061/dryad.ck4h0) :
                3876 codas des Caraibes orientales avec identification
                des individus, unites sociales et types de codas.
                Cliquez sur un point pour voir les details.
                """)

                if GERO_DF is not None and GERO_EMBEDDING is not None:
                    gero_color_choices = [
                        "Type de coda", "Unite sociale", "Individu", "Annee"
                    ]

                    with gr.Row():
                        with gr.Column(scale=3):
                            gero_scatter = gr.ScatterPlot(
                                value=build_gero_scatter_df("CodaName"),
                                x="umap_1",
                                y="umap_2",
                                color="group",
                                title="Codas de cachalots — Projection ICI (Gero et al.)",
                                x_title="UMAP 1",
                                y_title="UMAP 2",
                                tooltip="all",
                                height=600,
                                label="Cliquez sur un point",
                            )
                        with gr.Column(scale=1):
                            gero_color_by = gr.Dropdown(
                                choices=gero_color_choices,
                                value="Type de coda",
                                label="Colorer par",
                            )
                            gero_info = gr.Markdown(
                                value=get_gero_summary("CodaName"),
                            )
                            gero_detail = gr.Markdown(
                                "Cliquez sur un point pour voir ses details.",
                            )

                    gero_color_by.change(
                        fn=on_gero_color_change,
                        inputs=[gero_color_by],
                        outputs=[gero_scatter, gero_info],
                    )

                    gero_scatter.select(
                        fn=on_gero_point_click,
                        inputs=[gero_color_by],
                        outputs=[gero_detail],
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
    )
    print("App lancee sur http://localhost:7860")
