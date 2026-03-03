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
import plotly.graph_objects as go
import gradio as gr
import scipy.io.wavfile as wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO

DATA_DIR = Path(__file__).parent / "exploration_output"
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
    fig = build_scatter_plot(cluster_choice)

    if cluster_choice == "Tous":
        summary = "### Vue d'ensemble\n\n"
        summary += f"- **Total codas analysees**: {len(CLUSTER_LABELS)}\n"
        summary += f"- **Clusters decouverts**: {N_CLUSTERS}\n"
        summary += f"- **Non classes**: {(CLUSTER_LABELS == -1).sum()}\n\n"
        summary += "Selectionnez un cluster pour voir ses details,\n"
        summary += "ou cliquez sur un point du graphique."
    elif cluster_choice == "Bruit":
        summary = get_cluster_summary(-1)
    else:
        cid = int(cluster_choice.replace("Cluster ", ""))
        summary = get_cluster_summary(cid)

    return fig, summary


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


def build_app():
    """Construit l'application Gradio."""

    cluster_choices = ["Tous", "Bruit"] + [f"Cluster {i}" for i in range(N_CLUSTERS)]

    with gr.Blocks(title="Whale Coda Explorer") as app:
        gr.HTML("""
        <div class="main-header">
            <h1>Whale Coda Explorer</h1>
            <p>Exploration interactive des vocalisations de cachalots via WhAM (Project CETI)</p>
            <p style="font-size: 0.8em; color: #666;">
                620 codas analysees · 15 clusters decouverts · Espace d'embeddings WhAM (1280 dim)
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                scatter_plot = gr.Plot(
                    value=build_scatter_plot(),
                    label="Carte des codas",
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
                        f"Selectionnez un cluster pour voir ses details."
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
                    placeholder="Entrez un index...",
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
                coda_info = gr.Markdown("Cliquez sur **Charger** pour analyser un coda.")

            with gr.Column(scale=2):
                audio_player = gr.Audio(
                    label="Ecouter le coda",
                    type="filepath",
                )
                spectrogram = gr.Plot(
                    label="Spectrogramme",
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

        gr.Markdown("""
        ---
        <p style="text-align: center; color: #666; font-size: 0.85em;">
            Whale Coda Explorer — Donnees: DSWP (Project CETI) · Modele: WhAM · Clustering: UMAP + HDBSCAN<br>
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
