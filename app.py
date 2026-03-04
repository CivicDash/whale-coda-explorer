"""
Whale Coda Explorer — Interactive web application
for exploring sperm whale vocalizations.

Interactive cluster visualization, audio playback,
per-cluster statistics, and spectrograms.

Project by Claude, with Kevin's blessing.
3 March 2026.
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
from translations import t, whale_name, whale_display, WHALE_NAMES_I18N

DATA_DIR = Path(__file__).parent / "exploration_output"
GERO_PATH = Path(__file__).parent / "data" / "gero2015_codas.xlsx"
EMBEDDINGS_2D = np.load(DATA_DIR / "embedding_2d.npy")
CLUSTER_LABELS = np.load(DATA_DIR / "cluster_labels.npy")
FULL_EMBEDDINGS = np.load(DATA_DIR / "embeddings.npy")

with open(DATA_DIR / "filenames.txt") as f:
    FILENAMES = [line.strip() for line in f if line.strip()]

N_CLUSTERS = len(set(CLUSTER_LABELS)) - (1 if -1 in CLUSTER_LABELS else 0)

CLUSTER_COLORS = {-1: "#CCCCCC"}
SPECTRAL_PALETTE = [
    "#9E0142", "#D53E4F", "#F46D43", "#FDAE61", "#FEE08B",
    "#E6F598", "#ABDDA4", "#66C2A5", "#3288BD", "#5E4FA2",
    "#8073AC", "#B2ABD2", "#D8DAEB", "#FDB863", "#E08214",
    "#B35806", "#542788", "#998EC3", "#F1A340", "#01665E",
]
for i in range(N_CLUSTERS):
    CLUSTER_COLORS[i] = SPECTRAL_PALETTE[i % len(SPECTRAL_PALETTE)]


# ── Dropdown choice builders ──

def _cluster_choices(lang="fr"):
    return ([(t("data.all", lang), "all"), (t("data.noise", lang), "noise")]
            + [(f"Cluster {i}", f"c{i}") for i in range(N_CLUSTERS)])

def _gero_color_choices(lang="fr"):
    return [
        (t("identity.coda_type", lang), "CodaName"),
        (t("identity.social_unit", lang), "UnitName"),
        (t("identity.individual", lang), "WhaleID"),
        (t("identity.year", lang), "Year"),
    ]

def _whale_choices(lang="fr"):
    choices = [(t("identity.all_individuals", lang), "all")]
    if GERO_DF is not None:
        for wid in sorted([w for w in GERO_DF['WhaleID'].unique() if w != '0'], key=str):
            choices.append((whale_display(wid, lang), wid))
    return choices


def _parse_click_value(val):
    """Extract index from 'idx_timestamp' or plain 'idx' string."""
    val = val.strip()
    if '_' in val:
        val = val.split('_')[0]
    return int(val)


def _cluster_label(cid, lang="fr"):
    if cid == -1:
        return t("data.noise", lang)
    return t("data.cluster", lang, cid=cid)


# ── Click & coda selection handlers ──

def on_plotly_click(clicked_idx_str, lang="fr"):
    try:
        if not clicked_idx_str or clicked_idx_str.strip() == "":
            return None, None, t("info.click_point", lang)

        idx = _parse_click_value(clicked_idx_str)
        if idx < 0 or idx >= len(FILENAMES):
            return None, None, t("info.out_of_bounds", lang, idx=idx)

        filepath = FILENAMES[idx]
        if not os.path.exists(filepath):
            return None, None, t("info.file_not_found", lang, path=filepath)

        cid = CLUSTER_LABELS[idx]
        specfig = make_spectrogram(filepath, lang)

        info = f"### Coda {Path(filepath).stem}\n\n"
        info += f"- **{t('info.cluster', lang)}**: {_cluster_label(cid, lang)}\n"
        info += f"- **{t('info.index', lang)}**: {idx}\n"
        info += f"- **{t('info.file', lang)}**: `{Path(filepath).name}`\n"
        info += f"- **{t('info.umap_pos', lang)}**: ({EMBEDDINGS_2D[idx, 0]:.3f}, {EMBEDDINGS_2D[idx, 1]:.3f})\n"

        neighbors = find_nearest_neighbors(idx, k=5)
        if neighbors:
            info += f"\n{t('info.nearest_codas', lang)}\n"
            for ni, dist in neighbors:
                ncid = CLUSTER_LABELS[ni]
                nlabel = t("data.noise", lang) if ncid == -1 else f"C{ncid}"
                info += f"- {Path(FILENAMES[ni]).stem} [{nlabel}] (dist: {dist:.4f})\n"

        return filepath, specfig, info

    except Exception as e:
        return None, None, t("data.error", lang, e=e)


def on_coda_select(cluster_choice, coda_index, lang="fr"):
    try:
        idx = _parse_click_value(coda_index)
        if idx < 0 or idx >= len(FILENAMES):
            return None, None, t("info.out_of_bounds_range", lang, idx=idx, max=len(FILENAMES)-1)

        filepath = FILENAMES[idx]
        if not os.path.exists(filepath):
            return None, None, t("info.file_not_found", lang, path=filepath)

        cid = CLUSTER_LABELS[idx]
        specfig = make_spectrogram(filepath, lang)

        info = f"### Coda {Path(filepath).stem}\n\n"
        info += f"- **{t('info.cluster', lang)}**: {_cluster_label(cid, lang)}\n"
        info += f"- **{t('info.index', lang)}**: {idx}\n"
        info += f"- **{t('info.file', lang)}**: `{Path(filepath).name}`\n"
        info += f"- **{t('info.umap_pos', lang)}**: ({EMBEDDINGS_2D[idx, 0]:.3f}, {EMBEDDINGS_2D[idx, 1]:.3f})\n"

        neighbors = find_nearest_neighbors(idx, k=5)
        if neighbors:
            info += f"\n{t('info.nearest_codas', lang)}\n"
            for ni, dist in neighbors:
                ncid = CLUSTER_LABELS[ni]
                nlabel = t("data.noise", lang) if ncid == -1 else f"C{ncid}"
                info += f"- {Path(FILENAMES[ni]).stem} [{nlabel}] (dist: {dist:.4f})\n"

        return filepath, specfig, info

    except (ValueError, IndexError) as e:
        return None, None, t("data.error", lang, e=e)


def find_nearest_neighbors(idx, k=5):
    target = FULL_EMBEDDINGS[idx]
    dists = np.linalg.norm(FULL_EMBEDDINGS - target, axis=1)
    dists[idx] = np.inf
    nearest = np.argsort(dists)[:k]
    return [(int(n), float(dists[n])) for n in nearest]


def get_random_coda(cluster_choice, lang="fr"):
    if cluster_choice == "all":
        idx = np.random.randint(len(FILENAMES))
    elif cluster_choice == "noise":
        candidates = np.where(CLUSTER_LABELS == -1)[0]
        if len(candidates) == 0:
            return gr.update(), None, None, t("summary.no_noise", lang)
        idx = np.random.choice(candidates)
    else:
        cid = int(cluster_choice.lstrip("c"))
        candidates = np.where(CLUSTER_LABELS == cid)[0]
        if len(candidates) == 0:
            return gr.update(), None, None, t("summary.empty_cluster", lang)
        idx = np.random.choice(candidates)

    audio, specfig, info = on_coda_select(cluster_choice, str(idx), lang)
    return gr.update(value=str(idx)), audio, specfig, info


# ── Plot builders ──

def build_scatter_plot(selected_cluster="all", lang="fr"):
    fig = go.Figure()

    if selected_cluster == "all":
        show_clusters = sorted(set(CLUSTER_LABELS))
    elif selected_cluster == "noise":
        show_clusters = [-1]
    else:
        cid = int(selected_cluster.lstrip("c"))
        show_clusters = [cid]

    file_label = t("plot.file_label", lang)
    for cid in sorted(set(CLUSTER_LABELS)):
        mask = CLUSTER_LABELS == cid
        label = _cluster_label(cid, lang)
        visible = cid in show_clusters

        indices = np.where(mask)[0]
        hover_texts = [
            f"<b>{label}</b><br>"
            f"{file_label} {Path(FILENAMES[i]).stem}<br>"
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
        title=dict(text=t("plot.main_title", lang), font=dict(size=18)),
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#444", borderwidth=1, font=dict(size=11)),
        height=650,
        margin=dict(l=60, r=30, t=60, b=60),
        hovermode='closest',
    )
    return fig


def make_spectrogram(filepath, lang="fr"):
    try:
        sr, data = wavfile.read(filepath)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float64)

        fig, axes = plt.subplots(2, 1, figsize=(10, 5), gridspec_kw={'height_ratios': [1, 2]})
        fig.patch.set_facecolor('#1a1a2e')

        tm = np.arange(len(data)) / sr

        axes[0].plot(tm, data, color='#66C2A5', linewidth=0.5, alpha=0.8)
        axes[0].set_xlim(0, tm[-1])
        axes[0].set_ylabel(t("spec.amplitude", lang), color='#e0e0e0', fontsize=9)
        axes[0].set_facecolor('#16213e')
        axes[0].tick_params(colors='#999')
        for spine in ['top', 'right']:
            axes[0].spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            axes[0].spines[spine].set_color('#444')

        axes[1].specgram(data, Fs=sr, cmap='inferno', NFFT=512, noverlap=384)
        axes[1].set_ylabel(t("spec.frequency", lang), color='#e0e0e0', fontsize=9)
        axes[1].set_xlabel(t("spec.time", lang), color='#e0e0e0', fontsize=9)
        axes[1].set_facecolor('#16213e')
        axes[1].tick_params(colors='#999')
        for spine in ['top', 'right']:
            axes[1].spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            axes[1].spines[spine].set_color('#444')
        axes[1].set_ylim(0, min(sr // 2, 10000))

        filename = Path(filepath).stem
        fig.suptitle(f"Coda {filename}", color='#e0e0e0', fontsize=12, fontweight='bold')
        fig.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.text(0.5, 0.5, t("data.error", lang, e=e), transform=ax.transAxes,
                ha='center', va='center', color='#ff6b6b', fontsize=12)
        return fig


def get_cluster_summary(cluster_id, lang="fr"):
    if cluster_id == -1:
        mask = CLUSTER_LABELS == -1
        label = t("summary.unclassified", lang)
    else:
        mask = CLUSTER_LABELS == cluster_id
        label = t("data.cluster", lang, cid=cluster_id)

    count = mask.sum()
    spread = FULL_EMBEDDINGS[mask].std()
    x_range = EMBEDDINGS_2D[mask, 0].max() - EMBEDDINGS_2D[mask, 0].min()
    y_range = EMBEDDINGS_2D[mask, 1].max() - EMBEDDINGS_2D[mask, 1].min()

    summary = f"### {label}\n\n"
    summary += f"- **{t('summary.n_codas', lang)}**: {count}\n"
    summary += f"- **{t('summary.proportion', lang)}**: {100*count/len(CLUSTER_LABELS):.1f}%\n"
    summary += f"- **{t('summary.spread', lang)}** (std embeddings): {spread:.4f}\n"
    summary += f"- **{t('summary.umap_extent', lang)}**: X={x_range:.2f}, Y={y_range:.2f}\n\n"

    if count <= 20:
        summary += f"{t('summary.files', lang)}\n"
        for idx in np.where(mask)[0]:
            summary += f"- {Path(FILENAMES[idx]).stem}\n"
    else:
        indices = np.where(mask)[0]
        summary += f"{t('summary.examples', lang, count=count)}\n"
        for idx in indices[:5]:
            summary += f"- {Path(FILENAMES[idx]).stem}\n"

    return summary


def build_overview_md(lang="fr"):
    return (
        f"{t('explorer.overview_title', lang)}\n\n"
        f"- **{t('explorer.total_codas', lang)}**: {len(CLUSTER_LABELS)}\n"
        f"- **{t('explorer.clusters_found', lang)}**: {N_CLUSTERS}\n"
        f"- **{t('explorer.unclassified', lang)}**: {(CLUSTER_LABELS == -1).sum()}\n\n"
        f"{t('explorer.click_hint', lang)}"
    )


def build_distribution_chart(lang="fr"):
    labels_sorted = sorted(set(CLUSTER_LABELS))
    names = [t("data.noise", lang) if l == -1 else f"C{l}" for l in labels_sorted]
    counts = [int((CLUSTER_LABELS == l).sum()) for l in labels_sorted]
    colors = [CLUSTER_COLORS.get(l, "#999") for l in labels_sorted]

    fig = go.Figure(go.Bar(
        x=names, y=counts, marker_color=colors,
        text=counts, textposition='outside',
        textfont=dict(color='#e0e0e0', size=10),
    ))
    fig.update_layout(
        title=t("dist.title", lang),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        height=350,
        margin=dict(l=50, r=30, t=50, b=50),
        xaxis_title=t("dist.x_label", lang),
        yaxis_title=t("dist.y_label", lang),
    )
    return fig


# ── Gero dataset ──

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

# k-NN classifier for individual identification
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


# ── Vocal activity classifier ──

def classify_vocal_activity(audio_path, lang="fr"):
    from pydub import AudioSegment
    from scipy.signal import butter, filtfilt, find_peaks
    import tempfile

    try:
        wav_path, duration_s, is_tmp = _convert_to_wav(audio_path)
    except Exception as e:
        return [], t("data.error", lang, e=e), None

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
                 'details': t("seg.no_acoustic", lang)}], "", data

    window_s = 3.0
    step_s = 1.0
    segments = []

    tt = 0.0
    while tt < duration_s:
        t_end = min(tt + window_s, duration_s)
        mask = (click_times >= tt) & (click_times < t_end)
        window_clicks = click_times[mask]

        if len(window_clicks) < 2:
            segments.append({
                'start': tt, 'end': t_end, 'type': 'SILENCE',
                'click_count': len(window_clicks), 'details': ''
            })
        else:
            icis = np.diff(window_clicks)
            mean_ici = np.mean(icis)
            std_ici = np.std(icis)
            click_rate = len(window_clicks) / (t_end - tt)
            cv = std_ici / mean_ici if mean_ici > 0 else 0

            if mean_ici < 0.030:
                activity = 'CREAK'
                details = t("seg.buzz_creak", lang, n=len(window_clicks),
                            rate=click_rate, ici=mean_ici*1000)
            elif mean_ici < 0.5 and cv < 1.0 and len(window_clicks) >= 3:
                activity = 'CODA'
                details = t("seg.codas", lang, n=len(window_clicks),
                            ici=mean_ici*1000)
            elif mean_ici >= 0.3 and cv < 0.5:
                activity = 'ECHOLOCATION'
                details = t("seg.echolocation", lang, n=len(window_clicks),
                            ici=mean_ici*1000, cv=cv)
            elif mean_ici >= 0.3:
                activity = 'ECHOLOCATION'
                details = t("seg.regular_clicks", lang, n=len(window_clicks),
                            ici=mean_ici*1000)
            else:
                activity = 'CODA'
                details = t("seg.vocal_activity", lang, n=len(window_clicks),
                            ici=mean_ici*1000)

            segments.append({
                'start': tt, 'end': t_end, 'type': activity,
                'click_count': len(window_clicks),
                'mean_ici': mean_ici, 'std_ici': std_ici,
                'click_rate': click_rate, 'details': details,
            })

        tt += step_s

    merged = _merge_segments(segments)
    return merged, "", data


def _merge_segments(segments):
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


def analyze_vocal_activity(audio_path, lang="fr"):
    if audio_path is None:
        return None, t("va.upload_file", lang)

    segments, error, data = classify_vocal_activity(audio_path, lang)
    if error:
        return None, error
    if not segments:
        return None, t("va.no_activity", lang)

    activity_counts = {}
    for seg in segments:
        dur = seg['end'] - seg['start']
        activity_counts[seg['type']] = activity_counts.get(seg['type'], 0) + dur

    total_dur = segments[-1]['end'] - segments[0]['start']

    colors = {
        'ECHOLOCATION': '#3498db', 'CODA': '#2ecc71',
        'CREAK': '#e74c3c', 'SILENCE': '#555555',
    }
    labels_i18n = {
        'ECHOLOCATION': t("va.echo_label", lang),
        'CODA': t("va.coda_label", lang),
        'CREAK': t("va.creak_label", lang),
        'SILENCE': t("va.silence_label", lang),
    }
    icons = {'ECHOLOCATION': '📡', 'CODA': '💬', 'CREAK': '🎯', 'SILENCE': '🔇'}

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), gridspec_kw={'height_ratios': [1, 3]})
    fig.patch.set_facecolor('#1a1a2e')

    ax_timeline = axes[0]
    ax_timeline.set_facecolor('#16213e')
    for seg in segments:
        ax_timeline.axvspan(seg['start'], seg['end'], alpha=0.7, color=colors.get(seg['type'], '#555'))
    ax_timeline.set_xlim(segments[0]['start'], segments[-1]['end'])
    ax_timeline.set_ylim(0, 1)
    ax_timeline.set_yticks([])
    ax_timeline.set_title(t("va.timeline_title", lang), color='#e0e0e0', fontsize=12, pad=8)
    ax_timeline.tick_params(colors='#999')

    import matplotlib.patches as mpatches
    handles = []
    for act_type in ['ECHOLOCATION', 'CODA', 'CREAK', 'SILENCE']:
        if act_type in activity_counts:
            handles.append(mpatches.Patch(color=colors[act_type], label=labels_i18n[act_type]))
    ax_timeline.legend(handles=handles, loc='upper right', fontsize=8,
                       facecolor='#1a1a2e', edgecolor='#444', labelcolor='#e0e0e0')

    ax_wave = axes[1]
    ax_wave.set_facecolor('#16213e')
    if data is not None:
        sr = 44100
        t_axis = np.arange(len(data)) / sr
        step = max(1, len(data) // 10000)
        ax_wave.plot(t_axis[::step], data[::step], color='#00d4aa', linewidth=0.3, alpha=0.7)
        for seg in segments:
            ax_wave.axvspan(seg['start'], seg['end'], alpha=0.15, color=colors.get(seg['type'], '#555'))

    ax_wave.set_xlabel(t("va.time", lang), color='#999', fontsize=10)
    ax_wave.set_ylabel(t("va.amplitude", lang), color='#999', fontsize=10)
    ax_wave.tick_params(colors='#999')
    ax_wave.set_xlim(segments[0]['start'], segments[-1]['end'])

    for ax in axes:
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#444')

    fig.tight_layout()

    md = f"{t('va.analysis_title', lang)}\n\n"
    md += f"{t('va.summary', lang)}\n\n"
    md += f"| {t('va.activity', lang)} | {t('va.duration', lang)} | {t('va.proportion', lang)} |\n"
    md += "|----------|-------|------------|\n"
    for act_type in ['CODA', 'ECHOLOCATION', 'CREAK', 'SILENCE']:
        if act_type in activity_counts:
            dur = activity_counts[act_type]
            pct = dur / total_dur * 100
            md += f"| {icons[act_type]} {labels_i18n[act_type]} | {dur:.1f}s | {pct:.0f}% |\n"

    md += f"\n*{t('va.total_duration', lang, s=total_dur)}*\n\n"

    md += f"{t('va.detail_by_segment', lang)}\n\n"
    for i, seg in enumerate(segments, 1):
        icon = icons.get(seg['type'], '')
        label = labels_i18n.get(seg['type'], seg['type'])
        dur = seg['end'] - seg['start']
        t_min = int(seg['start'] // 60)
        t_sec = seg['start'] % 60
        md += f"**{i}. {icon} {label}** — {t_min}m{t_sec:04.1f}s → +{dur:.1f}s"
        if seg.get('click_count', 0) > 0:
            md += f" | {seg['click_count']} {t('va.clicks_count', lang)}"
        if 'mean_ici' in seg:
            md += f" | ICI={seg['mean_ici']*1000:.0f}ms"
        md += "\n\n"

    if 'ECHOLOCATION' in activity_counts:
        md += f"---\n{t('va.echo_analysis_title', lang)}\n\n"
        md += f"{t('va.echo_description', lang)}\n\n"

        echo_segs = [s for s in segments if s['type'] == 'ECHOLOCATION']
        if echo_segs:
            avg_ici = np.mean([s.get('mean_ici', 0) for s in echo_segs if 'mean_ici' in s])
            if avg_ici > 0:
                depth_est = avg_ici * 750
                md += (f"- **{t('va.avg_ici', lang)}** : {avg_ici*1000:.0f}ms\n"
                       f"- **{t('va.target_depth', lang)}** : ~{depth_est:.0f}m "
                       f"(ICI \u00d7 v_sound / 2)\n")
                if avg_ici > 1.0:
                    md += f"- **{t('va.interpretation', lang)}** : {t('va.deep_hunt', lang)}\n"
                elif avg_ici > 0.5:
                    md += f"- **{t('va.interpretation', lang)}** : {t('va.approaching', lang)}\n"
                else:
                    md += f"- **{t('va.interpretation', lang)}** : {t('va.close_target', lang)}\n"
                md += "\n"

    if 'CREAK' in activity_counts:
        md += f"---\n{t('va.creaks_title', lang)}\n\n"
        md += f"{t('va.creaks_description', lang)}\n\n"

    if 'CODA' in activity_counts and WHALE_CLASSIFIER is not None:
        md += f"---\n{t('va.codas_detected', lang)}\n\n"
        md += f"{t('va.codas_id_hint', lang)}\n"

    return fig, md


def _convert_to_wav(audio_path):
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


def identify_from_audio(audio_path, det_threshold=0.3, snr_threshold=10, lang="fr"):
    if audio_path is None:
        return None, t("idr.upload_file", lang)

    try:
        wav_path, duration_s, is_tmp = _convert_to_wav(audio_path)
    except Exception as e:
        return None, t("idr.conversion_error", lang, e=e)

    params = DetectorParams(detection_threshold=det_threshold, snr_threshold=snr_threshold)

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
        return None, t("idr.no_coda", lang)

    ext = os.path.splitext(audio_path)[1].lower()
    format_info = f"{t('idr.format', lang)}: {ext.upper().strip('.')} | " if ext != '.wav' else ""
    md = t("idr.n_codas", lang, n=len(all_codas)) + "\n"
    md += f"*{format_info}{t('idr.duration', lang)}: {duration_s:.1f}s"
    if duration_s > SEGMENT_MAX_S:
        md += f" ({t('idr.segments', lang, n=int(np.ceil(duration_s / SEGMENT_MAX_S)))})"
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
        md += f"- **{t('idr.clicks', lang)}**: {n_clicks} | **{t('idr.duration', lang)}**: {duration*1000:.0f}ms\n"
        md += f"- **ICIs**: [{icis_str}]\n\n"

        candidates = identify_coda_from_icis(icis, n_clicks, duration)

        if not candidates:
            md += t("idr.classifier_unavailable", lang) + "\n\n"
            continue

        md += f"| {t('idr.rank', lang)} | {t('idr.individual', lang)} | {t('idr.confidence', lang)} |\n"
        md += "|------|----------|----------|\n"

        for rank, (wid, prob) in enumerate(candidates[:5], 1):
            name_str = whale_display(wid, lang)
            bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
            md += f"| {rank} | {name_str} | {bar} {prob*100:.1f}% |\n"

        best_wid, best_prob = candidates[0]
        best_name, best_desc = whale_name(best_wid, lang)
        if best_prob > 0.4:
            md += f"\n{t('idr.best_match', lang, name=best_name, pct=best_prob*100)}"
            if best_desc:
                md += f" — *{best_desc}*"
            md += "\n\n"
        elif best_prob > 0.2:
            md += f"\n{t('idr.possible_match', lang, name=best_name, pct=best_prob*100)}\n\n"
        else:
            md += f"\n{t('idr.low_confidence', lang)}\n\n"

    if len(all_codas) > 1:
        md += f"---\n{t('idr.multi_synthesis', lang)}\n\n"
        vote_counts = {}
        for r in all_results:
            candidates = identify_coda_from_icis(r['icis'], r['n_clicks'], r['duration'])
            for wid, prob in candidates:
                vote_counts[wid] = vote_counts.get(wid, 0) + prob

        if vote_counts:
            total = sum(vote_counts.values())
            sorted_votes = sorted(vote_counts.items(), key=lambda x: -x[1])
            md += f"{t('idr.combining_all', lang)}\n\n"
            md += f"| {t('idr.individual', lang)} | {t('idr.cumulative_score', lang)} |\n"
            md += "|----------|------------|\n"
            for wid, score in sorted_votes[:7]:
                name_str = whale_display(wid, lang)
                pct = score / total * 100
                md += f"| {name_str} | {pct:.1f}% |\n"

            top = sorted_votes[:3]
            if len(top) >= 2 and top[1][1] / total > 0.15:
                md += f"\n{t('idr.multi_individual', lang)}\n"

            best_wid = sorted_votes[0][0]
            best_name, _ = whale_name(best_wid, lang)
            md += t("idr.main_identification", lang, name=best_name) + "\n"

    return _build_identification_plot(all_codas, all_results, lang), md


def _build_identification_plot(codas, results, lang="fr"):
    fig, axes = plt.subplots(len(codas), 1, figsize=(10, 3 * len(codas)), squeeze=False)
    fig.patch.set_facecolor('#1a1a2e')

    for i, (coda, r) in enumerate(zip(codas, results)):
        ax = axes[i, 0]
        ax.set_facecolor('#16213e')
        icis = r['icis']
        clicks_t = [0]
        for ici in icis:
            clicks_t.append(clicks_t[-1] + ici)

        ax.stem(clicks_t, [1] * len(clicks_t), linefmt='#00d4aa', markerfmt='o', basefmt='none')
        ax.set_title(
            t("idplot.coda_title", lang, i=i+1, n=r['n_clicks'], ms=f"{r['duration']*1000:.0f}"),
            color='#e0e0e0', fontsize=11,
        )
        ax.set_xlabel(t("idplot.time", lang), color='#999', fontsize=9)
        ax.set_ylabel(t("idplot.click", lang), color='#999', fontsize=9)
        ax.set_ylim(0, 1.5)
        ax.tick_params(colors='#999')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_color('#444')

    fig.tight_layout()
    return fig


# ── Gero plots & summaries ──

def build_gero_plotly(color_by="CodaName", lang="fr"):
    if GERO_DF is None or GERO_EMBEDDING is None:
        fig = go.Figure()
        fig.update_layout(title=t("identity.dataset_not_loaded", lang))
        return fig

    title_map = {
        "CodaName": t("gero.title_coda_type", lang),
        "UnitName": t("gero.title_social_unit", lang),
        "WhaleID": t("gero.title_individual", lang),
        "Year": t("gero.title_year", lang),
    }

    fig = go.Figure()

    if color_by == "WhaleID":
        groups = GERO_DF['WhaleID'].apply(
            lambda x: whale_display(x, lang) if str(x) != '0' else t("identity.not_identified", lang)
        ).values
    elif color_by == "UnitName":
        groups = ("Unit " + GERO_DF['UnitName']).values
    elif color_by == "Year":
        groups = GERO_DF['Year'].astype(str).values
    else:
        groups = GERO_DF[color_by].values

    unique_groups = sorted(set(groups))
    palette = SPECTRAL_PALETTE

    for gi, grp in enumerate(unique_groups):
        mask = groups == grp
        indices = np.where(mask)[0]

        hover_texts = []
        for i_idx in indices:
            row = GERO_DF.iloc[i_idx]
            whale_str = whale_display(row['WhaleID'], lang) if str(row['WhaleID']) != '0' else t("identity.not_id_short", lang)
            hover_texts.append(
                f"<b>{grp}</b><br>"
                f"Type: {row['CodaName']}<br>"
                f"Unit: {row['UnitName']}<br>"
                f"{t('gero.indiv_label', lang)}: {whale_str}<br>"
                f"{t('gero.clicks_label', lang)}: {row['nClicks']}<br>"
                f"Index: {i_idx}"
            )

        fig.add_trace(go.Scatter(
            x=GERO_EMBEDDING[mask, 0], y=GERO_EMBEDDING[mask, 1],
            mode='markers',
            marker=dict(size=6, color=palette[gi % len(palette)], opacity=0.7,
                        line=dict(width=0.3, color='white')),
            name=str(grp), text=hover_texts, hoverinfo='text',
            customdata=indices.tolist(),
        ))

    fig.update_layout(
        title=dict(
            text=t("gero.plot_title", lang, subtitle=title_map.get(color_by, '')),
            font=dict(size=16),
        ),
        xaxis_title="UMAP 1", yaxis_title="UMAP 2",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", bordercolor="#444", borderwidth=1, font=dict(size=10)),
        height=600, margin=dict(l=50, r=30, t=50, b=50),
        hovermode='closest',
    )
    return fig


def on_gero_color_change(color_by, lang="fr"):
    fig = build_gero_plotly(color_by, lang)
    return fig, get_gero_summary(color_by, lang)


def get_gero_summary(color_by="CodaName", lang="fr"):
    if GERO_DF is None:
        return t("identity.dataset_not_loaded", lang)

    md = f"{t('gero.dataset_title', lang)}\n\n"
    md += f"- **{t('data.codas', lang)}**: {len(GERO_DF)}\n"
    md += f"- **{t('gero.types', lang)}**: {GERO_DF['CodaName'].nunique()}\n"
    md += f"- **{t('gero.social_units', lang)}**: {GERO_DF['Unit'].nunique()}\n"
    md += f"- **{t('gero.identified', lang)}**: {GERO_DF[GERO_DF['WhaleID'] != '0']['WhaleID'].nunique()}\n"
    md += f"- **{t('gero.period', lang)}**: {GERO_DF['Year'].min()}-{GERO_DF['Year'].max()}\n\n"

    if color_by == "CodaName":
        md += f"{t('gero.most_frequent', lang)}\n"
        for name, count in GERO_DF['CodaName'].value_counts().head(8).items():
            pct = 100 * count / len(GERO_DF)
            md += f"- {name}: {count} ({pct:.1f}%)\n"
    elif color_by == "UnitName":
        md += f"{t('gero.social_units_label', lang)}\n"
        for unit in sorted(GERO_DF['UnitName'].unique()):
            count = (GERO_DF['UnitName'] == unit).sum()
            md += f"- Unit {unit}: {count} codas\n"
    elif color_by == "WhaleID":
        md += f"{t('gero.identified_label', lang)}\n"
        identified = GERO_DF[GERO_DF['WhaleID'] != '0']
        for wid, count in identified['WhaleID'].value_counts().head(10).items():
            unit = identified[identified['WhaleID'] == wid]['UnitName'].iloc[0]
            display = whale_display(wid, lang)
            md += f"- {display} (Unit {unit}): {count} codas\n"
        unid = (GERO_DF['WhaleID'] == '0').sum()
        md += f"- {t('gero.unidentified', lang)}: {unid}\n"
    elif color_by == "Year":
        md += f"{t('gero.by_year', lang)}\n"
        for year in sorted(GERO_DF['Year'].unique()):
            count = (GERO_DF['Year'] == year).sum()
            md += f"- {year}: {count} codas\n"

    return md


def on_gero_point_click(clicked_idx_str, lang="fr"):
    if GERO_DF is None:
        return t("identity.dataset_not_loaded", lang)

    try:
        if not clicked_idx_str or clicked_idx_str.strip() == "":
            return t("identity.click_detail", lang)

        idx = _parse_click_value(clicked_idx_str)
        if idx < 0 or idx >= len(GERO_DF):
            return t("info.out_of_bounds", lang, idx=idx)

        row = GERO_DF.iloc[idx]
        ici_cols = ['ICI1', 'ICI2', 'ICI3', 'ICI4', 'ICI5', 'ICI6', 'ICI7', 'ICI8', 'ICI9']
        icis = [row[c] for c in ici_cols if row[c] > 0]
        icis_str = ", ".join(f"{ici*1000:.0f}ms" for ici in icis)

        wid = str(row['WhaleID'])
        wd = whale_display(wid, lang)
        whale_desc = ""
        if wid in WHALE_NAMES_I18N:
            _, desc = whale_name(wid, lang)
            if desc:
                whale_desc = f"\n  *{desc}*"

        md = f"### Coda #{row['CodaNumber']}\n\n"
        md += f"- **{t('gero.type_label', lang)}**: {row['CodaName']}\n"
        md += f"- **{t('gero.unit_label', lang)}**: Unit {row['UnitName']}\n"
        md += f"- **{t('gero.indiv_label', lang)}**: {wd}{whale_desc}\n"
        md += f"- **{t('gero.clicks_label', lang)}**: {row['nClicks']}\n"
        md += f"- **{t('gero.duration_label', lang)}**: {row['Length']*1000:.0f} ms\n"
        md += f"- **{t('gero.icis_label', lang)}**: [{icis_str}]\n"
        md += f"- **{t('gero.date_label', lang)}**: {row['Date'].strftime('%Y-%m-%d')}\n"
        return md
    except Exception as e:
        return t("data.error", lang, e=e)


def build_whale_profile(whale_choice, lang="fr"):
    if GERO_DF is None or GERO_EMBEDDING is None:
        return "", None

    if not whale_choice or whale_choice == "all":
        fig = build_gero_plotly("WhaleID", lang)
        return t("profile.select", lang), fig

    wid = whale_choice
    if wid not in GERO_DF['WhaleID'].values:
        return t("profile.not_found", lang, wid=wid), ""

    sub = GERO_DF[GERO_DF['WhaleID'] == wid]
    name, desc = whale_name(wid, lang)
    unit = sub['UnitName'].iloc[0]

    md = f"## {name}\n"
    md += f"*{desc}*\n\n"
    md += f"{t('profile.scientific_id', lang)}: #{wid}\n\n"
    md += f"{t('profile.social_unit', lang)}: Unit {unit}\n\n"

    years = sorted(sub['Year'].unique())
    year_range = f"{min(years)} - {max(years)}" if len(years) > 1 else str(years[0])
    yr_word = t("profile.years", lang) if len(years) > 1 else t("profile.year", lang)
    md += f"{t('profile.observation_period', lang)}: {year_range} ({len(years)} {yr_word})\n\n"
    md += f"{t('profile.n_codas', lang)}: {len(sub)}\n\n"
    md += f"{t('profile.clicks_avg', lang)}: {sub['nClicks'].mean():.1f}\n\n"
    md += f"{t('profile.avg_duration', lang)}: {sub['Length'].mean()*1000:.0f} ms\n\n"

    md += f"{t('profile.vocal_repertoire', lang)}\n\n"
    md += f"| {t('profile.coda_type', lang)} | {t('profile.count', lang)} | {t('profile.proportion', lang)} |\n"
    md += "|:------------|-------:|-----------:|\n"
    for ctype, count in sub['CodaName'].value_counts().items():
        pct = 100 * count / len(sub)
        bar = "█" * int(pct / 5)
        md += f"| {ctype} | {count} | {bar} {pct:.1f}% |\n"

    md += f"\n{t('profile.activity_by_year', lang)}\n\n"
    for year in sorted(sub['Year'].unique()):
        yr_sub = sub[sub['Year'] == year]
        md += f"- **{year}**: {len(yr_sub)} codas"
        top = yr_sub['CodaName'].value_counts().head(2)
        types_str = ", ".join(f"{tp}({c})" for tp, c in top.items())
        md += f" — {types_str}\n"

    family = GERO_DF[(GERO_DF['UnitName'] == unit) & (GERO_DF['WhaleID'] != '0') & (GERO_DF['WhaleID'] != wid)]
    if len(family) > 0:
        relatives = family['WhaleID'].unique()
        md += f"\n{t('profile.family', lang, unit=unit)}\n\n"
        for rel_wid in sorted(relatives, key=str):
            rel_name = whale_display(rel_wid, lang)
            rel_count = (family['WhaleID'] == rel_wid).sum()
            md += f"- {rel_name}: {rel_count} codas\n"

    fig = go.Figure()
    others_mask = GERO_DF['WhaleID'] != wid
    fig.add_trace(go.Scatter(
        x=GERO_EMBEDDING[others_mask, 0], y=GERO_EMBEDDING[others_mask, 1],
        mode='markers', marker=dict(size=4, color='#555', opacity=0.2),
        name=t("data.others", lang), hoverinfo='skip',
    ))

    whale_mask = GERO_DF['WhaleID'] == wid
    whale_indices = np.where(whale_mask)[0]
    hover_texts = []
    for i in whale_indices:
        row = GERO_DF.iloc[i]
        hover_texts.append(
            f"<b>{name}</b><br>"
            f"Type: {row['CodaName']}<br>"
            f"{t('gero.clicks_label', lang)}: {row['nClicks']}<br>"
            f"{t('gero.date_label', lang)}: {row['Date'].strftime('%Y-%m-%d')}"
        )

    unit_palette = {
        "A": "#D53E4F", "B": "#F46D43", "F": "#66C2A5",
        "J": "#3288BD", "N": "#FDAE61", "R": "#9E0142",
        "S": "#5E4FA2", "T": "#FEE08B", "U": "#E6F598",
    }
    whale_color = unit_palette.get(unit, "#00FFAA")

    fig.add_trace(go.Scatter(
        x=GERO_EMBEDDING[whale_mask, 0], y=GERO_EMBEDDING[whale_mask, 1],
        mode='markers',
        marker=dict(size=10, color=whale_color, opacity=0.9,
                    line=dict(width=1, color='white'), symbol='circle'),
        name=name, text=hover_texts, hoverinfo='text',
        customdata=whale_indices.tolist(),
    ))

    fig.update_layout(
        title=dict(text=t("profile.plot_title", lang, name=name, n=len(sub)), font=dict(size=16)),
        xaxis_title="UMAP 1", yaxis_title="UMAP 2",
        template="plotly_dark",
        paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e",
        font=dict(color="#e0e0e0"),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(size=11)),
        height=550, margin=dict(l=50, r=30, t=50, b=50),
        hovermode='closest',
    )

    return md, fig


# ── Detector ──

def run_detector(audio_file, det_threshold, snr_threshold, lang="fr"):
    if audio_file is None:
        return None, t("detector.upload_wav", lang)

    params = DetectorParams(detection_threshold=det_threshold, snr_threshold=snr_threshold)
    codas = detect_codas(audio_file, params=params)
    results = codas_to_dict(codas)
    fig = make_detection_plot(audio_file, codas, lang)

    if not results:
        return fig, t("detector.no_coda", lang)

    md = t("detector.n_codas", lang, n=len(results)) + "\n\n"
    md += t("detector.table_header", lang) + "\n"
    md += "|---|-------|-----------|------------|-----------|---------------|----------|\n"
    for r in results:
        icis_str = ", ".join(f"{ici*1000:.0f}" for ici in r['icis'])
        md += (f"| {r['coda_id']} | {r['n_clicks']} | {r['start_time']:.2f} | "
               f"{r['duration']*1000:.0f} | {icis_str} | "
               f"{r['mean_ipi']:.1f} | {r['mean_snr']:.0f} dB |\n")

    return fig, md


def make_detection_plot(audio_path, codas, lang="fr"):
    try:
        sr, data = wavfile.read(audio_path)
        if data.ndim > 1:
            data = data[:, 0]
        data = data.astype(np.float64)

        fig, axes = plt.subplots(3, 1, figsize=(12, 7), gridspec_kw={'height_ratios': [2, 1, 2]})
        fig.patch.set_facecolor('#1a1a2e')

        tm = np.arange(len(data)) / sr

        axes[0].plot(tm, data, color='#66C2A5', linewidth=0.3, alpha=0.7)
        axes[0].set_ylabel(t("detplot.amplitude", lang), color='#e0e0e0', fontsize=9)
        axes[0].set_title(t("detplot.title", lang), color='#e0e0e0', fontsize=12, fontweight='bold')

        det_colors = ['#D53E4F', '#F46D43', '#FDAE61', '#66C2A5', '#3288BD',
                      '#9E0142', '#5E4FA2', '#ABDDA4', '#FEE08B', '#E6F598']
        for i, coda in enumerate(codas):
            color = det_colors[i % len(det_colors)]
            for click in coda.clicks:
                axes[0].axvline(click.time, color=color, alpha=0.7, linewidth=1.5, linestyle='-')
            if coda.clicks:
                t_start = min(c.time for c in coda.clicks)
                t_end = max(c.time for c in coda.clicks)
                axes[0].axvspan(t_start - 0.01, t_end + 0.01, alpha=0.15, color=color)
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
        axes[2].set_ylabel(t("detplot.freq", lang), color='#e0e0e0', fontsize=9)
        axes[2].set_xlabel(t("detplot.time", lang), color='#e0e0e0', fontsize=9)
        axes[2].set_ylim(0, min(sr // 2, 12000))

        for i, coda in enumerate(codas):
            color = det_colors[i % len(det_colors)]
            for click in coda.clicks:
                axes[2].axvline(click.time, color=color, alpha=0.5, linewidth=1, linestyle='--')

        for ax in axes:
            ax.set_facecolor('#16213e')
            ax.tick_params(colors='#999')
            for spine in ['bottom', 'left']:
                ax.spines[spine].set_color('#444')
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            ax.set_xlim(0, tm[-1])

        fig.tight_layout()
        return fig
    except Exception as e:
        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#16213e')
        ax.text(0.5, 0.5, t("data.error", lang, e=e), transform=ax.transAxes,
                ha='center', va='center', color='#ff6b6b', fontsize=12)
        return fig


def run_detector_on_dataset(coda_index, det_threshold, snr_threshold, lang="fr"):
    try:
        idx = int(coda_index)
        if idx < 0 or idx >= len(FILENAMES):
            return None, t("detector.out_of_bounds", lang, max=len(FILENAMES)-1)
        filepath = FILENAMES[idx]
        if not os.path.exists(filepath):
            return None, t("detector.file_not_found", lang, path=filepath)
        return run_detector(filepath, det_threshold, snr_threshold, lang)
    except ValueError:
        return None, t("detector.invalid_index", lang)


# ── Study area map builder ──

def build_study_map(lang="fr"):
    study_map = go.Figure()
    study_map.add_trace(go.Scattergeo(
        lon=[-61.37], lat=[15.41],
        mode='markers+text',
        marker=dict(size=16, color='#00d4aa', symbol='circle', line=dict(width=2, color='white')),
        text=[t("study.dominica", lang)],
        textposition="top center",
        textfont=dict(size=14, color='white'),
        name=t("study.study_zone", lang),
    ))
    study_map.add_trace(go.Scattergeo(
        lon=[-61.37, -61.20, -61.55, -61.30, -61.45],
        lat=[15.41, 15.55, 15.30, 15.65, 15.20],
        mode='markers',
        marker=dict(size=8, color='#00d4aa', opacity=0.3, symbol='circle'),
        name=t("study.observation_zones", lang),
        hoverinfo='skip',
    ))
    study_map.update_geos(
        center=dict(lon=-61.37, lat=15.41), projection_scale=80,
        showland=True, landcolor="#1a1a2e",
        showocean=True, oceancolor="#16213e",
        showcoastlines=True, coastlinecolor="#444",
        showframe=False, bgcolor="#0f0f23",
    )
    study_map.update_layout(
        title=dict(text=t("study.map_title", lang), font=dict(size=16, color="#e0e0e0")),
        template="plotly_dark", paper_bgcolor="#0f0f23",
        height=500, margin=dict(l=0, r=0, t=50, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0.5)", font=dict(size=11, color="#e0e0e0")),
        geo=dict(resolution=50, showlakes=False),
    )
    return study_map


# ── Plotly click bridge JS ──

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
        for (var plotId in PLOT_MAP) {
            attachToPlot(plotId, PLOT_MAP[plotId]);
        }
    }

    var observer = new MutationObserver(function() { attachAll(); });
    observer.observe(document.body, {childList: true, subtree: true});
    setInterval(attachAll, 2000);
})();
</script>
"""


# ── Build the Gradio app ──

def build_app():
    initial_lang = "fr"

    with gr.Blocks(title="Whale Coda Explorer") as app:

        lang_state = gr.State(value=initial_lang)

        # ── Header + language selector ──
        with gr.Row():
            with gr.Column(scale=5):
                gr.HTML("""
                <div class="main-header">
                    <h1>Whale Coda Explorer</h1>
                </div>
                """)
            with gr.Column(scale=1, min_width=150):
                lang_radio = gr.Radio(
                    choices=[("Français", "fr"), ("English", "en")],
                    value="fr",
                    label="🌐",
                    container=False,
                )

        header_subtitle = gr.Markdown(
            f"{t('header.subtitle', initial_lang)}\n\n"
            f"<p style='font-size:0.8em;color:#666;'>{t('header.stats', initial_lang)}</p>"
        )

        with gr.Tabs():

            # ─── Tab 1: Explorer ───
            with gr.Tab(t("tab.explorer", initial_lang)):
                with gr.Row():
                    with gr.Column(scale=3):
                        scatter_plot = gr.Plot(
                            value=build_scatter_plot("all", initial_lang),
                            elem_id="main-scatter",
                        )
                    with gr.Column(scale=1):
                        cluster_filter = gr.Dropdown(
                            choices=_cluster_choices(initial_lang),
                            value="all",
                            label=t("explorer.filter_label", initial_lang),
                        )
                        cluster_info = gr.Markdown(value=build_overview_md(initial_lang))
                        distribution_chart = gr.Plot(
                            value=build_distribution_chart(initial_lang),
                            label=t("explorer.distribution", initial_lang),
                        )

                gr.Markdown("---")
                listen_title_md = gr.Markdown(t("explorer.listen_title", initial_lang))

                click_bridge = gr.Textbox(visible=False, elem_id="click_bridge")

                with gr.Row():
                    with gr.Column(scale=1):
                        coda_index_input = gr.Textbox(
                            label=t("explorer.coda_index_label", initial_lang),
                            placeholder=t("explorer.coda_index_placeholder", initial_lang),
                            value="0",
                        )
                        random_btn = gr.Button(t("explorer.random_btn", initial_lang), variant="secondary", size="sm")
                        load_btn = gr.Button(t("explorer.load_btn", initial_lang), variant="primary")
                        coda_info = gr.Markdown(t("explorer.click_placeholder", initial_lang))

                    with gr.Column(scale=2):
                        audio_player = gr.Audio(
                            label=t("explorer.listen_label", initial_lang),
                            type="filepath",
                        )
                        spectrogram = gr.Plot(label=t("explorer.spectrogram_label", initial_lang))

                click_bridge.change(
                    fn=on_plotly_click,
                    inputs=[click_bridge, lang_state],
                    outputs=[audio_player, spectrogram, coda_info],
                )

                def on_cluster_filter_change(cluster_choice, lang):
                    fig = build_scatter_plot(cluster_choice, lang)
                    if cluster_choice == "all":
                        summary = build_overview_md(lang)
                    elif cluster_choice == "noise":
                        summary = get_cluster_summary(-1, lang)
                    else:
                        cid = int(cluster_choice.lstrip("c"))
                        summary = get_cluster_summary(cid, lang)
                    return fig, summary

                cluster_filter.change(
                    fn=on_cluster_filter_change,
                    inputs=[cluster_filter, lang_state],
                    outputs=[scatter_plot, cluster_info],
                )

                load_btn.click(
                    fn=on_coda_select,
                    inputs=[cluster_filter, coda_index_input, lang_state],
                    outputs=[audio_player, spectrogram, coda_info],
                )

                random_btn.click(
                    fn=get_random_coda,
                    inputs=[cluster_filter, lang_state],
                    outputs=[coda_index_input, audio_player, spectrogram, coda_info],
                )

            # ─── Tab 2: Identity ───
            with gr.Tab(t("tab.identity", initial_lang)):
                identity_md = gr.Markdown(t("identity.title", initial_lang))

                if GERO_DF is not None and GERO_EMBEDDING is not None:
                    with gr.Row():
                        with gr.Column(scale=3):
                            gero_scatter_plot = gr.Plot(
                                value=build_gero_plotly("CodaName", initial_lang),
                                elem_id="gero-scatter",
                            )
                        with gr.Column(scale=1):
                            gero_color_by = gr.Dropdown(
                                choices=_gero_color_choices(initial_lang),
                                value="CodaName",
                                label=t("identity.color_by", initial_lang),
                            )
                            whale_selector = gr.Dropdown(
                                choices=_whale_choices(initial_lang),
                                value="all",
                                label=t("identity.search_individual", initial_lang),
                            )
                            gero_info = gr.Markdown(value=get_gero_summary("CodaName", initial_lang))
                            gero_click_bridge = gr.Textbox(elem_id="gero_click_bridge", visible=False)
                            gero_detail = gr.Markdown(t("identity.click_detail", initial_lang))

                    with gr.Row(visible=True):
                        whale_profile_md = gr.Markdown(visible=False)

                    def on_gero_color_change_and_reset(color_by, lang):
                        fig, summary = on_gero_color_change(color_by, lang)
                        return (
                            fig, summary,
                            gr.update(value="all"),
                            gr.update(visible=False, value=""),
                        )

                    gero_color_by.change(
                        fn=on_gero_color_change_and_reset,
                        inputs=[gero_color_by, lang_state],
                        outputs=[gero_scatter_plot, gero_info, whale_selector, whale_profile_md],
                    )

                    gero_click_bridge.change(
                        fn=on_gero_point_click,
                        inputs=[gero_click_bridge, lang_state],
                        outputs=[gero_detail],
                    )

                    def on_whale_select(whale_choice, lang):
                        md, fig = build_whale_profile(whale_choice, lang)
                        show_profile = whale_choice != "all"
                        return (
                            fig,
                            gr.update(value=md, visible=show_profile),
                            get_gero_summary("CodaName", lang) if not show_profile else get_gero_summary("WhaleID", lang),
                        )

                    whale_selector.change(
                        fn=on_whale_select,
                        inputs=[whale_selector, lang_state],
                        outputs=[gero_scatter_plot, whale_profile_md, gero_info],
                    )
                else:
                    gr.Markdown(t("identity.launch_script", initial_lang))

            # ─── Tab 3: Detector ───
            with gr.Tab(t("tab.detector", initial_lang)):
                detector_md = gr.Markdown(t("detector.title", initial_lang))

                with gr.Row():
                    with gr.Column(scale=1):
                        det_opt1_md = gr.Markdown(t("detector.option1", initial_lang))
                        audio_upload = gr.Audio(
                            label=t("detector.wav_label", initial_lang),
                            type="filepath",
                        )
                        det_opt2_md = gr.Markdown(t("detector.option2", initial_lang))
                        det_coda_index = gr.Textbox(
                            label=t("detector.coda_index", initial_lang),
                            placeholder="Ex: 42",
                        )
                        det_params_md = gr.Markdown(t("detector.parameters", initial_lang))
                        det_threshold = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                            label=t("detector.tkeo_threshold", initial_lang),
                        )
                        snr_threshold_slider = gr.Slider(
                            minimum=3, maximum=40, value=10, step=1,
                            label=t("detector.snr_threshold", initial_lang),
                        )
                        detect_upload_btn = gr.Button(
                            t("detector.detect_upload", initial_lang), variant="primary",
                        )
                        detect_dataset_btn = gr.Button(
                            t("detector.detect_dataset", initial_lang), variant="secondary",
                        )

                    with gr.Column(scale=2):
                        detection_plot = gr.Plot(label=t("detector.plot_label", initial_lang))
                        detection_results = gr.Markdown(t("detector.placeholder", initial_lang))

                detect_upload_btn.click(
                    fn=run_detector,
                    inputs=[audio_upload, det_threshold, snr_threshold_slider, lang_state],
                    outputs=[detection_plot, detection_results],
                )
                detect_dataset_btn.click(
                    fn=run_detector_on_dataset,
                    inputs=[det_coda_index, det_threshold, snr_threshold_slider, lang_state],
                    outputs=[detection_plot, detection_results],
                )

            # ─── Tab 4: Identify ───
            with gr.Tab(t("tab.identify", initial_lang)):
                identify_title_md = gr.Markdown(t("identify.title", initial_lang))

                if WHALE_CLASSIFIER is not None:
                    with gr.Row():
                        with gr.Column(scale=1):
                            id_audio = gr.File(
                                label=t("identify.file_label", initial_lang),
                                file_types=[".wav", ".mp3", ".mp4", ".ogg", ".flac",
                                            ".m4a", ".webm", ".mkv", ".avi", ".wma", ".aac"],
                                type="filepath",
                            )
                            id_det_threshold = gr.Slider(
                                minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                                label=t("detector.tkeo_threshold", initial_lang),
                            )
                            id_snr_threshold = gr.Slider(
                                minimum=3, maximum=40, value=10, step=1,
                                label=t("detector.snr_threshold", initial_lang),
                            )
                            id_btn = gr.Button(t("identify.btn", initial_lang), variant="primary")
                            identify_how_md = gr.Markdown(t("identify.how_to_read", initial_lang))

                        with gr.Column(scale=2):
                            id_plot = gr.Plot(label=t("identify.plot_label", initial_lang))
                            id_results = gr.Markdown(t("identify.placeholder", initial_lang))

                    id_btn.click(
                        fn=identify_from_audio,
                        inputs=[id_audio, id_det_threshold, id_snr_threshold, lang_state],
                        outputs=[id_plot, id_results],
                    )
                else:
                    gr.Markdown(t("identify.classifier_unavailable", initial_lang))

            # ─── Tab 5: Vocal analysis ───
            with gr.Tab(t("tab.vocal", initial_lang)):
                vocal_md = gr.Markdown(t("vocal.title", initial_lang))

                with gr.Row():
                    with gr.Column(scale=1):
                        va_audio = gr.File(
                            label=t("vocal.file_label", initial_lang),
                            file_types=[".wav", ".mp3", ".mp4", ".ogg", ".flac",
                                        ".m4a", ".webm", ".mkv", ".avi", ".wma", ".aac"],
                            type="filepath",
                        )
                        va_btn = gr.Button(t("vocal.btn", initial_lang), variant="primary")
                        vocal_legend_md = gr.Markdown(t("vocal.legend", initial_lang))

                    with gr.Column(scale=2):
                        va_plot = gr.Plot(label=t("vocal.plot_label", initial_lang))
                        va_results = gr.Markdown(t("vocal.placeholder", initial_lang))

                va_btn.click(
                    fn=analyze_vocal_activity,
                    inputs=[va_audio, lang_state],
                    outputs=[va_plot, va_results],
                )

            # ─── Tab 6: Study area ───
            with gr.Tab(t("tab.study", initial_lang)):
                study_md = gr.Markdown(f"{t('study.title', initial_lang)}\n\n{t('study.description', initial_lang)}")
                study_map_plot = gr.Plot(value=build_study_map(initial_lang))
                study_why_md = gr.Markdown(f"---\n{t('study.why_dominica', initial_lang)}")

            # ─── Tab 7: Guide & Glossary ───
            with gr.Tab(t("tab.guide", initial_lang)):
                guide_md = gr.Markdown(t("guide.full", initial_lang))

        footer_md = gr.Markdown(f"---\n{t('footer', initial_lang)}")

        # ── Language switch handler ──
        def on_lang_change(lang_choice):
            lang = lang_choice

            # Collect all the components that need updating
            updates = [
                lang,  # lang_state
                # header
                f"{t('header.subtitle', lang)}\n\n<p style='font-size:0.8em;color:#666;'>{t('header.stats', lang)}</p>",
                # explorer
                gr.update(choices=_cluster_choices(lang), label=t("explorer.filter_label", lang)),
                build_overview_md(lang),
                build_distribution_chart(lang),
                build_scatter_plot("all", lang),
                t("explorer.listen_title", lang),
                gr.update(label=t("explorer.coda_index_label", lang),
                          placeholder=t("explorer.coda_index_placeholder", lang)),
                gr.update(value=t("explorer.random_btn", lang)),
                gr.update(value=t("explorer.load_btn", lang)),
                t("explorer.click_placeholder", lang),
                gr.update(label=t("explorer.listen_label", lang)),
                gr.update(label=t("explorer.spectrogram_label", lang)),
                # identity
                t("identity.title", lang),
            ]

            if GERO_DF is not None and GERO_EMBEDDING is not None:
                updates += [
                    build_gero_plotly("CodaName", lang),
                    gr.update(choices=_gero_color_choices(lang), label=t("identity.color_by", lang)),
                    gr.update(choices=_whale_choices(lang), label=t("identity.search_individual", lang), value="all"),
                    get_gero_summary("CodaName", lang),
                    t("identity.click_detail", lang),
                ]
            else:
                updates += [None, gr.update(), gr.update(), "", ""]

            updates += [
                # detector
                t("detector.title", lang),
                t("detector.option1", lang),
                t("detector.option2", lang),
                t("detector.parameters", lang),
                gr.update(label=t("detector.wav_label", lang)),
                gr.update(label=t("detector.coda_index", lang)),
                gr.update(label=t("detector.tkeo_threshold", lang)),
                gr.update(label=t("detector.snr_threshold", lang)),
                gr.update(value=t("detector.detect_upload", lang)),
                gr.update(value=t("detector.detect_dataset", lang)),
                gr.update(label=t("detector.plot_label", lang)),
                t("detector.placeholder", lang),
                # identify
                t("identify.title", lang),
            ]

            if WHALE_CLASSIFIER is not None:
                updates += [
                    gr.update(label=t("identify.file_label", lang)),
                    gr.update(label=t("detector.tkeo_threshold", lang)),
                    gr.update(label=t("detector.snr_threshold", lang)),
                    gr.update(value=t("identify.btn", lang)),
                    t("identify.how_to_read", lang),
                    gr.update(label=t("identify.plot_label", lang)),
                    t("identify.placeholder", lang),
                ]
            else:
                updates += [gr.update(), gr.update(), gr.update(), gr.update(), "", gr.update(), ""]

            updates += [
                # vocal
                t("vocal.title", lang),
                gr.update(label=t("vocal.file_label", lang)),
                gr.update(value=t("vocal.btn", lang)),
                t("vocal.legend", lang),
                gr.update(label=t("vocal.plot_label", lang)),
                t("vocal.placeholder", lang),
                # study
                f"{t('study.title', lang)}\n\n{t('study.description', lang)}",
                build_study_map(lang),
                f"---\n{t('study.why_dominica', lang)}",
                # guide
                t("guide.full", lang),
                # footer
                f"---\n{t('footer', lang)}",
            ]

            return updates

        lang_outputs = [
            lang_state,
            header_subtitle,
            # explorer
            cluster_filter, cluster_info, distribution_chart, scatter_plot,
            listen_title_md, coda_index_input, random_btn, load_btn,
            coda_info, audio_player, spectrogram,
            # identity
            identity_md,
        ]

        if GERO_DF is not None and GERO_EMBEDDING is not None:
            lang_outputs += [gero_scatter_plot, gero_color_by, whale_selector, gero_info, gero_detail]
        else:
            lang_outputs += [gr.Markdown(visible=False)] * 5

        lang_outputs += [
            # detector
            detector_md, det_opt1_md, det_opt2_md, det_params_md,
            audio_upload, det_coda_index, det_threshold, snr_threshold_slider,
            detect_upload_btn, detect_dataset_btn, detection_plot, detection_results,
            # identify
            identify_title_md,
        ]

        if WHALE_CLASSIFIER is not None:
            lang_outputs += [id_audio, id_det_threshold, id_snr_threshold, id_btn,
                             identify_how_md, id_plot, id_results]
        else:
            lang_outputs += [gr.Markdown(visible=False)] * 7

        lang_outputs += [
            # vocal
            vocal_md, va_audio, va_btn, vocal_legend_md, va_plot, va_results,
            # study
            study_md, study_map_plot, study_why_md,
            # guide
            guide_md,
            # footer
            footer_md,
        ]

        lang_radio.change(
            fn=on_lang_change,
            inputs=[lang_radio],
            outputs=lang_outputs,
        )

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
