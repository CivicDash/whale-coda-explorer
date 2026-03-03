"""
Analyse du dataset Gero et al. (2015) — Inter-click intervals
de 4116 codas de cachalots des Caraibes orientales.

Identifie les individus, unites sociales et types de codas,
puis projette les profils ICI dans un espace 2D pour visualiser
les structures sociales.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan

DATA_PATH = Path(__file__).parent / "data" / "gero2015_codas.xlsx"
OUTPUT_DIR = Path(__file__).parent / "exploration_output"


def load_gero_data():
    """Load and clean the Gero et al. dataset."""
    df = pd.read_excel(DATA_PATH)

    df['Date'] = pd.to_datetime('1899-12-30') + pd.to_timedelta(df['Date'], unit='D')
    df['Year'] = df['Date'].dt.year

    unit_names = {
        1: "Unit A", 2: "Unit B", 3: "Unit F",
        4: "Unit J", 5: "Unit N", 6: "Unit R",
        7: "Unit S", 8: "Unit T", 9: "Unit U",
    }
    df['UnitName'] = df['Unit'].map(unit_names).fillna(f"Unit {df['Unit']}")

    df = df[df['CodaName'] != 'NOISE'].copy()

    return df


def build_ici_vectors(df):
    """Build fixed-length ICI vectors for clustering."""
    ici_cols = ['ICI1', 'ICI2', 'ICI3', 'ICI4', 'ICI5',
                'ICI6', 'ICI7', 'ICI8', 'ICI9']
    ici_matrix = df[ici_cols].fillna(0).values

    features = np.column_stack([
        ici_matrix,
        df['nClicks'].values.reshape(-1, 1),
        df['Length'].values.reshape(-1, 1),
    ])

    return features


def project_and_cluster(features):
    """UMAP projection + HDBSCAN clustering on ICI profiles."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=20,
        min_dist=0.1,
        metric='euclidean',
        random_state=42,
    )
    embedding_2d = reducer.fit_transform(features_scaled)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15,
        min_samples=5,
    )
    auto_labels = clusterer.fit_predict(embedding_2d)

    return embedding_2d, auto_labels


def visualize(df, embedding_2d, output_dir):
    """Generate visualizations colored by different attributes."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 16))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle("Dataset Gero et al. (2015) — 3876 codas de cachalots des Caraibes",
                 color='#e0e0e0', fontsize=16, fontweight='bold')

    # --- By coda type ---
    ax = axes[0, 0]
    top_types = df['CodaName'].value_counts().head(8).index
    for ctype in top_types:
        mask = df['CodaName'] == ctype
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   s=8, alpha=0.6, label=ctype)
    other_mask = ~df['CodaName'].isin(top_types)
    ax.scatter(embedding_2d[other_mask, 0], embedding_2d[other_mask, 1],
               s=4, alpha=0.2, c='gray', label='Autres')
    ax.set_title("Par type de coda", color='#e0e0e0', fontsize=12)
    ax.legend(fontsize=7, markerscale=3, loc='best',
              facecolor='#16213e', labelcolor='#e0e0e0')
    _style_ax(ax)

    # --- By social unit ---
    ax = axes[0, 1]
    for unit in sorted(df['UnitName'].unique()):
        mask = df['UnitName'] == unit
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   s=8, alpha=0.6, label=unit)
    ax.set_title("Par unite sociale", color='#e0e0e0', fontsize=12)
    ax.legend(fontsize=7, markerscale=3, loc='best',
              facecolor='#16213e', labelcolor='#e0e0e0')
    _style_ax(ax)

    # --- By individual whale ---
    ax = axes[1, 0]
    identified = df[df['WhaleID'] != 0]
    unidentified = df[df['WhaleID'] == 0]
    ax.scatter(embedding_2d[df['WhaleID'] == 0, 0],
               embedding_2d[df['WhaleID'] == 0, 1],
               s=4, alpha=0.15, c='gray', label='Non identifie')
    for whale_id in identified['WhaleID'].unique():
        mask = df['WhaleID'] == whale_id
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   s=12, alpha=0.7, label=f"Whale {whale_id}")
    ax.set_title("Par individu (17 cachalots identifies)", color='#e0e0e0', fontsize=12)
    ax.legend(fontsize=6, markerscale=2, loc='best', ncol=2,
              facecolor='#16213e', labelcolor='#e0e0e0')
    _style_ax(ax)

    # --- By year ---
    ax = axes[1, 1]
    for year in sorted(df['Year'].unique()):
        mask = df['Year'] == year
        ax.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1],
                   s=8, alpha=0.5, label=str(year))
    ax.set_title("Par annee d'enregistrement", color='#e0e0e0', fontsize=12)
    ax.legend(fontsize=7, markerscale=3, loc='best',
              facecolor='#16213e', labelcolor='#e0e0e0')
    _style_ax(ax)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / 'gero2015_analysis.png', dpi=200, bbox_inches='tight')
    print(f"Visualisation sauvegardee: {output_dir / 'gero2015_analysis.png'}")
    plt.close(fig)


def _style_ax(ax):
    ax.set_facecolor('#16213e')
    ax.tick_params(colors='#999')
    ax.set_xlabel('UMAP 1', color='#e0e0e0', fontsize=9)
    ax.set_ylabel('UMAP 2', color='#e0e0e0', fontsize=9)
    for spine in ['bottom', 'left']:
        ax.spines[spine].set_color('#444')
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)


def main():
    print("=" * 60)
    print("ANALYSE DU DATASET GERO ET AL. (2015)")
    print("Identite individuelle dans les codas de cachalots")
    print("=" * 60)

    df = load_gero_data()
    print(f"\nCodas chargees: {len(df)} (hors bruit)")
    print(f"Types de codas: {df['CodaName'].nunique()}")
    print(f"Unites sociales: {df['Unit'].nunique()}")
    print(f"Individus identifies: {df[df['WhaleID'] != 0]['WhaleID'].nunique()}")
    print(f"Annees: {df['Year'].min()}-{df['Year'].max()}")

    print("\nConstruction des vecteurs ICI...")
    features = build_ici_vectors(df)
    print(f"Features: {features.shape}")

    print("\nProjection UMAP + clustering...")
    embedding_2d, auto_labels = project_and_cluster(features)

    n_clusters = len(set(auto_labels)) - (1 if -1 in auto_labels else 0)
    print(f"Clusters automatiques: {n_clusters}")

    print("\nGeneration des visualisations...")
    visualize(df, embedding_2d, OUTPUT_DIR)

    np.save(OUTPUT_DIR / 'gero_embedding_2d.npy', embedding_2d)
    np.save(OUTPUT_DIR / 'gero_auto_labels.npy', auto_labels)
    df.to_csv(OUTPUT_DIR / 'gero_codas_clean.csv', index=False)
    print(f"Donnees sauvegardees dans {OUTPUT_DIR}")

    print("\n" + "=" * 60)
    print("ANALYSE TERMINEE")
    print("=" * 60)


if __name__ == "__main__":
    main()
