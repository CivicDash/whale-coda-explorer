"""
Coda Explorer — Extraction d'embeddings et analyse non supervisee
des codas de cachalots du dataset DSWP via le modele WhAM.

Projet personnel de Claude, avec la benediction de Kevin.
3 mars 2026.
"""

import os
import sys
import glob
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

WHAM_DIR = Path(__file__).parent
VAMPNET_DIR = WHAM_DIR / "vampnet"
MODELS_DIR = VAMPNET_DIR / "models"
DSWP_DIR = WHAM_DIR / "data" / "dswp"
OUTPUT_DIR = WHAM_DIR / "exploration_output"

sys.path.insert(0, str(WHAM_DIR))
os.environ["VAMPNET_DIR"] = str(VAMPNET_DIR)


def load_model(device="cuda"):
    """Charge le modele WhAM sur le device specifie."""
    from vampnet.interface import Interface
    print(f"Chargement du modele WhAM sur {device}...")
    interface = Interface(
        codec_ckpt=MODELS_DIR / "codec.pth",
        coarse_ckpt=MODELS_DIR / "coarse.pth",
        coarse2fine_ckpt=MODELS_DIR / "c2f.pth",
        device=device,
    )
    print("Modele charge.")
    return interface


def find_coda_files():
    """Trouve tous les fichiers WAV reels (pas des pointeurs LFS) dans le dataset DSWP."""
    patterns = [str(DSWP_DIR / "*.wav")]
    all_files = []
    for p in patterns:
        all_files.extend(glob.glob(p, recursive=True))

    real_files = []
    for f in all_files:
        size = os.path.getsize(f)
        if size > 10000:
            real_files.append(f)

    files = sorted(set(real_files))
    print(f"Trouve {len(files)} fichiers de codas reels (sur {len(all_files)} total).")
    return files


def extract_embeddings(interface, coda_files, device="cuda", layer=10):
    """
    Extrait les embeddings du transformer WhAM (couche `layer`)
    pour chaque coda. Retourne un array (N, embedding_dim).
    """
    from audiotools import AudioSignal

    embeddings = []
    valid_files = []
    errors = []

    for filepath in tqdm(coda_files, desc="Extraction des embeddings"):
        try:
            sig = AudioSignal(filepath)
            sig = interface.preprocess(sig).to(device)

            with torch.inference_mode():
                z = interface.encode(sig)
                n_codebooks = interface.coarse.n_codebooks
                z = z[:, :n_codebooks, :]
                z_latents = interface.coarse.embedding.from_codes(z, interface.codec)
                _, activations = interface.coarse(z_latents, return_activations=True)

                emb = activations[layer]
                emb = emb.mean(dim=1).squeeze(0).cpu().numpy()

            embeddings.append(emb)
            valid_files.append(filepath)

        except Exception as e:
            errors.append((filepath, str(e)))

    if errors:
        print(f"\n{len(errors)} erreurs lors de l'extraction:")
        for f, e in errors[:5]:
            print(f"  {Path(f).name}: {e}")

    embeddings = np.stack(embeddings)
    print(f"\nEmbeddings extraits: {embeddings.shape} ({len(valid_files)} codas, {embeddings.shape[1]} dimensions)")
    return embeddings, valid_files


def cluster_and_visualize(embeddings, filenames, output_dir):
    """
    Applique UMAP + HDBSCAN pour trouver des clusters dans les embeddings
    et genere des visualisations.
    """
    import umap
    import hdbscan

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nReduction dimensionnelle avec UMAP...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
    )
    embedding_2d = reducer.fit_transform(embeddings)

    print("Clustering avec HDBSCAN...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=10,
        min_samples=5,
        metric='euclidean',
    )
    labels = clusterer.fit_predict(embedding_2d)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"\nResultats du clustering:")
    print(f"  Clusters trouves: {n_clusters}")
    print(f"  Points classes comme bruit: {n_noise}")
    print(f"  Points assignes a un cluster: {len(labels) - n_noise}")

    for cluster_id in range(n_clusters):
        count = (labels == cluster_id).sum()
        print(f"  Cluster {cluster_id}: {count} codas")

    # --- Visualisation principale ---
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    scatter = ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=labels,
        cmap='Spectral',
        s=20,
        alpha=0.7,
        edgecolors='none',
    )

    noise_mask = labels == -1
    if noise_mask.any():
        ax.scatter(
            embedding_2d[noise_mask, 0],
            embedding_2d[noise_mask, 1],
            c='lightgray',
            s=10,
            alpha=0.3,
            label='Non classe',
        )

    ax.set_title(
        f'Carte des codas de cachalots — Espace WhAM\n'
        f'{n_clusters} clusters decouverts dans {len(labels)} codas',
        fontsize=14, fontweight='bold'
    )
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster ID')

    plt.tight_layout()
    fig.savefig(output_dir / 'coda_clusters_map.png', dpi=200, bbox_inches='tight')
    print(f"\nCarte sauvegardee: {output_dir / 'coda_clusters_map.png'}")

    # --- Distribution des clusters ---
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    unique_labels = sorted(set(labels))
    counts = [np.sum(labels == l) for l in unique_labels]
    colors = ['lightgray' if l == -1 else plt.cm.Spectral(l / max(1, n_clusters)) for l in unique_labels]
    bar_labels = ['Bruit' if l == -1 else f'Cluster {l}' for l in unique_labels]

    ax2.bar(bar_labels, counts, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_title('Distribution des codas par cluster', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Nombre de codas')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig2.savefig(output_dir / 'cluster_distribution.png', dpi=200, bbox_inches='tight')
    print(f"Distribution sauvegardee: {output_dir / 'cluster_distribution.png'}")

    # --- Sauvegarder les resultats ---
    np.save(output_dir / 'embeddings.npy', embeddings)
    np.save(output_dir / 'embedding_2d.npy', embedding_2d)
    np.save(output_dir / 'cluster_labels.npy', labels)

    with open(output_dir / 'filenames.txt', 'w') as f:
        for fn in filenames:
            f.write(fn + '\n')

    with open(output_dir / 'analysis_report.txt', 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("ANALYSE DES CODAS DE CACHALOTS — PROJET CETI/WhAM\n")
        f.write(f"Date: 3 mars 2026\n")
        f.write(f"Auteur: Claude (avec Kevin)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Nombre de codas analysees: {len(filenames)}\n")
        f.write(f"Dimensions des embeddings: {embeddings.shape[1]}\n")
        f.write(f"Couche du transformer utilisee: 10/16\n")
        f.write(f"Methode de clustering: UMAP + HDBSCAN\n\n")
        f.write(f"Clusters trouves: {n_clusters}\n")
        f.write(f"Points bruit: {n_noise}\n\n")
        for cluster_id in sorted(set(labels)):
            count = (labels == cluster_id).sum()
            label = "Bruit (non classe)" if cluster_id == -1 else f"Cluster {cluster_id}"
            f.write(f"  {label}: {count} codas\n")
        f.write("\n")
        f.write("Ces clusters pourraient correspondre a:\n")
        f.write("  - Differentes unites sociales de cachalots\n")
        f.write("  - Differents types de codas (rythme, tempo)\n")
        f.write("  - Differents contextes comportementaux\n")
        f.write("  - Differentes conditions d'enregistrement\n")
        f.write("\nUne investigation plus poussee avec des annotations\n")
        f.write("est necessaire pour valider ces hypotheses.\n")

    print(f"Rapport sauvegarde: {output_dir / 'analysis_report.txt'}")

    return labels, embedding_2d


def main():
    print("=" * 60)
    print("EXPLORATION DES CODAS DE CACHALOTS")
    print("Projet CETI — WhAM Embedding Analysis")
    print("=" * 60)
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    interface = load_model(device)
    coda_files = find_coda_files()

    if not coda_files:
        print("Aucun fichier de coda trouve ! Verifiez le chemin du dataset DSWP.")
        print(f"Chemin attendu: {DSWP_DIR}")
        sys.exit(1)

    embeddings, valid_files = extract_embeddings(interface, coda_files, device)
    labels, embedding_2d = cluster_and_visualize(embeddings, valid_files, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("EXPLORATION TERMINEE")
    print(f"Resultats dans: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
