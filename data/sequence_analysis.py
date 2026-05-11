"""
Sequence analysis: DBSCAN clustering of protein sequences via ESM-2 embeddings.

Workflow
--------
1. Load per-residue ESM-2 embeddings from data/embeddings/ for each entry in
   pdb_list.txt, then mean-pool to one fixed-length vector per protein.
2. Reduce to N_PCA_COMPONENTS principal components.
3. Compute the full pairwise Euclidean distance matrix.
4. Select DBSCAN eps automatically via the Kneedle elbow on 4-NN distances,
   then cluster with min_samples=MIN_SAMPLES.
5. Produce a two-panel figure:
     • PCA scatter   – spatial layout coloured by cluster
     • Cluster sizes – member counts per group
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from collections import Counter
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from data_parsing import parse_pdb_list, parse_cif
# ── Configuration ─────────────────────────────────────────────────────────────
EMBEDDINGS_DIR   = Path("data/embeddings")
PDB_LIST         = Path("pdb_list.txt")
PDBS_DIR         = Path("data/pdbs")
N_PCA_COMPONENTS = 10      # captures ~66 % of embedding variance
MIN_SAMPLES      = 3       # DBSCAN core-point threshold
# eps chosen at the p10 of 4-NN distances — sits below the sharp rise in the
# k-NN curve for this dataset, yielding 4 tight clusters + noise outliers.
# (The Kneedle algorithm is also implemented below as a reference tool.)
DEFAULT_EPS      = 0.9
SCATTER_PATH     = Path("data/sequence_scatter.png")
SIZES_PATH       = Path("data/sequence_cluster_sizes.png")
SS_PATH          = Path("data/sequence_ss_composition.png")
NOISE_COLOR      = "#aaaaaa"


# ── Data loading ──────────────────────────────────────────────────────────────
def load_embeddings(
    pdb_list_path: Path,
    emb_dir: Path,
) -> tuple[list[str], np.ndarray]:
    """
    For each (pdb_id, chain) in pdb_list, load the .pt embedding file and
    mean-pool the per-residue vectors to one (d,) representation.

    Tries <pdb_id>_<chain>.pt first, falls back to <pdb_id>.pt.
    Entries with no embedding file are silently skipped.
    """
    pairs = parse_pdb_list(pdb_list_path)
    ids: list[str] = []
    vecs: list[np.ndarray] = []

    for pdb_id, chain in pairs:
        f = emb_dir / f"{pdb_id}_{chain}.pt"
        if not f.exists():
            f = emb_dir / f"{pdb_id}.pt"
        if not f.exists():
            continue
        item = torch.load(f, weights_only=False)
        emb = item["embedding"].float().numpy()  # (L, d)
        vecs.append(emb.mean(axis=0))            # mean-pool → (d,)
        ids.append(f"{pdb_id}_{chain}")

    return ids, np.array(vecs)


# ── Secondary-structure fractions ────────────────────────────────────────────
def load_ss_fractions(
    ids: list[str],
    pdbs_dir: Path,
) -> np.ndarray:
    """
    For each "<pdb_id>_<chain>" entry parse its CIF file and return the
    fraction of observed residues in each of the three SS states.

    Returns an (N, 3) float array with columns [H_frac, E_frac, C_frac].
    Rows where the CIF cannot be parsed are left as NaN.
    """
    fracs = np.full((len(ids), 3), np.nan)
    for i, entry in enumerate(ids):
        parts = entry.rsplit("_", 1)
        if len(parts) != 2:
            continue
        pdb_id, chain = parts
        cif_path = pdbs_dir / f"{pdb_id}.cif"
        if not cif_path.exists():
            continue
        try:
            chains = parse_cif(cif_path, chain)
        except Exception:
            continue
        if chain not in chains:
            continue
        _, ss3 = chains[chain]
        n = len(ss3)
        if n == 0:
            continue
        fracs[i, 0] = ss3.count("H") / n
        fracs[i, 1] = ss3.count("E") / n
        fracs[i, 2] = ss3.count("C") / n
    return fracs


# ── Eps selection ─────────────────────────────────────────────────────────────
def kneedle_eps(D: np.ndarray, k: int = 4) -> float:
    """
    Estimate a good DBSCAN eps from the sorted k-NN distance curve.

    Uses the Kneedle algorithm: the "knee" is the point on the sorted k-NN
    distance curve whose perpendicular distance to the line connecting the
    first and last points is maximised.
    """
    nn = NearestNeighbors(n_neighbors=k, metric="precomputed").fit(D)
    knn_dists = np.sort(nn.kneighbors(D)[0][:, k - 1])

    # Normalise curve to [0,1] × [0,1]
    n = len(knn_dists)
    x = np.linspace(0.0, 1.0, n)
    y = (knn_dists - knn_dists[0]) / (knn_dists[-1] - knn_dists[0] + 1e-12)

    # Direction vector of the baseline (from (0,0) to (1,1))
    d_vec = np.array([1.0, 1.0]) / np.sqrt(2)
    pts = np.column_stack([x, y])
    # Perpendicular distance of each point from the baseline
    proj_lengths = pts @ d_vec
    proj_pts = np.outer(proj_lengths, d_vec)
    perp_dists = np.linalg.norm(pts - proj_pts, axis=1)

    knee_idx = int(np.argmax(perp_dists))
    return float(knn_dists[knee_idx])


# ── Main analysis pipeline ────────────────────────────────────────────────────
def run_analysis(
    eps: float | None = None,
    min_samples: int = MIN_SAMPLES,
    n_pcs: int = N_PCA_COMPONENTS,
) -> tuple:
    """
    Load embeddings, reduce dimensions, compute distances, and cluster.

    Returns
    -------
    ids, Xr, D, labels, pca, eps
    """
    ids, X = load_embeddings(PDB_LIST, EMBEDDINGS_DIR)
    print(f"Loaded {len(ids)} sequence embeddings  (dim={X.shape[1]})")

    pca = PCA(n_components=n_pcs, random_state=42)
    Xr = pca.fit_transform(X)
    cum_var = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"PCA: {n_pcs} components capture {cum_var:.1%} of variance")

    D = pairwise_distances(Xr, metric="euclidean")

    if eps is None:
        eps = DEFAULT_EPS
    print(f"DBSCAN eps: {eps:.3f}")

    labels = DBSCAN(eps=eps, min_samples=min_samples,
                    metric="precomputed").fit_predict(D)
    cnt = Counter(labels)
    n_clusters = sum(1 for l in cnt if l != -1)
    print(f"DBSCAN: {n_clusters} clusters, {cnt.get(-1, 0)} noise points")
    for lbl in sorted(cnt):
        tag = f"  Cluster {lbl}" if lbl != -1 else "  Noise   "
        print(f"{tag}: {cnt[lbl]:3d} sequences")

    ss_fracs = load_ss_fractions(ids, PDBS_DIR)
    print(f"SS fractions loaded for {(~np.isnan(ss_fracs[:, 0])).sum()} sequences")

    return ids, Xr, D, labels, pca, eps, ss_fracs


# ── Visualisation ─────────────────────────────────────────────────────────────
def _cluster_style(labels: np.ndarray) -> tuple[list, dict, int, int]:
    """Shared colour map and counts used by all three plot functions."""
    unique_labels = sorted(set(labels))
    cluster_ids   = [l for l in unique_labels if l != -1]
    palette       = sns.color_palette("husl", max(len(cluster_ids), 1))
    color_map     = {l: palette[i] for i, l in enumerate(cluster_ids)}
    color_map[-1] = NOISE_COLOR
    n_noise       = int((labels == -1).sum())
    return cluster_ids, color_map, len(cluster_ids), n_noise


def plot_scatter(
    ids: list[str],
    Xr: np.ndarray,
    labels: np.ndarray,
    pca: PCA,
    eps: float,
    min_samples: int = MIN_SAMPLES,
    save_path: Path | None = SCATTER_PATH,
) -> plt.Figure:
    """PCA scatter coloured by cluster assignment."""
    sns.set_theme(style="whitegrid", font_scale=1.05)
    cluster_ids, color_map, n_clusters, n_noise = _cluster_style(labels)
    explained = pca.explained_variance_ratio_

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("white")
    ax.set_title(
        f"DBSCAN Clustering – ESM-2 Embeddings ({N_PCA_COMPONENTS}-PC PCA)\n"
        f"eps={eps:.3f}  min_samples={min_samples}  "
        f"{n_clusters} cluster{'s' if n_clusters != 1 else ''}  "
        f"{n_noise} noise",
        fontsize=11, pad=10,
    )

    noise_mask = labels == -1
    ax.scatter(
        Xr[noise_mask, 0], Xr[noise_mask, 1],
        c=NOISE_COLOR, s=35, alpha=0.45, edgecolors="none", zorder=1,
        label=f"Noise  (n={n_noise})",
    )
    for lbl in cluster_ids:
        mask  = labels == lbl
        color = color_map[lbl]
        ax.scatter(
            Xr[mask, 0], Xr[mask, 1],
            c=[color] * int(mask.sum()),
            s=80, alpha=0.85, edgecolors="white", linewidths=0.5,
            zorder=2, label=f"Cluster {lbl}  (n={mask.sum()})",
        )
        cx, cy = Xr[mask, 0].mean(), Xr[mask, 1].mean()
        ax.annotate(
            str(lbl), xy=(cx, cy), xycoords="data",
            fontsize=9, fontweight="bold", color="white", ha="center", va="center",
            bbox=dict(boxstyle="circle,pad=0.25", fc=color, ec="none", alpha=0.9),
            zorder=3,
        )

    ax.set_xlabel(f"PC1  ({explained[0]*100:.1f} % var)", fontsize=11)
    ax.set_ylabel(f"PC2  ({explained[1]*100:.1f} % var)", fontsize=11)
    ax.legend(loc="upper right", fontsize=8.5, framealpha=0.88,
              markerscale=1.1, handlelength=1.2, borderpad=0.6)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Figure saved: {save_path}")
    return fig


def plot_cluster_sizes(
    labels: np.ndarray,
    save_path: Path | None = SIZES_PATH,
) -> plt.Figure:
    """Horizontal bar chart of sequences per cluster / noise."""
    sns.set_theme(style="whitegrid", font_scale=1.05)
    cluster_ids, color_map, n_clusters, n_noise = _cluster_style(labels)

    bar_labels = [f"Cluster {l}" for l in cluster_ids] + ["Noise"]
    bar_sizes  = [(labels == l).sum() for l in cluster_ids] + [n_noise]
    bar_colors = [color_map[l] for l in cluster_ids] + [NOISE_COLOR]

    fig, ax = plt.subplots(figsize=(7, max(3, len(bar_labels) * 0.9)))
    fig.patch.set_facecolor("white")
    ax.set_title("Cluster Membership Counts", fontsize=12, pad=10)

    y_pos = np.arange(len(bar_labels))
    bars  = ax.barh(y_pos, bar_sizes, color=bar_colors,
                    edgecolor="white", linewidth=0.8, height=0.62)
    for bar, count in zip(bars, bar_sizes):
        ax.text(
            bar.get_width() + 0.4,
            bar.get_y() + bar.get_height() / 2,
            str(int(count)), va="center", ha="left", fontsize=10.5,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_labels, fontsize=10)
    ax.set_xlabel("Number of sequences", fontsize=11)
    ax.set_xlim(0, max(bar_sizes) * 1.22)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Figure saved: {save_path}")
    return fig


def plot_ss_composition(
    labels: np.ndarray,
    ss_fracs: np.ndarray,
    save_path: Path | None = SS_PATH,
) -> plt.Figure:
    """Grouped bar chart of mean H/E/C proportions per cluster (mean ± std)."""
    sns.set_theme(style="whitegrid", font_scale=1.05)
    cluster_ids, _, _, _ = _cluster_style(labels)

    valid = ~np.isnan(ss_fracs[:, 0])
    group_names: list[str]       = ["All"]
    group_data:  list[np.ndarray] = [ss_fracs[valid]]
    for lbl in cluster_ids:
        mask = (labels == lbl) & valid
        group_names.append(f"Cluster {lbl}")
        group_data.append(ss_fracs[mask])
    group_names.append("Noise")
    group_data.append(ss_fracs[(labels == -1) & valid])

    ss_labels = ["H – Helix", "E – Strand", "C – Coil"]
    ss_colors = ["#d95f5f", "#5b8fd9", "#5dba6e"]
    n_ss      = len(ss_labels)
    bw        = 0.22
    x         = np.arange(len(group_names))

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("white")
    ax.set_title(
        "True Secondary Structure Composition by Cluster  (mean ± std across sequences)",
        fontsize=11, pad=10,
    )

    for j, (ss_name, ss_color) in enumerate(zip(ss_labels, ss_colors)):
        means  = np.array([g[:, j].mean() if len(g) else 0.0 for g in group_data])
        stds   = np.array([g[:, j].std()  if len(g) else 0.0 for g in group_data])
        offset = (j - (n_ss - 1) / 2) * bw
        ax.bar(
            x + offset, means, bw,
            yerr=stds, capsize=3,
            color=ss_color, alpha=0.82,
            edgecolor="white", linewidth=0.6,
            label=ss_name,
            error_kw={"linewidth": 1.1, "ecolor": "#333333", "alpha": 0.6},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(group_names, fontsize=10.5)
    ax.set_ylabel("Proportion of residues", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9.5, framealpha=0.85, loc="upper right")
    ax.grid(axis="y", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
        print(f"Figure saved: {save_path}")
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ids, Xr, D, labels, pca, eps, ss_fracs = run_analysis()
    plot_scatter(ids, Xr, labels, pca, eps)
    plot_cluster_sizes(labels)
    plot_ss_composition(labels, ss_fracs)
    plt.show()
