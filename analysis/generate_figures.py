#!/usr/bin/env python3
"""Generate the 10 publication figures + the basis-misalignment cartoon
for the AAAI 2027 submission of FedOmniAdapt.

Reads /tmp/fedomni_results/batch9/*.json (133 cells), groups by
(task, split, family, strategy, K, M, fedprox_mu, cmld_alpha), and
produces PDF figures into analysis/figures/.

Conventions copied from topic3 generate_figures.py:
- Times serif, bold black text everywhere
- 600 DPI PDF output
- Colorblind-friendly palette
- No top/right spines, light grid
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.patches as mpatches

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

ROOT = Path("/home/Aboya_25R9803/projects/perso/LLMium/projects/03-SLM-CoreMethods/05-federated-slm/fed_omni_adapt")
B9 = Path("/tmp/fedomni_results/batch9")
VERIFY = Path("/tmp/fedomni_results/batch9_verify")
MANIFEST = Path("/tmp/fedomni_results/batch9_manifest.json")
FIG_DIR = ROOT / "analysis" / "figures"

# --- publication style ---
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "font.weight": "bold",
    "text.color": "black",
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "axes.labelcolor": "black",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.color": "black",
    "ytick.color": "black",
    "legend.fontsize": 10,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "lines.linewidth": 2.0,
    "axes.linewidth": 1.2,
    "patch.linewidth": 1.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# colorblind-friendly named palette
C_FLOOR = "#888888"     # neutral gray
C_SMT = "#0288D1"       # dark blue
C_SPATIAL = "#388E3C"   # dark green
C_DCT = "#D32F2F"       # deep red (failed)
C_MIXED = "#7B1FA2"     # purple
C_QWEN = "#F57C00"      # orange
C_SMOLLM = "#0288D1"    # blue
C_TIER_WIN = "#1B5E20"
C_TIER_TOL = "#558B2F"
C_TIER_LOSE = "#BDBDBD"
C_HIGHLIGHT = "#D32F2F"

# --- data loading ---

def expected_filename(cell: dict) -> str:
    c = cell
    return (f"seed{c['seed']}_task{c['task']}_N5_{c['strategy']}_"
            f"K{c['K']}_M{c['M']}_split{c['split']}_"
            f"r{c['r']}_w{c['K_warmup']}_cmld{c['cmld_alpha']:g}_"
            f"fp{c['fedprox_mu']:g}_fam{c['slm_family']}_clip{c['grad_clip']:g}.json")


def load_groups() -> Dict[Tuple, Dict]:
    """Returns dict mapping (task, split, family, strategy, K, M, fp, cmld) → group stats."""
    by_group: Dict[Tuple, List[Tuple[int, float]]] = defaultdict(list)
    manifest = json.loads(MANIFEST.read_text())["cells"]
    for c in manifest:
        p = B9 / expected_filename(c)
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        key = (c["task"], c["split"], c["slm_family"], c["strategy"],
               c["K"], c["M"], c["fedprox_mu"], c["cmld_alpha"])
        by_group[key].append((int(c["seed"]), float(d["final_mean_acc"])))

    out = {}
    for k, seed_accs in by_group.items():
        seed_accs = sorted(set(seed_accs))  # dedupe (Phase A/C reuse)
        seeds = [s for s, _ in seed_accs]
        accs = [a for _, a in seed_accs]
        out[k] = {"seeds": seeds, "accs": accs, "n": len(accs),
                  "mean": np.mean(accs) if accs else None,
                  "std": np.std(accs, ddof=0) if len(accs) >= 2 else 0.0}
    return out


def load_cells() -> List[dict]:
    """Return raw per-cell dicts (for per-round / per-client analyses)."""
    out = []
    for c in json.loads(MANIFEST.read_text())["cells"]:
        p = B9 / expected_filename(c)
        if not p.exists(): continue
        out.append(json.loads(p.read_text()))
    return out


def find_floor(groups, task, split, family, K, M):
    return groups.get((task, split, family, "local_only", K, M, 0, 0))


def paired_delta(method, floor):
    a = dict(zip(method["seeds"], method["accs"]))
    b = dict(zip(floor["seeds"], floor["accs"]))
    shared = sorted(set(a) & set(b))
    deltas = [(a[s] - b[s]) * 100 for s in shared]
    if not deltas:
        return None, None, None
    p = None
    if HAVE_SCIPY and len(deltas) >= 5:
        try:
            p = float(wilcoxon([a[s] for s in shared], [b[s] for s in shared]).pvalue)
        except Exception:
            p = None
    return np.mean(deltas), np.std(deltas, ddof=0) if len(deltas) >= 2 else 0.0, p


def sig_stars(p):
    if p is None: return ""
    if p < 0.05: return "*"
    if p < 0.10: return "$\\circ$"
    return ""


# --- the figures ---

def fig01_main_results(groups):
    """F1: Phase A main results - bar chart, 3 tasks × 2 families × 3 methods."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=False)
    tasks = ["ng20", "yahoo", "ag4"]
    task_labels = ["20NewsGroups (20-cls)", "Yahoo (10-cls)", "AG-News (4-cls)"]
    methods = [("local_only", "Floor", C_FLOOR),
               ("submatrix", r"$\Phi$-SMT", C_SMT),
               ("spatial_bicubic", r"$\Phi$-Spatial", C_SPATIAL)]
    families = ["mixed", "qwen"]

    for ax, task, tlabel in zip(axes, tasks, task_labels):
        x_positions = []
        x_labels = []
        bar_idx = 0
        for fam in families:
            floor = find_floor(groups, task, "iid", fam, 4, 5)
            for strat, name, color in methods:
                g = groups.get((task, "iid", fam, strat, 4, 5, 0, 0))
                if g is None:
                    bar_idx += 1
                    continue
                mean_acc = g["mean"] * 100
                if strat == "local_only":
                    err = g["std"] * 100
                    ax.bar(bar_idx, mean_acc, yerr=err, color=color, edgecolor="black",
                           alpha=0.7, capsize=3, width=0.7)
                else:
                    delta, dstd, p = paired_delta(g, floor)
                    err = g["std"] * 100
                    ax.bar(bar_idx, mean_acc, yerr=err, color=color, edgecolor="black",
                           alpha=0.9, capsize=3, width=0.7)
                    if delta is not None:
                        label = f"+{delta:.1f}{sig_stars(p)}" if delta >= 0 else f"{delta:.1f}"
                        ax.text(bar_idx, mean_acc + err + 0.5, label,
                                ha="center", fontsize=8.5, fontweight="bold")
                bar_idx += 1
            bar_idx += 0.5  # gap between families

        ax.set_title(tlabel)
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks([1, 4.5])
        ax.set_xticklabels(["Mixed", "Qwen-only"])
        ax.set_ylim(bottom=max(0, ax.get_ylim()[0] - 1), top=ax.get_ylim()[1] + 2)

    legend_elements = [Patch(facecolor=c, edgecolor="black", label=n)
                       for _, n, c in methods]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    plt.tight_layout()
    out = FIG_DIR / "F01_main_results.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig02_family_ablation(groups):
    """F5: Family-composition ablation on ng20 (Phase B)."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    fams = [("mixed", "Mixed\n(SmolLM + Qwen)", C_MIXED),
            ("smollm", "SmolLM-only", C_SMOLLM),
            ("qwen", "Qwen-only", C_QWEN)]
    methods = [("local_only", "Floor"),
               ("submatrix", r"$\Phi$-SMT"),
               ("spatial_bicubic", r"$\Phi$-Spatial")]

    x = np.arange(len(methods))
    width = 0.26
    for i, (fam_key, fam_label, color) in enumerate(fams):
        ys = []
        errs = []
        for strat, _ in methods:
            g = groups.get(("ng20", "iid", fam_key, strat, 4, 5, 0, 0))
            ys.append(g["mean"] * 100 if g else 0)
            errs.append(g["std"] * 100 if g else 0)
        positions = x + (i - 1) * width
        ax.bar(positions, ys, width, yerr=errs, label=fam_label,
               color=color, edgecolor="black", alpha=0.85, capsize=3)
        # in-bar value labels (black, set a bit below the top line)
        for xi, y in zip(positions, ys):
            if y > 0:
                ax.text(xi, y - 4.0, f"{y:.2f}", ha="center", va="top",
                        color="black", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in methods])
    ax.set_ylabel("Accuracy (%) on ng20")
    ax.set_title("Family-composition ablation")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "F02_family_ablation.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig03_labelshift(groups):
    """F6: Labelshift comparison - honest negative."""
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    methods = [("local_only", "Floor", C_FLOOR),
               ("submatrix", r"$\Phi$-SMT", C_SMT),
               ("spatial_bicubic", r"$\Phi$-Spatial", C_SPATIAL)]
    splits = [("iid", "IID"), ("labelshift", "Labelshift\n(Dirichlet $\\alpha$=0.3)")]
    x = np.arange(len(splits))
    width = 0.27

    for i, (strat, name, color) in enumerate(methods):
        ys, errs = [], []
        for split, _ in splits:
            g = groups.get(("ng20", split, "mixed", strat, 4, 5, 0, 0))
            ys.append(g["mean"] * 100 if g else 0)
            errs.append(g["std"] * 100 if g else 0)
        positions = x + (i - 1) * width
        ax.bar(positions, ys, width, yerr=errs, label=name,
               color=color, edgecolor="black", alpha=0.85, capsize=3)
        # above-bar value labels in black
        for xi, y, e in zip(positions, ys, errs):
            if y > 0:
                ax.text(xi, y + e + 0.6, f"{y:.2f}", ha="center", va="bottom",
                        color="black", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([s[1] for s in splits])
    ax.set_ylabel("Accuracy (%) on ng20")
    ax.set_title(r"Labelshift: IID vs.\ Dirichlet $\alpha{=}0.3$")
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "F03_labelshift.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig04_task_landscape(groups):
    """F9: Per-task floor vs Φ lift landscape - frames the headroom problem."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    tasks = ["ng20", "yahoo", "ag4"]
    fams = ["mixed", "qwen"]
    width = 0.13
    x = np.arange(len(tasks))

    offset = -2.5 * width
    for fam, hatch in zip(fams, [None, "//"]):
        for j, (strat, color, label) in enumerate([
            ("local_only", C_FLOOR, f"Floor ({fam})"),
            ("submatrix", C_SMT, f"$\\Phi$-SMT ({fam})"),
            ("spatial_bicubic", C_SPATIAL, f"$\\Phi$-Spatial ({fam})"),
        ]):
            ys = [groups.get((t, "iid", fam, strat, 4, 5, 0, 0), {}).get("mean", 0) * 100
                  for t in tasks]
            ax.bar(x + offset, ys, width, color=color, edgecolor="black",
                   alpha=0.85, hatch=hatch, label=label if fam == "qwen" else None)
            offset += width
        offset += 0.08

    ax.set_xticks(x)
    ax.set_xticklabels(["ng20\n(20-class)", "yahoo\n(10-class)", "ag4\n(4-class)"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-task accuracy landscape (Mixed = solid, Qwen = hatched)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=3, fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "F04_task_landscape.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig00_basis_misalignment_cartoon():
    """F0: 1×4 illustration of the V_i basis-misalignment bug (Batch 5/6 finding).

    Synthetic illustration:
      (a) Three random A matrices A_1, A_2, A_3 with different d_i.
      (b) Their canonical Σ V^T forms - different rotations.
      (c) Naive average in unaligned canonical space - distorted.
      (d) Shared-anchor aligned average - clean reconstruction.
    """
    rng = np.random.default_rng(42)
    r = 8
    ds = [64, 96, 128]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))

    # signal model: shared rank-r latent + per-client noise
    Z = rng.standard_normal((r, r)) * 0.5
    As, canonicals, Us, Vs = [], [], [], []
    for d in ds:
        U = rng.standard_normal((d, r))
        U, _ = np.linalg.qr(U)
        A = U @ Z + 0.1 * rng.standard_normal((d, r))
        As.append(A)
        U_i, S_i, Vt_i = np.linalg.svd(A, full_matrices=False)
        canon = np.diag(S_i) @ Vt_i  # Σ V^T
        canonicals.append(canon)
        Us.append(U_i); Vs.append(Vt_i)

    # (a) raw client matrices
    for i, A in enumerate(As):
        # show as stacked heat strip
        ax = axes[0]
        # show only first 24 rows for fit
        block = np.abs(A[:24, :]) / max(0.001, np.abs(A).max())
        ax.imshow(block, aspect="auto", cmap="viridis", extent=[i*(r+1), i*(r+1)+r-0.5, 24, 0])
        ax.text(i*(r+1) + r/2 - 0.5, -2, f"$A_{i+1}\\in\\mathbb{{R}}^{{{ds[i]}\\times{r}}}$",
                ha="center", fontsize=10, fontweight="bold")
    axes[0].set_xlim(-0.5, 3*(r+1))
    axes[0].set_ylim(26, -3)
    axes[0].axis("off")
    axes[0].set_title("(a) Per-client LoRA $A_i$")
    # (a) bottom annotation: neutral observation (gray)
    axes[0].text(0.5, -0.18, f"3 distinct $d_i \\in \\{{{ds[0]}, {ds[1]}, {ds[2]}\\}}$",
                 transform=axes[0].transAxes, ha="center", fontsize=11,
                 fontweight="bold", color="#424242")

    # (b) canonical Σ V^T per client - stacked vertically (3x1) with visible gaps
    gap = 3  # vertical units between blocks
    for i, canon in enumerate(canonicals):
        ax = axes[1]
        norm = np.abs(canon).max()
        block = canon / norm if norm > 0 else canon
        y_top = i * (r + gap)
        y_bot = y_top + r
        # extent=[left, right, bottom, top] with bottom > top for top-down origin
        ax.imshow(block, aspect="equal", cmap="RdBu_r", vmin=-1, vmax=1,
                  extent=[0, r, y_bot, y_top])
        ax.text(-1.0, y_top + r / 2, f"$\\Sigma_{i+1}V_{i+1}^\\top$",
                ha="right", va="center", fontsize=10, fontweight="bold")
    axes[1].set_xlim(-6.0, r + 0.5)
    axes[1].set_ylim(3 * r + 2 * gap + 0.5, -0.5)
    axes[1].axis("off")
    axes[1].set_title("(b) Canonical forms:\neach $V_i$ rotation differs")
    # quantify the misalignment: avg pairwise ||V_i V_j^T - I||_F
    pair_dists = []
    for i in range(len(Vs)):
        for j in range(i + 1, len(Vs)):
            pair_dists.append(np.linalg.norm(Vs[i] @ Vs[j].T - np.eye(r)))
    mean_misalign = float(np.mean(pair_dists)) if pair_dists else 0.0
    # (b) bottom annotation: misalignment quantified (warning - orange)
    axes[1].text(0.5, -0.18,
                 f"avg pairwise $\\|V_iV_j^\\top - I\\|_F$ = {mean_misalign:.2f}",
                 transform=axes[1].transAxes, ha="center", fontsize=11,
                 fontweight="bold", color="#E65100")

    # (c) naive average in canonical space (BAD: rotations unaligned)
    naive_avg = np.mean(canonicals, axis=0)
    norm = np.abs(naive_avg).max()
    axes[2].imshow(naive_avg / norm, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    axes[2].axis("off")
    axes[2].set_title("(c) Naive mean: distorted by\nunaligned $V_i$ rotations")
    fro_naive = np.linalg.norm(naive_avg) / np.linalg.norm(canonicals[0])
    # (c) bottom annotation: naive mean fails (failure - red)
    axes[2].text(0.5, -0.18, f"$\\|\\bar{{A}}\\|_F$ ratio = {fro_naive:.2f}",
                 transform=axes[2].transAxes, ha="center", fontsize=11,
                 fontweight="bold", color=C_HIGHLIGHT)

    # (d) shared-anchor: rotate each canon to a shared V* then average (GOOD)
    # use V_1 as the anchor; align each V_i to V_1 by orthogonal Procrustes
    V_star = Vs[0]
    aligned = []
    for canon, V in zip(canonicals, Vs):
        # align V to V_star via Procrustes: R = argmin |V_star - R V|_F
        M = V_star @ V.T
        Uw, _, Vtw = np.linalg.svd(M)
        R = Uw @ Vtw
        aligned.append(R @ canon)
    aligned_avg = np.mean(aligned, axis=0)
    norm = np.abs(aligned_avg).max()
    axes[3].imshow(aligned_avg / norm, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    axes[3].axis("off")
    axes[3].set_title("(d) Shared-anchor mean:\naligned, recovers signal")
    fro_aligned = np.linalg.norm(aligned_avg) / np.linalg.norm(canonicals[0])
    # (d) bottom annotation: shared-anchor mean succeeds (success - green)
    axes[3].text(0.5, -0.18, f"$\\|\\bar{{A}}\\|_F$ ratio = {fro_aligned:.2f}",
                 transform=axes[3].transAxes, ha="center", fontsize=11,
                 fontweight="bold", color=C_TIER_WIN)

    plt.tight_layout()
    out = FIG_DIR / "F00_basis_misalignment_cartoon.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig05_training_curves(cells):
    """F11: per-round mean accuracy curves (averaged across seeds), 3 panels (tasks)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=False)
    methods = [("local_only", "Floor", C_FLOOR, "-"),
               ("submatrix", r"$\Phi$-SMT", C_SMT, "-"),
               ("spatial_bicubic", r"$\Phi$-Spatial", C_SPATIAL, "-")]
    tasks = [("ng20", "20NewsGroups"), ("yahoo", "Yahoo"), ("ag4", "AG-News")]

    for ax, (task, tlabel) in zip(axes, tasks):
        for strat, name, color, ls in methods:
            # gather per-round mean_acc curves across mixed-family 5 seeds
            curves = []
            for c in cells:
                if (c["task"] == task and c["data_split"] == "iid"
                        and c["slm_family"] == "mixed" and c["strategy"] == strat
                        and c["K_rounds"] == 4 and c["M_local"] == 5):
                    accs = [r["mean_acc"] for r in c["round_metrics"]]
                    if len(accs) == 4:
                        curves.append(accs)
            if not curves: continue
            curves = np.array(curves) * 100  # to %
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            rounds = list(range(1, 5))
            ax.plot(rounds, mean, color=color, marker="o", linestyle=ls,
                    label=name, linewidth=2, markersize=7,
                    markeredgecolor="black", markeredgewidth=0.6)
            ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.18)
        ax.set_title(tlabel)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Mean accuracy (%)")
        ax.set_xticks([1, 2, 3, 4])
        ax.legend(loc="lower right", fontsize=8.5)

    plt.tight_layout()
    out = FIG_DIR / "F05_training_curves.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig06_backbone_dim_vs_acc(cells):
    """F13: aggregated per-backbone accuracy (mean +/- std), Floor vs Phi-Spatial.

    Aggregates all per-client accuracies across all seeds and cells (mixed family,
    K=4, M=5) by client hidden dimension, then shows mean +/- std per backbone for
    Floor vs Phi-Spatial. Cleaner than dumping ~125 raw dots per panel.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=False)
    tasks = [("ng20", "20NewsGroups"), ("yahoo", "Yahoo"), ("ag4", "AG-News")]
    methods = [("local_only", "Floor", C_FLOOR, "o", "-"),
               ("submatrix", r"$\Phi$-SMT", C_SMT, "D", "--"),
               ("spatial_bicubic", r"$\Phi$-Spatial", C_SPATIAL, "s", "-")]
    dims = [576, 896, 960, 1536]
    dim_labels = ["576\nSmolLM2\n-135M", "896\nQwen2.5\n-0.5B",
                  "960\nSmolLM2\n-360M", "1536\nQwen2.5\n-1.5B"]

    for ax, (task, tlabel) in zip(axes, tasks):
        per_strat_stats = {}
        for strat, name, color, marker, ls in methods:
            by_dim = {d: [] for d in dims}
            for c in cells:
                if (c["task"] == task and c["data_split"] == "iid"
                        and c["slm_family"] == "mixed" and c["strategy"] == strat
                        and c["K_rounds"] == 4 and c["M_local"] == 5):
                    for d, a in zip(c["client_dims"], c["final_per_client_acc"]):
                        if d in by_dim:
                            by_dim[d].append(a * 100)
            means = [np.mean(by_dim[d]) if by_dim[d] else np.nan for d in dims]
            stds = [np.std(by_dim[d]) if by_dim[d] else 0 for d in dims]
            per_strat_stats[strat] = (means, stds)
            ax.errorbar(range(len(dims)), means, yerr=stds, marker=marker,
                        color=color, label=name, linewidth=2.2, markersize=8,
                        linestyle=ls, capsize=4,
                        markeredgecolor="black", markeredgewidth=0.6)
        # shaded lift band between Floor and Phi-Spatial
        floor_means = per_strat_stats["local_only"][0]
        spa_means = per_strat_stats["spatial_bicubic"][0]
        ax.fill_between(range(len(dims)), floor_means, spa_means,
                        color=C_SPATIAL, alpha=0.12, zorder=0)
        # per-backbone lift annotations: place above the higher error-bar top
        # with a white bbox so the text never collides with a line or cap
        floor_stds = per_strat_stats["local_only"][1]
        spa_stds = per_strat_stats["spatial_bicubic"][1]
        for k, (fm, sm, fs, ss) in enumerate(zip(floor_means, spa_means,
                                                  floor_stds, spa_stds)):
            if np.isnan(fm) or np.isnan(sm):
                continue
            lift = sm - fm
            top_y = max(fm + fs, sm + ss) + 0.6
            ax.text(k, top_y, f"Δ{lift:+.1f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold",
                    color=C_TIER_WIN if lift >= 1 else "black",
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.85, pad=1.2),
                    zorder=10)
        # bump the y-axis a touch so the annotations always have headroom
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax + 1.5)
        ax.set_title(tlabel)
        ax.set_xlabel("Client backbone (hidden dim)")
        ax.set_ylabel("Per-client accuracy (%)")
        ax.set_xticks(range(len(dims)))
        ax.set_xticklabels(dim_labels, fontsize=8)
        # shift legend slightly inboard (right edge at 3/4 of axes width) so it
        # doesn't sit flush against the right spine
        ax.legend(loc="lower right", bbox_to_anchor=(0.75, 0.0),
                  fontsize=8.5)
    plt.tight_layout()
    out = FIG_DIR / "F06_backbone_dim_vs_acc.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig07_wall_time_pareto(cells):
    """F14: wall-time vs accuracy Pareto: shows Φ adds modest compute for measurable lift."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=False)
    tasks = [("ng20", "20NewsGroups"), ("yahoo", "Yahoo"), ("ag4", "AG-News")]
    methods = [("local_only", "Floor", C_FLOOR, "o"),
               ("submatrix", r"$\Phi$-SMT", C_SMT, "D"),
               ("spatial_bicubic", r"$\Phi$-Spatial", C_SPATIAL, "s")]

    for ax, (task, tlabel) in zip(axes, tasks):
        for strat, name, color, marker in methods:
            xs, ys = [], []
            for c in cells:
                if (c["task"] == task and c["data_split"] == "iid"
                        and c["slm_family"] == "mixed" and c["strategy"] == strat
                        and c["K_rounds"] == 4 and c["M_local"] == 5):
                    xs.append(c["wall_seconds"] / 60.0)  # minutes
                    ys.append(c["final_mean_acc"] * 100)
            if xs:
                ax.scatter(xs, ys, color=color, marker=marker, s=110,
                           edgecolor="black", linewidth=0.7, alpha=0.85,
                           label=name)
        ax.set_title(tlabel)
        ax.set_xlabel("Wall time per cell (min)")
        ax.set_ylabel("Final mean accuracy (%)")
        ax.legend(loc="lower right", fontsize=8.5)
    plt.tight_layout()
    out = FIG_DIR / "F07_wall_time_pareto.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig08_backbone_diversity():
    """F17: 4-backbone diversity bar chart - shows the heterogeneity that Φ must accommodate."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 3.4))
    backbones = [
        ("SmolLM2-135M", 576, "0.135B", C_SMOLLM),
        ("Qwen2.5-0.5B", 896, "0.5B", C_QWEN),
        ("SmolLM2-360M", 960, "0.360B", C_SMOLLM),
        ("Qwen2.5-1.5B", 1536, "1.5B", C_QWEN),
    ]
    x = np.arange(len(backbones))
    dims = [b[1] for b in backbones]
    colors = [b[3] for b in backbones]
    ax.bar(x, dims, color=colors, edgecolor="black", alpha=0.85, width=0.6)
    for i, (name, dim, size, _) in enumerate(backbones):
        ax.text(i, dim + 30, f"{dim}", ha="center", fontsize=11, fontweight="bold")
        ax.text(i, -90, size, ha="center", fontsize=10, fontweight="bold", color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([b[0] for b in backbones], fontsize=10, rotation=0)
    ax.set_ylabel("Hidden dimension $d_i$")
    ax.set_ylim(-160, 1750)
    ax.set_title("Heterogeneous-backbone fleet (4 distinct hidden dims)")
    # legend for family
    legend_elements = [
        Patch(facecolor=C_SMOLLM, alpha=0.85, label="SmolLM family"),
        Patch(facecolor=C_QWEN, alpha=0.85, label="Qwen family"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "F08_backbone_diversity.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig09_phi_operator_schematic():
    """F19: schematic of Φ-SMT vs Φ-Spatial vs Φ-DCT, 1x3 panel."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    rng = np.random.default_rng(7)
    titles = [r"$\Phi$-SMT (SVD)",
              r"$\Phi$-Spatial (bicubic resize)",
              r"$\Phi$-DCT (Fourier)"]
    colors = [C_SMT, C_SPATIAL, C_DCT]
    descriptions = [
        r"$A_i = U_i \Sigma_i V_i^\top$" + "\nCanonical: $\\Sigma_i V_i^\\top$",
        r"Resize to $d_\star$" + "\n(bicubic kernel)",
        r"2-D DCT $\to$ keep" + "\nlow-freq $r{\\times}r$ block",
    ]

    for ax, title, color, desc in zip(axes, titles, colors, descriptions):
        # client A_i representations as 3 small heatmaps
        for k, d in enumerate([64, 96, 128]):
            block = np.abs(rng.standard_normal((d, 8))) / 3
            block = block[:24, :]  # show only top 24 rows for compactness
            ax.imshow(block, cmap="viridis", aspect="auto",
                      extent=[k*9, k*9 + 8 - 0.2, 24, 0])
            ax.text(k*9 + 4 - 0.1, -2, f"$A_{k+1}$",
                    ha="center", fontsize=9, fontweight="bold")
        # arrow to canonical r×r block
        arr_y = 12
        ax.annotate("", xy=(30, arr_y), xytext=(27, arr_y),
                    arrowprops=dict(arrowstyle="->", color="black", linewidth=1.5))
        # canonical r×r block
        canon = np.abs(rng.standard_normal((8, 8))) / 3
        ax.imshow(canon, cmap="viridis", aspect="auto",
                  extent=[31, 39, 16, 8])
        ax.text(35, 7, r"$\bar{A}\in\mathbb{R}^{r\times r}$",
                ha="center", fontsize=9, fontweight="bold")
        ax.text(35, 23, desc, ha="center", fontsize=8,
                color=color, fontweight="bold")
        ax.set_xlim(-1, 41)
        ax.set_ylim(28, -3)
        ax.axis("off")
        ax.set_title(title, color=color)

    plt.tight_layout()
    out = FIG_DIR / "F09_phi_operator_schematic.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig10_cdf_of_delta(groups):
    """F20: survival function (CCDF) of Delta across all paired diffs.

    Shows P(Delta >= t) directly - the y-axis reads "fraction of cells where
    Phi delivers at least t pp". Threshold crossings at +0 / +3 / +5 are
    annotated with explicit percentages.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 4.5))
    smt_deltas, spa_deltas = [], []
    for task in ["ng20", "yahoo", "ag4"]:
        for fam in ["mixed", "qwen"]:
            floor = find_floor(groups, task, "iid", fam, 4, 5)
            for strat, lst in [("submatrix", smt_deltas),
                                ("spatial_bicubic", spa_deltas)]:
                g = groups.get((task, "iid", fam, strat, 4, 5, 0, 0))
                if g and floor:
                    a = dict(zip(g["seeds"], g["accs"]))
                    b = dict(zip(floor["seeds"], floor["accs"]))
                    shared = sorted(set(a) & set(b))
                    lst.extend([(a[s] - b[s]) * 100 for s in shared])

    def ccdf(arr):
        a = np.sort(arr)
        n = len(a)
        # P(X >= t) at each t in sorted arr is (n - i) / n for the i-th smallest
        return a, (n - np.arange(n)) / n

    def p_at(arr, t):
        return float(np.mean(np.asarray(arr) >= t)) if len(arr) else 0.0

    method_data = [(smt_deltas, r"$\Phi$-SMT", C_SMT, "-"),
                   (spa_deltas, r"$\Phi$-Spatial", C_SPATIAL, "-")]

    for arr, name, color, ls in method_data:
        if not arr: continue
        xs, ys = ccdf(arr)
        # add a leading point at the left edge for visual flat-start
        xs = np.concatenate([[-3.0], xs])
        ys = np.concatenate([[1.0], ys])
        ax.step(xs, ys, where="post", color=color, linewidth=2.5,
                linestyle=ls, label=f"{name} (n={len(arr)})")

    # tier zone shading: only render bands that any method actually populates
    all_arr = smt_deltas + spa_deltas
    ax.axvspan(-3, 0, color=C_HIGHLIGHT, alpha=0.06, zorder=0)
    if any(0 <= d < 3 for d in all_arr):
        ax.axvspan(0, 3, color=C_TIER_LOSE, alpha=0.18, zorder=0)
    if any(3 <= d < 5 for d in all_arr):
        ax.axvspan(3, 5, color=C_TIER_TOL, alpha=0.20, zorder=0)
    if any(d >= 5 for d in all_arr):
        ax.axvspan(5, 15, color=C_TIER_WIN, alpha=0.25, zorder=0)

    # threshold markers + annotations: keep a threshold only if at least one
    # method has any cell that crosses it (otherwise the reference is misleading)
    thresholds = [0, 3, 5]
    threshold_styles = [(r"$\geq 0$\,pp", "black"),
                        (r"$\geq +3$\,pp", C_TIER_TOL),
                        (r"$\geq +5$\,pp", C_TIER_WIN)]
    for t, (tlabel, tcolor) in zip(thresholds, threshold_styles):
        if not any(d >= t for d in all_arr):
            continue
        ax.axvline(t, color=tcolor, linewidth=1.1, linestyle="--", alpha=0.85)
        # annotate per-method % at this threshold
        for arr, name, color, _ in method_data:
            p = p_at(arr, t) * 100
            ax.scatter([t], [p / 100], color=color, marker="o", s=70,
                       edgecolor="black", linewidth=0.8, zorder=5)
            ax.text(t + 0.18, p / 100 + 0.025, f"{p:.0f}\\%",
                    color=color, fontsize=10, fontweight="bold",
                    ha="left", va="bottom")
        ax.text(t, 1.07, tlabel, ha="center", va="bottom", fontsize=9,
                color=tcolor, fontweight="bold")

    ax.set_xlabel(r"Threshold $t$ (pp)")
    ax.set_ylabel(r"$P(\Delta \geq t)$ across all (task, family, seed) cells")
    ax.set_title(r"Survival of $\Phi$ lift across cells")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(-3, 14)
    ax.set_ylim(0, 1.13)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    plt.tight_layout()
    out = FIG_DIR / "F10_cdf_of_delta.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig11_loss_curves(cells):
    """F21: per-round avg_loss curves: shows Φ reduces loss faster than Floor."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    methods = [("local_only", "Floor", C_FLOOR),
               ("submatrix", r"$\Phi$-SMT", C_SMT),
               ("spatial_bicubic", r"$\Phi$-Spatial", C_SPATIAL)]
    tasks = [("ng20", "20NewsGroups"), ("yahoo", "Yahoo"), ("ag4", "AG-News")]

    for ax, (task, tlabel) in zip(axes, tasks):
        for strat, name, color in methods:
            curves = []
            for c in cells:
                if (c["task"] == task and c["data_split"] == "iid"
                        and c["slm_family"] == "mixed" and c["strategy"] == strat
                        and c["K_rounds"] == 4 and c["M_local"] == 5):
                    losses = [r["avg_loss"] for r in c["round_metrics"]]
                    if len(losses) == 4:
                        curves.append(losses)
            if not curves: continue
            curves = np.array(curves)
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            rounds = list(range(1, 5))
            ax.plot(rounds, mean, color=color, marker="o", label=name,
                    linewidth=2, markersize=7, markeredgecolor="black", markeredgewidth=0.6)
            ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.18)
        ax.set_title(tlabel)
        ax.set_xlabel("Communication round")
        ax.set_ylabel("Average client loss")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=8.5)
    plt.tight_layout()
    out = FIG_DIR / "F11_loss_curves.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig12_seed2_deep_dive(cells):
    """F23: ag4-qwen seed-2 deep dive in a 2x2 grid.

    Row 1: Floor vs Phi-Spatial. Row 2: Floor vs Phi-SMT. Floor is repeated in
    each row so both Phi operators can be compared against the same matched
    baseline within their row.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4), sharey=True, sharex=True)
    grid = [
        [("local_only", "Floor"), ("spatial_bicubic", r"$\Phi$-Spatial")],
        [("local_only", "Floor"), ("submatrix", r"$\Phi$-SMT")],
    ]

    # 5 distinct colors, one per client (colorblind-friendly)
    client_palette = [C_HIGHLIGHT, C_SMT, C_SPATIAL, C_QWEN, C_MIXED]

    for row_idx, row in enumerate(grid):
        for col_idx, (strat, mname) in enumerate(row):
            ax = axes[row_idx, col_idx]
            target = [c for c in cells
                      if c["task"] == "ag4" and c["data_split"] == "iid"
                      and c["slm_family"] == "qwen" and c["strategy"] == strat
                      and c["K_rounds"] == 4 and c["M_local"] == 5
                      and c["seed"] == 2]
            if not target:
                ax.set_title(mname); continue
            c = target[0]
            rounds = [0] + [r["round"] for r in c["round_metrics"]]
            per_client = np.array([[0.0] * 5]
                                   + [r["per_client_acc"] for r in c["round_metrics"]])
            for k in range(5):
                d = c["client_dims"][k]
                color = client_palette[k % len(client_palette)]
                ax.plot(rounds, per_client[:, k] * 100, marker="o",
                        label=f"client {k+1} ($d{{=}}{d}$)", linewidth=1.8,
                        color=color, markeredgecolor="black", markeredgewidth=0.5)
            ax.plot(rounds,
                    np.array([0] + [r["mean_acc"] for r in c["round_metrics"]]) * 100,
                    marker="*", color="#1A237E", linewidth=2.5, markersize=11,
                    markeredgecolor="black", markeredgewidth=0.6,
                    label="mean", zorder=5)
            if row_idx == 1:
                ax.set_xlabel("Communication round")
            if col_idx == 0:
                ax.set_ylabel("Per-client accuracy (%)")
            ax.set_title(f"{mname}: seed-2 ag4 Qwen-only")
            ax.set_xticks(rounds)
            ax.legend(loc="lower right", fontsize=7.5, ncol=2)
    plt.tight_layout()
    out = FIG_DIR / "F12_seed2_deep_dive.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig13_round_delta_growth(cells):
    """F24: when does Φ start winning during training? Round-by-round Δ vs Floor."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=False)
    tasks = [("ng20", "20NewsGroups"), ("yahoo", "Yahoo"), ("ag4", "AG-News")]
    methods = [("submatrix", r"$\Phi$-SMT", C_SMT),
               ("spatial_bicubic", r"$\Phi$-Spatial", C_SPATIAL)]

    for ax, (task, tlabel) in zip(axes, tasks):
        # gather floor curves
        floor_curves = []
        for c in cells:
            if (c["task"] == task and c["data_split"] == "iid"
                    and c["slm_family"] == "mixed" and c["strategy"] == "local_only"
                    and c["K_rounds"] == 4 and c["M_local"] == 5):
                accs = [r["mean_acc"] for r in c["round_metrics"]]
                if len(accs) == 4:
                    floor_curves.append((c["seed"], accs))
        floor_by_seed = {s: a for s, a in floor_curves}
        for strat, name, color in methods:
            method_curves = []
            for c in cells:
                if (c["task"] == task and c["data_split"] == "iid"
                        and c["slm_family"] == "mixed" and c["strategy"] == strat
                        and c["K_rounds"] == 4 and c["M_local"] == 5):
                    accs = [r["mean_acc"] for r in c["round_metrics"]]
                    if len(accs) == 4 and c["seed"] in floor_by_seed:
                        method_curves.append((c["seed"], accs))
            # paired delta per round
            rounds = list(range(1, 5))
            per_round_deltas = []
            for r in range(4):
                deltas = [(m_accs[r] - floor_by_seed[s][r]) * 100
                          for s, m_accs in method_curves if s in floor_by_seed]
                per_round_deltas.append(deltas)
            means = [np.mean(d) for d in per_round_deltas]
            stds = [np.std(d) for d in per_round_deltas]
            ax.errorbar(rounds, means, yerr=stds, marker="o", color=color,
                        label=name, linewidth=2.5, capsize=4, markersize=8,
                        markeredgecolor="black", markeredgewidth=0.6)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax.axhline(3, color=C_TIER_TOL, linewidth=0.8, linestyle=":")
        ax.set_title(tlabel)
        ax.set_xlabel("Communication round")
        ax.set_ylabel(r"$\Delta$ vs floor (pp)")
        ax.set_xticks(rounds)
        ax.legend(loc="best", fontsize=8.5)
    plt.tight_layout()
    out = FIG_DIR / "F13_round_delta_growth.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig14_violin_per_method(groups):
    """F26: per-seed accuracy violin per (method, family) on ng20."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.8))
    rows = []
    for fam in ["mixed", "smollm", "qwen"]:
        for strat in ["local_only", "submatrix", "spatial_bicubic"]:
            g = groups.get(("ng20", "iid", fam, strat, 4, 5, 0, 0))
            if g and g["accs"]:
                rows.append({"fam": fam, "strat": strat,
                             "accs": [a * 100 for a in g["accs"]]})

    fam_colors = {"mixed": C_MIXED, "smollm": C_SMOLLM, "qwen": C_QWEN}
    # legend uses the full single-line family name
    fam_legend = {"mixed": "Mixed", "smollm": "SmolLM-only", "qwen": "Qwen-only"}
    # x-tick labels split long family names across 3 lines (method / family / suffix)
    fam_tick = {"mixed": "Mixed", "smollm": "SmolLM\n-only", "qwen": "Qwen\n-only"}
    strat_labels = {"local_only": "Floor", "submatrix": "$\\Phi$-SMT",
                    "spatial_bicubic": "$\\Phi$-Spatial"}

    positions = np.arange(len(rows))
    parts = ax.violinplot([r["accs"] for r in rows],
                           positions=positions, widths=0.7, showmeans=True,
                           showmedians=False, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(fam_colors[rows[i]["fam"]])
        pc.set_edgecolor("black")
        pc.set_alpha(0.7)
    for key in ("cbars", "cmins", "cmaxes", "cmeans"):
        parts[key].set_color("black")
    # scatter individual seeds on top
    for i, r in enumerate(rows):
        for a in r["accs"]:
            ax.plot(i, a, "o", color="black", markersize=4, zorder=5)

    labels = [f"{strat_labels[r['strat']]}\n{fam_tick[r['fam']]}" for r in rows]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=0)
    ax.set_ylabel("Accuracy (%) on ng20")
    ax.set_title("5-seed accuracy distribution (violin + per-seed dot)")
    # family legend (uses full single-line names)
    legend_elements = [Patch(facecolor=c, alpha=0.7, label=fam_legend[k])
                       for k, c in fam_colors.items()]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "F14_violin_per_method.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig15_cohens_d(groups):
    """F30: Cohen's d effect sizes per cell."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    TASK_PRETTY = {"ng20": "20NG", "yahoo": "Yahoo", "ag4": "AG-News"}
    FAM_PRETTY = {"mixed": "Mixed", "qwen": "Qwen-only", "smollm": "SmolLM-only"}
    STRAT_PRETTY = {"SMT": r"$\Phi$-SMT", "Spa": r"$\Phi$-Spatial"}

    points = []
    for task in ["ng20", "yahoo", "ag4"]:
        for fam in ["mixed", "qwen"]:
            floor = find_floor(groups, task, "iid", fam, 4, 5)
            for strat, sname, color in [("submatrix", "SMT", C_SMT),
                                          ("spatial_bicubic", "Spa", C_SPATIAL)]:
                g = groups.get((task, "iid", fam, strat, 4, 5, 0, 0))
                if not (g and floor): continue
                a = dict(zip(g["seeds"], g["accs"]))
                b = dict(zip(floor["seeds"], floor["accs"]))
                shared = sorted(set(a) & set(b))
                deltas = np.array([(a[s] - b[s]) * 100 for s in shared])
                if len(deltas) < 2: continue
                pooled_sd = np.std(deltas, ddof=1)
                d_eff = deltas.mean() / pooled_sd if pooled_sd > 0 else 0
                label = f"{TASK_PRETTY[task]} / {FAM_PRETTY[fam]} $\\cdot$ {STRAT_PRETTY[sname]}"
                points.append({"label": label, "d": d_eff,
                               "delta": deltas.mean(), "color": color})

    points.sort(key=lambda p: p["d"])
    y = np.arange(len(points))
    colors = [p["color"] for p in points]
    ax.barh(y, [p["d"] for p in points], color=colors, edgecolor="black", alpha=0.85)

    # in-bar value labels in black at the right end of each bar
    for yi, p in enumerate(points):
        d = p["d"]
        dx = 0.05 if d >= 0 else -0.05
        ha = "left" if d >= 0 else "right"
        ax.text(d + dx, yi, f"{d:.2f}", va="center", ha=ha,
                fontsize=9, fontweight="bold", color="black")

    # Cohen's d thresholds
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(0.2, color="gray", linewidth=0.8, linestyle=":")
    ax.axvline(0.5, color=C_TIER_TOL, linewidth=0.8, linestyle=":")
    ax.axvline(0.8, color=C_TIER_WIN, linewidth=0.8, linestyle=":")
    # threshold annotations at the top of the axes, rotated vertically so they
    # never run into one another even though the d=0.2/0.5/0.8 thresholds are
    # bunched tightly relative to the full x-axis range; clip_on=False ensures
    # the full label renders even when it extends beyond the top of the axes
    top_y = len(points) - 0.5
    ax.text(0.2, top_y, "small", fontsize=8.5, color="gray",
            ha="center", va="bottom", fontweight="bold", rotation=90,
            clip_on=False)
    ax.text(0.5, top_y, "medium", fontsize=8.5, color=C_TIER_TOL,
            ha="center", va="bottom", fontweight="bold", rotation=90,
            clip_on=False)
    ax.text(0.8, top_y, "large", fontsize=8.5, color=C_TIER_WIN,
            ha="center", va="bottom", fontweight="bold", rotation=90,
            clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels([p["label"] for p in points], fontsize=9)
    ax.set_xlabel(r"Cohen's $d$ (paired)")
    ax.set_title("Effect size by Cohen's $d$")

    # widen xlim so the in-bar value labels never get clipped
    if points:
        d_min = min(p["d"] for p in points)
        d_max = max(p["d"] for p in points)
        ax.set_xlim(min(0, d_min) - 0.4, d_max + 0.8)

    # method legend at bottom-right
    legend_elements = [
        Patch(facecolor=C_SMT, edgecolor="black", label=r"$\Phi$-SMT"),
        Patch(facecolor=C_SPATIAL, edgecolor="black", label=r"$\Phi$-Spatial"),
    ]
    ax.legend(handles=legend_elements, loc="lower right",
              fontsize=9, frameon=True, framealpha=0.92)

    plt.tight_layout()
    out = FIG_DIR / "F15_cohens_d.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig16_best_worst_trajectory(cells):
    """F32: best-case vs worst-case Φ-Spatial cell training trajectories."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    # best cell: ag4 qwen seed-2 (catastrophic floor → Φ rescues to 87)
    # worst cell: yahoo qwen seed-1 (small lift)
    candidates = []
    for c in cells:
        if (c["data_split"] == "iid" and c["strategy"] == "spatial_bicubic"
                and c["K_rounds"] == 4 and c["M_local"] == 5):
            label = f"{c['task']}-{c['slm_family']}-s{c['seed']}"
            accs = [r["mean_acc"] for r in c["round_metrics"]]
            candidates.append((label, accs, c["task"], c["slm_family"], c["seed"]))

    # find best lift cell (vs matched floor)
    floor_lookup = {}
    for c in cells:
        if (c["data_split"] == "iid" and c["strategy"] == "local_only"
                and c["K_rounds"] == 4 and c["M_local"] == 5):
            floor_lookup[(c["task"], c["slm_family"], c["seed"])] = \
                [r["mean_acc"] for r in c["round_metrics"]]
    lifts = []
    for label, accs, task, fam, seed in candidates:
        key = (task, fam, seed)
        if key in floor_lookup:
            lift = (accs[-1] - floor_lookup[key][-1]) * 100
            lifts.append((lift, label, accs, key))
    lifts.sort()
    if len(lifts) < 4:
        return
    pick = [lifts[-1], lifts[-2], lifts[0], lifts[1]]
    pick_colors = [C_TIER_WIN, C_TIER_TOL, C_TIER_LOSE, "#FF7043"]
    for (lift, label, accs, key), color in zip(pick, pick_colors):
        floor_accs = [a * 100 for a in floor_lookup[key]]
        spa_accs = [a * 100 for a in accs]
        rounds = list(range(1, 5))
        ax.plot(rounds, spa_accs, marker="o", color=color, linewidth=2.5,
                label=f"{label}: $\\Delta$={lift:+.1f}", markeredgecolor="black",
                markeredgewidth=0.6)
        ax.plot(rounds, floor_accs, marker="s", color=color, linewidth=1,
                linestyle="--", alpha=0.6, markeredgecolor="black",
                markeredgewidth=0.4)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Mean accuracy (%)")
    ax.set_title(r"Best-lift and worst-lift $\Phi$-Spatial cells (solid=$\Phi$, dashed=Floor)")
    ax.set_xticks([1, 2, 3, 4])
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out = FIG_DIR / "F16_best_worst_trajectory.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def fig17_method_lift_breakdown(groups):
    """F36: per-cell lift breakdown for SMT and Spatial, banded by Delta thresholds.

    Three threshold bands (Delta>=+5, +3<=Delta<+5, Delta<+3) use the same
    visual encoding as the internal tier scheme but are labelled with the
    numeric thresholds in the figure - the words "Win", "Tolerable", "Loser"
    are internal jargon and do not appear in the rendered output.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    all_deltas = []
    for ax, (strat, mname) in zip(axes, [("submatrix", r"$\Phi$-SMT"),
                                          ("spatial_bicubic", r"$\Phi$-Spatial")]):
        rows = []
        for task in ["ng20", "yahoo", "ag4"]:
            for fam in ["mixed", "qwen"]:
                floor = find_floor(groups, task, "iid", fam, 4, 5)
                g = groups.get((task, "iid", fam, strat, 4, 5, 0, 0))
                if g and floor:
                    d, _, _ = paired_delta(g, floor)
                    if d is not None:
                        rows.append((f"{task[:3]}\n{fam[:3]}", d))
        labels = [r[0] for r in rows]
        deltas = [r[1] for r in rows]
        all_deltas.extend(deltas)
        colors = [C_TIER_WIN if d >= 5 else (C_TIER_TOL if d >= 3 else C_TIER_LOSE)
                  for d in deltas]
        x = np.arange(len(rows))
        ax.bar(x, deltas, color=colors, edgecolor="black", alpha=0.85)
        for xi, d in zip(x, deltas):
            ax.text(xi, max(d, 0) + 0.3, f"{d:+.2f}", ha="center",
                    fontsize=9, fontweight="bold")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.axhline(3, color=C_TIER_TOL, linewidth=0.8, linestyle=":")
        # only render the +5 reference line if any cell actually reaches it
        if any(d >= 5 for d in deltas):
            ax.axhline(5, color=C_TIER_WIN, linewidth=0.8, linestyle=":")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel(r"$\Delta$ vs floor (pp)")
        ax.set_title(mname)
        ax.set_ylim(-1, 9)

    # build legend dynamically from the bands actually populated by the data
    legend_elements = []
    if any(d < 3 for d in all_deltas):
        legend_elements.append(Patch(facecolor=C_TIER_LOSE,
                                     label=r"$\Delta < +3$\,pp"))
    if any(3 <= d < 5 for d in all_deltas):
        legend_elements.append(Patch(facecolor=C_TIER_TOL,
                                     label=r"$+3 \leq \Delta < +5$\,pp"))
    if any(d >= 5 for d in all_deltas):
        legend_elements.append(Patch(facecolor=C_TIER_WIN,
                                     label=r"$\Delta \geq +5$\,pp"))
    if legend_elements:
        fig.legend(handles=legend_elements, loc="lower center",
                   ncol=len(legend_elements),
                   bbox_to_anchor=(0.5, -0.04), frameon=False)
    plt.tight_layout()
    out = FIG_DIR / "F17_method_lift_breakdown.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  wrote {out}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    groups = load_groups()
    cells = load_cells()
    print(f"loaded {len(groups)} groups, {len(cells)} cells\n")

    fig00_basis_misalignment_cartoon()
    fig01_main_results(groups)
    fig02_family_ablation(groups)
    fig03_labelshift(groups)
    fig04_task_landscape(groups)
    fig05_training_curves(cells)
    fig06_backbone_dim_vs_acc(cells)
    fig07_wall_time_pareto(cells)
    fig08_backbone_diversity()
    fig09_phi_operator_schematic()
    fig10_cdf_of_delta(groups)
    fig11_loss_curves(cells)
    fig12_seed2_deep_dive(cells)
    fig13_round_delta_growth(cells)
    fig14_violin_per_method(groups)
    fig15_cohens_d(groups)
    fig16_best_worst_trajectory(cells)
    fig17_method_lift_breakdown(groups)


if __name__ == "__main__":
    main()
