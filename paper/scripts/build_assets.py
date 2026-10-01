"""Build every generated table, figure, plot-data file and number macro for the preprint.

One documented entry point. Inputs are read-only:
  * results/campaigns/five_seed_v1/20260925_eval_fix_v3/   (compact campaign export ``E``)
  * results/compute_cost/benchmark_*.json                   (post hoc cost benchmarks)
  * paper/data/flop_recount.json        (from scripts/recount_flops.py)
  * config/campaigns/five_seed_v1/*.yaml, config/shared/mpi3d_position_manifests/*.json
  * data/concept2-mnist/                                     (local MNIST copy, illustration only)
  * notebooks/assets/mpi3d_one_to_many_scenarios.png         (cropped, not restyled)

Outputs (all under paper/): tables/*.tex, figures/*.{pdf,png},
data/*.csv and data/manifest.json. Nothing under results/ or E is written.

Primary numbers are recomputed with the repository analyzer
(``iwm_replication.five_seed_analysis.analyze_campaign``) and asserted equal to the
exported analysis.json before any table or figure is produced.

Usage (repository root):
  PYTHONPATH=src python paper/scripts/build_assets.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parent
E = ROOT / "results/campaigns/five_seed_v1/20260925_eval_fix_v3"
COST = ROOT / "results/compute_cost"
TABLES, FIGURES, DATA = PAPER / "tables", PAPER / "figures", PAPER / "data"
sys.path.insert(0, str(ROOT / "src"))

ROLES = ("R0", "R1", "S0", "S1")
REPS = (1, 2, 3, 4, 5)
DATASETS = ("moving_mnist", "mpi3d")
DSNAME = {"moving_mnist": "Moving-MNIST", "mpi3d": "MPI3D"}
# Descriptive model names: adapted reference recipe (BYOL/AdaSSL) or shared prediction family
# (Det = deterministic; Var = variational Gaussian residual). Internal role codes stay
# R0/R1/S0/S1 in the campaign evidence and plot data; LaTeX uses macros from main.tex.
NAME = {"R0": "BYOL reference", "R1": "AdaSSL reference", "S0": "Shared-Det", "S1": "Shared-Var"}
NAME2 = {"R0": "BYOL\nreference", "R1": "AdaSSL\nreference", "S0": "Shared-\nDet", "S1": "Shared-\nVar"}
TEX = {"R0": r"\emadet{}", "R1": r"\emavar{}", "S0": r"\shdet{}", "S1": r"\shvar{}"}
CONTRAST_TEX = {"S1-R1": r"\shvar{} $-$ \emavar{}", "S1-S0": r"\shvar{} $-$ \shdet{}"}
CONTRAST_PLOT = {"S1-R1": "Shared-Var − AdaSSL reference", "S1-S0": "Shared-Var − Shared-Det"}
DOWN, UP, NODIR = r"$\downarrow$", r"$\uparrow$", ""
# Validated colour-blind-safe palette (dataviz validator, all pairs, light surface):
# EMA roles blue, shared roles orange; deterministic lighter/open, stochastic darker/filled.
COLOR = {"R0": "#4B9FD5", "R1": "#1F4E99", "S0": "#E0A020", "S1": "#C4501A"}
MARKER = {"R0": "o", "R1": "o", "S0": "s", "S1": "s"}
FILLED = {"R0": False, "R1": True, "S0": False, "S1": True}
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#d9d9d9"
REP_OFFSET = {r: (r - 3) * 0.075 for r in REPS}  # replications 1..5 left to right in every panel

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.labelcolor": INK, "axes.titlecolor": INK, "pdf.fonttype": 42,
    "ps.fonttype": 42, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
})

MANIFEST = {"generator": "paper/scripts/build_assets.py", "inputs": {}, "outputs": {}}
PLOT_ROWS = defaultdict(list)
NUMBERS = {}


# ----------------------------------------------------------------------------- utilities
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def register_input(path):
    MANIFEST["inputs"][rel(path)] = sha256(path)
    return path


def load_json(path):
    return json.loads(Path(register_input(path)).read_text())


def record(output, **info):
    MANIFEST["outputs"][output] = info


def fmt(value, digits=3):
    return f"{value:.{digits}f}"


def sig(value, digits=3):
    return f"{value:.{digits}g}"


def num(name, text):
    if name in NUMBERS and NUMBERS[name] != text:
        raise ValueError(f"Conflicting number macro {name}")
    NUMBERS[name] = text
    return text


ROLE_WORD = {"R0": "RZero", "R1": "ROne", "S0": "SZero", "S1": "SOne"}
DS_WORD = {"moving_mnist": "MM", "mpi3d": "MP"}


def mean_sd(values):
    return st.mean(values), st.stdev(values)


def bold_best(values, direction, render, target=None):
    """Render values, wrapping the best one(s) in \best{}.

    direction: "min", "max", "target" (closest to target) or None. Ties at the displayed
    precision are all bolded; nothing is bolded when every displayed value is equal.
    """
    texts = [render(v) if v is not None else "--" for v in values]
    present = [(i, v) for i, v in enumerate(values) if v is not None]
    if direction is None or len(present) < 2 or len({texts[i] for i, _ in present}) == 1:
        return texts
    key = {"min": lambda v: v, "max": lambda v: -v, "target": lambda v: abs(v - target)}[direction]
    best = min(present, key=lambda item: key(item[1]))
    best_text = texts[best[0]]
    best_key = key(best[1])
    out = []
    for i, text in enumerate(texts):
        tie = values[i] is not None and (text == best_text or key(values[i]) == best_key)
        out.append(r"\best{" + text + "}" if tie else text)
    return out


# ----------------------------------------------------------------------------- evidence loading
def load_diagnostics():
    table = defaultdict(dict)
    with open(register_input(E / "diagnostics.csv")) as handle:
        for row in csv.DictReader(handle):
            key = (row["dataset"], row["role"], int(row["replication"]))
            if row["metric"] in table[key]:
                raise ValueError(f"Duplicate diagnostic {key} {row['metric']}")
            table[key][row["metric"]] = float(row["value"])
    return table


DIAG = None


def diag(dataset, metric, roles=ROLES):
    """Exactly five distinct replications per requested role; no fallback keys."""
    result = {}
    for role in roles:
        values = []
        for rep in REPS:
            row = DIAG.get((dataset, role, rep))
            if row is None or metric not in row:
                raise KeyError(f"Missing {dataset}/{role}/rep{rep}: {metric}")
            values.append(row[metric])
        result[role] = values
    return result


def plot_rows(figure, panel, series, dataset, metric, by_role, transform="none"):
    for role, values in by_role.items():
        for rep, value in zip(REPS, values):
            PLOT_ROWS[figure].append({"figure": figure, "panel": panel, "series": series, "dataset": dataset,
                                      "role": role, "replication": rep, "metric_key": metric,
                                      "transformation": transform, "value": repr(float(value))})


def verified_analysis():
    from iwm_replication.five_seed_analysis import analyze_campaign
    rows = load_json(E / "model_results.json")
    contract = load_json(E / "frozen_analysis.json")
    exported = load_json(E / "analysis.json")
    recomputed = analyze_campaign(rows, contract)

    def same(a, b, path="analysis"):
        if isinstance(a, dict):
            if set(a) != set(b):
                raise ValueError(f"Key mismatch at {path}")
            for k in a:
                same(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, list):
            if len(a) != len(b):
                raise ValueError(f"Length mismatch at {path}")
            for i, (u, v) in enumerate(zip(a, b)):
                same(u, v, f"{path}[{i}]")
        elif isinstance(a, float) and isinstance(b, (int, float)):
            if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(f"Value mismatch at {path}: {a} vs {b}")
        elif a != b:
            raise ValueError(f"Mismatch at {path}: {a} vs {b}")

    same(recomputed, exported)  # floating-point roundoff tolerance only
    if recomputed["counts"] != {"complete": 40, "failed": 0, "missing": 0, "incomplete-heads": 0}:
        raise ValueError("Expected 40 complete outcomes")
    key = {"moving_mnist": "forecasts/model_prior/strata/all/energy_score",
           "mpi3d": "forecasts/metrics/physical/prior/energy_score_euclidean/mean"}
    for row in rows:
        diagnostic = DIAG[(row["dataset"], row["role"], row["replication"])][key[row["dataset"]]]
        if diagnostic != row["metrics"]["physical_forecast_energy_score"]:
            raise ValueError("Primary result differs from its diagnostic record")
    checkpoints = [row["checkpoint_sha256"] for row in rows]
    if len(set(checkpoints)) != 40:
        raise ValueError("Checkpoint records are not distinct")
    return recomputed


# ----------------------------------------------------------------------------- plotting helpers
def role_axis(ax, roles=ROLES, labels=None, two_line=True):
    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(labels or [(NAME2 if two_line else NAME)[r] for r in roles],
                       fontsize=6.3 if two_line else 7.0)
    ax.set_xlim(-0.6, len(roles) - 0.4)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def scatter_role(ax, x, values, role, size=16, zorder=3, label=None):
    xs = [x + REP_OFFSET[r] for r in REPS]
    face = COLOR[role] if FILLED[role] else "white"
    ax.scatter(xs, values, s=size, marker=MARKER[role], facecolors=face, edgecolors=COLOR[role],
               linewidths=1.0, zorder=zorder, label=label)


def mean_bar(ax, x, values, log=False):
    m, sd = mean_sd(values)
    xm = x + 0.30
    if log:
        ax.plot([xm - 0.06, xm + 0.06], [m, m], color=INK, linewidth=1.1, zorder=4)
        lo, hi = max(m - sd, min(values) * 0.5), m + sd
        ax.plot([xm, xm], [lo, hi], color=INK, linewidth=0.8, zorder=4)
    else:
        ax.errorbar([xm], [m], yerr=[sd], fmt="_", color=INK, markersize=7, elinewidth=0.8, capsize=0, zorder=4)
    return m, sd


def hline(ax, y, text, style="--", color=MUTED, where="right", va="bottom", fontsize=6.5, x_text=None):
    ax.axhline(y, linestyle=style, color=color, linewidth=0.7, zorder=1)
    x0, x1 = ax.get_xlim()
    xt = x_text if x_text is not None else (x1 - 0.05 if where == "right" else x0 + 0.05)
    ax.text(xt, y, text, ha="right" if where == "right" else "left", va=va, fontsize=fontsize, color=MUTED)


def save(fig, stem, **info):
    for suffix in (".pdf", ".png"):
        fig.savefig(FIGURES / f"{stem}{suffix}", bbox_inches="tight", pad_inches=0.02,
                    dpi=300 if suffix == ".png" else None)
    plt.close(fig)
    record(f"figures/{stem}.pdf", **info, png=f"figures/{stem}.png", data=f"data/{stem}.csv")


def role_legend(ax, roles=ROLES, loc="upper right", **kwargs):
    handles = [Line2D([], [], linestyle="", marker=MARKER[r], markersize=5,
                      markerfacecolor=COLOR[r] if FILLED[r] else "white", markeredgecolor=COLOR[r],
                      markeredgewidth=1.0, label=NAME[r]) for r in roles]
    return ax.legend(handles=handles, loc=loc, frameon=False, handletextpad=0.2, **kwargs)


def label_points(fig, ax, points, fontsize=5.5):
    """Write short labels (replication IDs) next to points without overlaps.

    points: (x, y, text) in data coordinates. Call after the layout and limits are final. Greedy:
    each label takes the first candidate offset whose box overlaps no placed label and no marker.
    """
    fig.canvas.draw()
    scale = fig.dpi / 72.0
    marks = [ax.transData.transform((x, y)) for x, y, _ in points]
    radius, height, width_char, pad = 2.6 * scale, 4.6 * scale, 3.0 * scale, 0.8 * scale
    candidates = [(2.5, 1.5), (2.5, -5.5), (-5.5, 1.5), (-5.5, -5.5), (2.5, 6.5), (-5.5, 6.5),
                  (6.0, -2.0), (-9.0, -2.0), (2.5, -10.0), (-5.5, -10.0)]

    def overlap(a, b):
        return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))

    placed = []
    for (x, y, text), (px, py) in zip(points, marks):
        width = width_char * len(text)
        best = None
        for dx, dy in candidates:
            box = (px + dx * scale - pad, py + dy * scale - pad, px + dx * scale + width + pad, py + dy * scale + height + pad)
            cost = sum(overlap(box, b) for b in placed)
            cost += sum(overlap(box, (mx - radius, my - radius, mx + radius, my + radius)) for mx, my in marks)
            if best is None or cost < best[0]:
                best = (cost, dx, dy, box)
            if cost == 0:
                break
        placed.append(best[3])
        ax.annotate(text, (x, y), xytext=(best[1], best[2]), textcoords="offset points", fontsize=fontsize, color=MUTED)


# ----------------------------------------------------------------------------- tables: design
def table_role_matrix(mm_cfg, mpi_cfg):
    from iwm_replication.moving_mnist_campaign import MODEL_OPTIONS
    beta = MODEL_OPTIONS["R1"]["beta"]
    assert beta == MODEL_OPTIONS["S1"]["beta"] == mpi_cfg["loss"]["beta"] == 0.001
    lam = MODEL_OPTIONS["S0"]["sigreg_weight"]
    assert lam == MODEL_OPTIONS["S1"]["sigreg_weight"] == mpi_cfg["sigreg"]["weight"] == 0.1
    ema_mm = MODEL_OPTIONS["R0"]["ema_decay"]
    ema_start, ema_end = mpi_cfg["optim"]["ema_decay_start"], mpi_cfg["optim"]["ema_decay_end"]
    ema = rf"{ema_mm:g} / {ema_start:g}$\to${ema_end:g}"
    rows = [
        ("R0", r"EMA copy, stop-gradient", "none", r"$\mathcal{L}_{\mathrm{pred}}$", "--", "--", "2", ema),
        ("R1", r"EMA copy, stop-gradient", r"Gaussian", r"$\mathcal{L}_{\mathrm{pred}}+\beta\mathcal{L}_{\mathrm{KL}}$",
         f"{beta:g}", "--", "2", ema),
        ("S0", r"same encoder, gradients", "none", r"$\mathcal{L}_{\mathrm{pred}}+\lambda\mathcal{L}_{\mathrm{SIG}}$",
         "--", f"{lam:g}", "1", "none"),
        ("S1", r"same encoder, gradients", r"Gaussian",
         r"$\mathcal{L}_{\mathrm{pred}}+\beta\mathcal{L}_{\mathrm{KL}}+\lambda\mathcal{L}_{\mathrm{SIG}}$",
         f"{beta:g}", f"{lam:g}", "1", "none"),
    ]
    lines = [r"\begin{tabular}{@{}llll@{}}", r"\toprule",
             r"Model & Target branch (encoders) & Residual $r$ & Additional losses \\",
             r"\midrule"]
    for row in rows:
        target = "EMA, stopped (2)" if row[0] in ("R0", "R1") else "Shared, gradients (1)"
        added = "--" if row[0] == "R0" else row[3].replace(r"\mathcal{L}_{\mathrm{pred}}+", "")
        lines.append(" & ".join((TEX[row[0]], target, row[2], added)) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "role_matrix.tex").write_text("\n".join(lines) + "\n")
    record("tables/role_matrix.tex", source=["config/campaigns/five_seed_v1/mpi3d.yaml",
                                              "src/iwm_replication/moving_mnist_campaign.py:MODEL_OPTIONS"],
           names=NAME)


def split_counts():
    from iwm_replication.mpi3d_data import (MPI3DCleanImageDataset, MPI3DDeterministicTransitionDataset,
                                            MPI3DRepeatedFutureDataset, load_mpi3d_position_manifest,
                                            mpi3d_attribute_configurations, _all_mpi3d_common_legal_positions)
    manifest_dir = ROOT / "config/shared/mpi3d_position_manifests"
    positions = {name: load_mpi3d_position_manifest(str(register_input(manifest_dir / f"{name}.json")), expected_name=name)
                 for name in ("probe_train", "validation", "test")}
    counts = {"attributes": {s: len(mpi3d_attribute_configurations(s)) for s in ("train", "validation", "test")},
              "positions": {k: len(v) for k, v in positions.items()},
              "legal_positions": len(_all_mpi3d_common_legal_positions()),
              "train_pairs": len(MPI3DDeterministicTransitionDataset(None, split="train"))}
    for split in ("validation", "test"):
        counts[f"{split}_queries"] = len(MPI3DRepeatedFutureDataset(None, split=split, condition="S", positions=positions[split]))
    for split, name in (("train", "probe_train"), ("validation", "validation"), ("test", "test")):
        counts[f"{split}_clean"] = len(MPI3DCleanImageDataset(None, split=split, positions=positions[name]))
    counts["readout_population"] = counts["attributes"]["train"] * 3 * 40 * 40
    identity = json.loads(Path(register_input(ROOT / "assets/mnist_identity_manifest.json")).read_text())
    counts["mnist"] = {k: len(v) for k, v in identity["splits"].items()}
    counts["mnist_sha256"] = identity["sha256"]
    return counts


def table_splits(counts, protocol, mpi_cfg):
    mm, mp = protocol["final_banks"]["moving_mnist"]["settings"], protocol["final_banks"]["mpi3d"]["settings"]
    if protocol["final_banks"]["moving_mnist"]["data_fingerprints"]["identity_manifest_sha256"] != counts["mnist_sha256"]:
        raise ValueError("Local identity manifest differs from the frozen one")
    epochs, per_epoch, batch = mpi_cfg["train"]["epochs"], mpi_cfg["train"]["transition_samples_per_epoch"], mpi_cfg["data"]["batch_size"]
    c = counts
    dev = c["mnist"]["development"]
    mm_rows = [
        ("Image source", "MNIST, 28$\\times$28 digits"),
        ("Identity partition", f"{c['mnist']['train']:,} train / {dev:,} development ({dev // 2:,} selection, {dev - dev // 2:,} report) / {c['mnist']['test']:,} official test"),
        ("Training examples", "fresh online clips; 75{,}000 updates $\\times$ 128 = 9.6M presentations"),
        ("Observed $x$ / target $y$", "frames 1--3 / frames 4--6"),
        ("Posterior input $u$ (training)", "frames 7--9 of the same trajectory"),
        ("Readout and probe fit", f"{mm['sizes']['fit']:,} clips, train identities"),
        ("Readout and probe selection", f"{mm['sizes']['selection']:,} clips, development selection identities"),
        ("Final report / forecast queries", f"{mm['sizes']['report']:,} / {mm['sizes']['queries']:,} clips, official test identities"),
        ("Draws per query", f"{mm['samples']} forecast, {mm['futures']} physical truth"),
    ]
    mp_rows = [
        ("Image source", "MPI3D-realistic, 64$\\times$64, background fixed"),
        ("Attribute split", f"{c['attributes']['train']} / {c['attributes']['validation']} / {c['attributes']['test']} colour--shape--size combinations"),
        ("Training population", f"{c['train_pairs']:,} source/command pairs ({c['attributes']['train']} $\\times$ 3 cameras $\\times$ {c['legal_positions']:,} positions $\\times$ 4)"),
        ("Training examples", f"{epochs} epochs $\\times$ {per_epoch:,} = {epochs * per_epoch / 1e6:.2f}M presentations, {epochs * per_epoch // batch:,} updates"),
        ("Observed $x,a$ / target $y$", "source image and unit command / realized next image"),
        ("Posterior input $u$ (training)", "the realized target $y$ itself"),
        ("Readout fit / select / hold-out", f"{mp['readout_counts'][0]:,} / {mp['readout_counts'][1]:,} / {mp['readout_counts'][2]:,} train-attribute images (of {c['readout_population']:,})"),
        ("Probe fit / select / report", f"{mp['probe_counts']['fit']:,} train / {mp['probe_counts']['selection']:,} validation / {mp['probe_counts']['report']:,} test images"),
        ("Forecast queries", f"{mp['queries']:,} = {mp['queries'] // 4} test sources $\\times$ 4 commands (of {c['test_queries']:,})"),
        ("Truth and forecast support", f"exact two outcomes; {mp['quantiles']} equal-weight prior quantiles"),
    ]
    lines = [r"\begin{tabular}{@{}p{0.29\linewidth}p{0.66\linewidth}@{}}", r"\toprule",
             r"\multicolumn{2}{@{}l}{\textbf{Moving-MNIST}} \\", r"\midrule"]
    lines += [f"{a} & {b} \\\\" for a, b in mm_rows]
    lines += [r"\midrule", r"\multicolumn{2}{@{}l}{\textbf{MPI3D-S (balanced stochastic execution)}} \\", r"\midrule"]
    lines += [f"{a} & {b} \\\\" for a, b in mp_rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "dataset_splits.tex").write_text("\n".join(lines) + "\n")
    overview = [
        ("Observed input", "frames 1--3", "source image and unit command $a$"),
        ("Target $y$", "frames 4--6", "realized next image"),
        ("Posterior input $u$ (training only)", "frames 7--9", "the target $y$ itself"),
        ("Why the future is one-to-many", "a random axis changes velocity, by a random amount",
         "the command fails or succeeds, probability $\\tfrac12$ each"),
        ("Held-out evaluation population", f"{c['mnist']['test']:,} official test digits",
         f"{c['attributes']['test']} unseen colour--shape--size combinations"),
        ("Training budget", f"75{{,}}000 updates at batch 128 (fresh clips)",
         f"{epochs * per_epoch // batch:,} updates at batch {batch} ({epochs} epochs)"),
        ("Encoder $\\to$ projector", "3D CNN (768-d) $\\to$ 128-d", "ViT (256-d) $\\to$ 128-d"),
        ("Final forecast queries", f"{mm['sizes']['queries']:,}; {mm['samples']} prior vs.\\ {mm['futures']} truth draws",
         f"{mp['queries']:,} ({mp['queries'] // 4} sources $\\times$ 4 commands); {mp['quantiles']} prior atoms vs.\\ exact outcomes"),
        ("Physical quantity (readout target)", "new velocity (px/frame)", "object position (grid indices)"),
    ]
    rows_tex = [f"{a} & {b} & {m} \\\\" for a, b, m in overview]
    table = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.27\linewidth}>{\raggedright\arraybackslash}p{0.31\linewidth}>{\raggedright\arraybackslash}p{0.36\linewidth}@{}}",
             r"\toprule", r" & Moving-MNIST & MPI3D-S \\", r"\midrule", *rows_tex, r"\bottomrule", r"\end{tabular}"]
    (TABLES / "setup_overview.tex").write_text("\n".join(table).replace(",", "{,}").replace("{{,}}", "{,}") + "\n")
    record("tables/setup_overview.tex", source=["frozen_protocol.json:final_banks", "config/campaigns/five_seed_v1/*.yaml"])
    record("tables/dataset_splits.tex", counts=c, source=["frozen_protocol.json:final_banks", "src/iwm_replication/mpi3d_data.py",
           "data/concept2-mnist/concept2-identities-v1-seed0.json"])
    num("MMTrainIds", f"{c['mnist']['train']:,}".replace(",", "{,}"))
    num("MPTrainPairs", f"{c['train_pairs']:,}".replace(",", "{,}"))
    num("MPTestQueries", f"{c['test_queries']:,}".replace(",", "{,}"))
    num("MPTestSources", f"{c['test_queries'] // 4:,}".replace(",", "{,}"))
    num("MPReadoutPopulation", f"{c['readout_population']:,}".replace(",", "{,}"))


def table_training(mm_cfg, mpi_cfg, recount):
    params = {(r["dataset"], r["role"]): r for r in load_json(COST / "benchmark_local_cpu_flops.json")["rows"]}
    t = mm_cfg["training"]
    o, tr = mpi_cfg["optim"], mpi_cfg["train"]
    per_epoch = tr["transition_samples_per_epoch"] // mpi_cfg["data"]["batch_size"]
    rows = [
        ("Encoder; projector", "3D CNN, 5 layers, 768-d pooled; 768$\\to$1024$\\to$1024$\\to$128",
         f"ViT (patch 8, width {mpi_cfg['model']['vit_dim']}, depth {mpi_cfg['model']['vit_depth']}, {mpi_cfg['model']['vit_heads']} heads), mean-pooled; 256$\\to$1024$\\to$1024$\\to$128"),
        ("Predictor input; residual", "$z_x$ or $[z_x, r]$; $d_r{=}2$, softplus std", "$[z_x, a, r]$ ($r{=}0$ for Det models); $d_r{=}1$, clamped log-std"),
        ("Stored parameters (M), BYOL reference / AdaSSL reference / Shared-Det / Shared-Var", " / ".join(f"{params[('moving_mnist', r)]['stored_parameters'] / 1e6:.2f}" for r in ROLES),
         " / ".join(f"{params[('mpi3d', r)]['stored_parameters'] / 1e6:.2f}" for r in ROLES)),
        ("Batch; updates", f"{t['batch_size']}; {t['total_steps']:,}", f"{mpi_cfg['data']['batch_size']} (4 GPUs); {tr['epochs'] * per_epoch:,}"),
        ("AdamW learning rate", f"{t['learning_rate']:g}, cosine to 0", f"{o['lr']:g}, {tr['warmup_epochs']}-epoch warm-up, cosine over {o['learning_rate_schedule_scale']:g}$\\times$ budget"),
        ("Weight decay", f"{t['weight_decay']:g}, all parameters", f"{o['weight_decay_start']:g}$\\to${o['weight_decay_end']:g}; no decay on norms and most biases (App.~\\ref{{app:optim}})"),
        ("EMA momentum (EMA models)", f"{mm_cfg['models']['R0']['ema_decay']:g}, constant", f"{o['ema_decay_start']:g}$\\to${o['ema_decay_end']:g}, cosine"),
        ("$\\beta$ (Var models); $\\lambda$ (Shared models)", "0.001; 0.1 (128 directions, 17 knots, $[0.2,4]$)", "same"),
    ]
    lines = [r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.22\linewidth}>{\raggedright\arraybackslash}p{0.34\linewidth}>{\raggedright\arraybackslash}p{0.38\linewidth}@{}}", r"\toprule",
             r" & Moving-MNIST & MPI3D \\", r"\midrule"]
    lines += [f"{a} & {b} & {c} \\\\" for a, b, c in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "training_settings.tex").write_text("\n".join(lines) + "\n")
    record("tables/training_settings.tex", source=["config/campaigns/five_seed_v1/moving_mnist.yaml",
           "config/campaigns/five_seed_v1/mpi3d.yaml", "results/compute_cost/benchmark_local_cpu_flops.json"])


# ----------------------------------------------------------------------------- tables: results
def es_text(dataset, value, sd=None):
    if dataset == "moving_mnist":
        return fmt(value, 3) if sd is None else f"{fmt(value, 3)} $\\pm$ {fmt(sd, 3)}"
    if abs(value) >= 1e5:
        def sci(v):
            exponent = int(math.floor(math.log10(abs(v))))
            return f"{v / 10 ** exponent:.2f}$\\times$10$^{{{exponent}}}$"
        return sci(value) if sd is None else f"{sci(value)} $\\pm$ {sci(sd)}"
    if abs(value) >= 100:
        return f"{value:,.0f}".replace(",", "{,}") if sd is None else f"{value:,.0f} $\\pm$ {sd:,.0f}".replace(",", "{,}")
    return fmt(value, 2) if sd is None else f"{fmt(value, 2)} $\\pm$ {fmt(sd, 2)}"


def table_primary(analysis):
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Moving-MNIST ES " + DOWN + r" (px/frame)} & \multicolumn{2}{c}{MPI3D ES " + DOWN + r" (grid indices)} \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
             r"Model & mean $\pm$ SD & range & mean $\pm$ SD & range \\", r"\midrule"]
    body = {ds: analysis["datasets"][ds]["metrics"]["physical_forecast_energy_score"]["roles"] for ds in DATASETS}
    means = {}
    for ds in DATASETS:
        texts = bold_best([body[ds][r]["mean"] for r in ROLES], "min", lambda v, ds=ds: es_text(ds, v))
        for role, text in zip(ROLES, texts):
            sd = es_text(ds, body[ds][role]["sample_sd"])
            raw = es_text(ds, body[ds][role]["mean"])
            means[(ds, role)] = (r"\best{" + f"{raw} $\\pm$ {sd}" + "}" if text.startswith(r"\best{")
                                 else f"{raw} $\\pm$ {sd}")
    for role in ROLES:
        cells = [TEX[role]]
        for ds in DATASETS:
            rec = body[ds][role]
            cells += [means[(ds, role)], f"{es_text(ds, min(rec['values']))}--{es_text(ds, max(rec['values']))}"]
            num(f"{DS_WORD[ds]}{ROLE_WORD[role]}Mean", es_text(ds, rec["mean"]))
            num(f"{DS_WORD[ds]}{ROLE_WORD[role]}SD", es_text(ds, rec["sample_sd"]))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "primary_results.tex").write_text("\n".join(lines) + "\n")
    record("tables/primary_results.tex", source="analysis.json (recomputed and asserted equal)",
           metric="physical_forecast_energy_score", uncertainty="sample SD over five training replications",
           bold="lowest mean per dataset")


def diff_text(dataset, value):
    if dataset == "moving_mnist":
        return f"${value:+.3f}$".replace("+", "{+}").replace("-", "-") if False else f"${value:.3f}$"
    if abs(value) >= 100:
        return f"${value:,.0f}$".replace(",", "{,}")
    return f"${value:.2f}$"


def table_paired(analysis):
    lines = [r"\begin{tabular}{@{}llccc@{}}", r"\toprule",
             r"Dataset & Contrast & Mean diff.\ " + DOWN + r" & 95\% paired $t$ interval & Runs $<0$ \\",
             r"\midrule"]
    for ds in DATASETS:
        for name in ("S1-R1", "S1-S0"):
            rec = analysis["datasets"][ds]["metrics"]["physical_forecast_energy_score"]["contrasts"][name]
            lo, hi = rec["interval_95"]
            wins = sum(d < 0 for d in rec["paired_differences"])
            lines.append(f"{DSNAME[ds]} & {CONTRAST_TEX[name]} & {diff_text(ds, rec['mean'])} & [{diff_text(ds, lo)}, {diff_text(ds, hi)}] & {wins} of 5 \\\\")
            base = f"{DS_WORD[ds]}{'SOneROne' if name == 'S1-R1' else 'SOneSZero'}"
            num(base + "Diff", diff_text(ds, rec["mean"]).strip("$"))
            num(base + "Lo", diff_text(ds, lo).strip("$"))
            num(base + "Hi", diff_text(ds, hi).strip("$"))
            num(base + "Wins", str(wins))
        if ds == "moving_mnist":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "paired_effects.tex").write_text("\n".join(lines) + "\n")
    record("tables/paired_effects.tex", source="analysis.json (recomputed)", interval="paired Student t, df=4, individual 95%")


def table_per_replication(analysis):
    lines = [r"\begin{tabular}{@{}llrrrrr@{}}", r"\toprule",
             r"Dataset & Model (ES " + DOWN + r") & Rep.~1 & Rep.~2 & Rep.~3 & Rep.~4 & Rep.~5 \\", r"\midrule"]
    for ds in DATASETS:
        body = analysis["datasets"][ds]["metrics"]["physical_forecast_energy_score"]
        columns = [bold_best([body["roles"][r]["values"][k] for r in ROLES], "min", lambda v, ds=ds: es_text(ds, v))
                   for k in range(5)]
        for i, role in enumerate(ROLES):
            lines.append(f"{DSNAME[ds]} & {TEX[role]} & " + " & ".join(col[i] for col in columns) + r" \\")
        for name in ("S1-R1", "S1-S0"):
            vals = body["contrasts"][name]["paired_differences"]
            lines.append(f"{DSNAME[ds]} & {CONTRAST_TEX[name]} & " + " & ".join(diff_text(ds, v) for v in vals) + r" \\")
        if ds == "moving_mnist":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "per_replication.tex").write_text("\n".join(lines) + "\n")
    record("tables/per_replication.tex", source="analysis.json (recomputed)", bold="lowest ES per replication and dataset")
    s1 = analysis["datasets"]["moving_mnist"]["metrics"]["physical_forecast_energy_score"]["roles"]["S1"]["values"]
    num("MMSOneWeak", fmt(s1[0], 3))
    num("MMSOneOthersLo", fmt(min(s1[1:]), 3))
    num("MMSOneOthersHi", fmt(max(s1[1:]), 3))
    so = analysis["datasets"]["moving_mnist"]["metrics"]["physical_forecast_energy_score"]["roles"]["S0"]["mean"]
    num("MMSOneVsSZeroPct", f"{100 * (1 - st.mean(s1) / so):.0f}")


def mm_selected(role, rep):
    row = DIAG[("moving_mnist", role, rep)]
    raw, unit = row["readout_selection/candidates/raw/selection_mse"], row["readout_selection/candidates/unit/selection_mse"]
    return "raw" if raw <= unit else "unit"


def mp_selected(role, rep):
    row = DIAG[("mpi3d", role, rep)]
    raw, unit = row["readout_selection/candidates/raw/selection_mse"], row["readout_selection/candidates/unit/selection_mse"]
    return "raw" if raw <= unit else "unit"


def ms(values, digits=3):
    m, sd = mean_sd(values)
    return f"{m:.{digits}f} $\\pm$ {sd:.{digits}f}"


def ms_best(per_role, direction, digits=3, target=None):
    """Cells 'mean ± SD' with the best mean in bold; per_role maps role -> values or None."""
    means = [st.mean(per_role[r]) if per_role.get(r) is not None else None for r in ROLES]
    heads = bold_best(means, direction, lambda v: f"{v:.{digits}f}", target=target)
    cells = []
    for role, head in zip(ROLES, heads):
        values = per_role.get(role)
        if values is None:
            cells.append("--")
            continue
        sd = f"{st.stdev(values):.{digits}f}"
        raw = f"{st.mean(values):.{digits}f}"
        cells.append(r"\best{" + f"{raw} $\\pm$ {sd}" + "}" if head.startswith(r"\best{") else f"{raw} $\\pm$ {sd}")
    return cells


def table_mm_controls():
    keys = [("Sampled prior (primary)", "forecasts/model_prior/strata/all/energy_score"),
            ("Fixed prior mean", "forecasts/model_fixed_prior_mean/strata/all/energy_score"),
            ("Target-space persistence", "forecasts/target_space_persistence/strata/all/energy_score"),
            ("Encode--decode oracle", "forecasts/encoded_decoded_oracle/strata/all/energy_score")]
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Forecast (physical ES " + DOWN + r", px/frame) & " + " & ".join(TEX[r] for r in ROLES) + r" \\", r"\midrule"]
    for label, key in keys:
        per_role = {}
        for role in ROLES:
            try:
                per_role[role] = diag("moving_mnist", key, (role,))[role]
            except KeyError:
                per_role[role] = None
        lines.append(f"{label} & " + " & ".join(ms_best(per_role, "min" if "primary" in label else None)) + r" \\")
    shared = {}
    for label, key in (("Pixel constant velocity", "forecasts/pixel_constant_velocity/strata/all/energy_score"),
                       ("Simulator oracle (privileged)", "forecasts/independent_privileged_oracle/strata/all/energy_score")):
        vals = diag("moving_mnist", key)
        flat = {v for r in ROLES for v in vals[r]}
        if len(flat) != 1:
            raise ValueError(f"Shared baseline {key} is not a single fixed-bank value")
        shared[key] = flat.pop()
        lines.append(f"{label} & \\multicolumn{{4}}{{c}}{{{fmt(shared[key], 3)} (one fixed-bank calculation)}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "mm_controls.tex").write_text("\n".join(lines) + "\n")
    record("tables/mm_controls.tex", keys=[k for _, k in keys], uncertainty="mean ± sample SD over five replications",
           bold="lowest mean in the primary row only")
    num("MMPixelCV", fmt(shared["forecasts/pixel_constant_velocity/strata/all/energy_score"], 3))
    num("MMOracle", fmt(shared["forecasts/independent_privileged_oracle/strata/all/energy_score"], 3))
    for role in ("R1", "S1"):
        d = diag("moving_mnist", "diagnostics/delta_fixed_es", (role,))[role]
        num(f"MM{ROLE_WORD[role]}SamplingGain", fmt(st.mean(d), 3))
        num(f"MM{ROLE_WORD[role]}SamplingGainMin", fmt(min(d), 3))
        num(f"MM{ROLE_WORD[role]}SamplingWins", str(sum(v > 0 for v in d)))
        fx = diag("moving_mnist", "forecasts/model_fixed_prior_mean/strata/all/energy_score", (role,))[role]
        num(f"MM{ROLE_WORD[role]}FixedMean", fmt(st.mean(fx), 3))
    for role in ROLES:
        o = diag("moving_mnist", "forecasts/encoded_decoded_oracle/strata/all/energy_score", (role,))[role]
        num(f"MM{ROLE_WORD[role]}OracleED", fmt(st.mean(o), 3))
        p = diag("moving_mnist", "forecasts/target_space_persistence/strata/all/energy_score", (role,))[role]
        num(f"MM{ROLE_WORD[role]}Persist", fmt(st.mean(p), 3))
        primary = diag("moving_mnist", "forecasts/model_prior/strata/all/energy_score", (role,))[role]
        num(f"MM{ROLE_WORD[role]}BeatsPixelCV", str(sum(v < shared["forecasts/pixel_constant_velocity/strata/all/energy_score"]
                                                        for v in primary)))
        num(f"MM{ROLE_WORD[role]}BeatsPersist", str(sum(a < b for a, b in zip(primary, p))))
    s0 = st.mean(diag("moving_mnist", "forecasts/encoded_decoded_oracle/strata/all/energy_score", ("S0",))["S0"])
    s1 = st.mean(diag("moving_mnist", "forecasts/encoded_decoded_oracle/strata/all/energy_score", ("S1",))["S1"])
    gap = (st.mean(diag("moving_mnist", "forecasts/model_prior/strata/all/energy_score", ("S0",))["S0"])
           - st.mean(diag("moving_mnist", "forecasts/model_prior/strata/all/energy_score", ("S1",))["S1"]))
    num("MMOracleGapSZeroSOne", fmt(s0 - s1, 3))
    num("MMOracleGapShare", f"{100 * (s0 - s1) / gap:.0f}")


def table_mm_representation():
    keys = [("Source backbone, digit accuracy", "probes/source/online_backbone/digit/accuracy", "max"),
            ("Source projector, digit accuracy", "probes/source/online_projector/digit/accuracy", "max"),
            ("Source backbone, velocity $R^2$ ($x$ / $y$)", "probes/source/online_backbone/velocity/r2_xy/", "max"),
            ("Source projector, velocity $R^2$ ($x$ / $y$)", "probes/source/online_projector/velocity/r2_xy/", "max"),
            ("Forecast readout, velocity $R^2$ ($x$ / $y$)", "forecast_readout/r2_xy/", "max"),
            ("Encode--decode velocity RMSE (px/frame)", "diagnostics/oracle_decode_vector_rmse", "min")]
    arrow = {"max": UP, "min": DOWN}
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
             r"Measurement & " + " & ".join(TEX[r] for r in ROLES) + r" \\", r"\midrule"]
    for label, key, direction in keys:
        if key.endswith("/"):
            axes = [bold_best([st.mean(diag("moving_mnist", key + str(i), (r,))[r]) for r in ROLES], direction,
                              lambda v: f"{v:.3f}") for i in (0, 1)]
            cells = [f"{axes[0][j]} / {axes[1][j]}" for j in range(len(ROLES))]
        else:
            cells = bold_best([st.mean(diag("moving_mnist", key, (r,))[r]) for r in ROLES], direction, lambda v: f"{v:.3f}")
        lines.append(f"{label} {arrow[direction]} & " + " & ".join(cells) + r" \\")
    support = {}
    for role in ROLES:
        per_run = []
        for rep in REPS:
            row = DIAG[("moving_mnist", role, rep)]
            values = [v for k, v in row.items() if k.startswith("diagnostics/predicted_feature_support/")
                      and k.endswith("/fraction_above_train_p99")]
            if len(values) != 1024:
                raise ValueError("Expected one support record per Moving-MNIST query")
            per_run.append(st.mean(values))
        support[role] = per_run
        PLOT_ROWS["mm_predicted_support"] += [{"figure": "mm_predicted_support", "panel": "", "series": "fraction_above_fit_p99",
                                               "dataset": "moving_mnist", "role": role, "replication": k,
                                               "metric_key": "mean over queries of diagnostics/predicted_feature_support/<q>/fraction_above_train_p99",
                                               "transformation": "mean over 1,024 queries", "value": repr(v)} for k, v in zip(REPS, per_run)]
        num(f"MM{ROLE_WORD[role]}FracAbove", fmt(st.mean(per_run), 2))
    lines.append(f"Predicted features above fit 99th percentile (fraction) {DOWN} & " +
                 " & ".join(bold_best([st.mean(support[r]) for r in ROLES], "min", lambda v: f"{v:.3f}")) + r" \\")
    sel = {role: "/".join(sorted({mm_selected(role, r) for r in REPS})) for role in ROLES}
    lines.append(f"Selected readout transform {NODIR} & " + " & ".join(sel[r] for r in ROLES) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "mm_representation.tex").write_text("\n".join(lines) + "\n")
    record("tables/mm_representation.tex", keys=[k for _, k, _ in keys], aggregation="mean over five replications",
           bold="best mean per row (per axis for x/y cells)")
    for role in ROLES:
        for i, ax in ((0, "X"), (1, "Y")):
            num(f"MM{ROLE_WORD[role]}ReadoutRtwo{ax}", fmt(st.mean(diag("moving_mnist", f"forecast_readout/r2_xy/{i}", (role,))[role]), 3))
            num(f"MM{ROLE_WORD[role]}ProjVelRtwo{ax}", fmt(st.mean(diag("moving_mnist", f"probes/source/online_projector/velocity/r2_xy/{i}", (role,))[role]), 3))
        num(f"MM{ROLE_WORD[role]}ProjDigit", fmt(st.mean(diag("moving_mnist", "probes/source/online_projector/digit/accuracy", (role,))[role]), 3))
        num(f"MM{ROLE_WORD[role]}BackDigit", fmt(st.mean(diag("moving_mnist", "probes/source/online_backbone/digit/accuracy", (role,))[role]), 3))
        num(f"MM{ROLE_WORD[role]}DecodeRMSE", fmt(st.mean(diag("moving_mnist", "diagnostics/oracle_decode_vector_rmse", (role,))[role]), 3))
    s1x = diag("moving_mnist", "forecast_readout/r2_xy/0", ("S1",))["S1"]
    num("MMSOneReadoutRtwoXWeak", fmt(s1x[0], 3))
    s1d = diag("moving_mnist", "probes/source/online_projector/digit/accuracy", ("S1",))["S1"]
    num("MMSOneProjDigitMin", fmt(min(s1d), 3))


def table_mm_residual():
    keys = [("Posterior--prior KL (nats)", "latent/gaussian_diagnostics/mean_training_information_kl", 2, None),
            ("Mean prior std", "latent/gaussian_diagnostics/prior_std_mean", 2, None),
            ("Mean posterior std", "latent/gaussian_diagnostics/posterior_std_mean", 2, None),
            ("90\\% interval coverage, $x$", "diagnostics/marginal_90_interval/coverage_xy/0", 3, 0.9),
            ("90\\% interval coverage, $y$", "diagnostics/marginal_90_interval/coverage_xy/1", 3, 0.9),
            ("90\\% interval width, $x$ (px/frame)", "diagnostics/marginal_90_interval/mean_width_xy/0", 2, None),
            ("90\\% interval width, $y$ (px/frame)", "diagnostics/marginal_90_interval/mean_width_xy/1", 2, None)]
    lines = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"Endpoint diagnostic & \emavar{} & \shvar{} \\", r"\midrule"]
    for label, key, d, target in keys:
        vals = diag("moving_mnist", key, ("R1", "S1"))
        per_role = {"R1": vals["R1"], "S1": vals["S1"]}
        cells = ms_best(per_role, "target" if target else None, d, target=target)
        cells = [c for r, c in zip(ROLES, cells) if r in ("R1", "S1")]
        tag = r"($\to$0.90)" if target else NODIR
        lines.append(f"{label} {tag} & {cells[0]} & {cells[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "mm_residual.tex").write_text("\n".join(lines) + "\n")
    record("tables/mm_residual.tex", keys=[k for _, k, _, _ in keys], bold="coverage closer to nominal 0.90")
    for role in ("R1", "S1"):
        w = ROLE_WORD[role]
        num(f"MM{w}KL", fmt(st.mean(diag("moving_mnist", "latent/gaussian_diagnostics/mean_training_information_kl", (role,))[role]), 2))
        num(f"MM{w}PriorStd", fmt(st.mean(diag("moving_mnist", "latent/gaussian_diagnostics/prior_std_mean", (role,))[role]), 2))
        num(f"MM{w}PostStd", fmt(st.mean(diag("moving_mnist", "latent/gaussian_diagnostics/posterior_std_mean", (role,))[role]), 2))
        for i, ax in ((0, "X"), (1, "Y")):
            num(f"MM{w}Cover{ax}", fmt(st.mean(diag("moving_mnist", f"diagnostics/marginal_90_interval/coverage_xy/{i}", (role,))[role]), 2))


def table_mp_controls():
    P = "forecasts/metrics/"
    phys = [("Sampled prior (primary)", "physical/prior"), ("Fixed prior mean", "physical/fixed_prior_mean"),
            ("Target-space persistence", "physical/persistence"), ("Wrong command", "physical/wrong_action"),
            ("Wrong source", "physical/wrong_source"), ("Encode--decode oracle", "physical/encode_decode_oracle")]
    lat = [("Sampled prior", "latent/prior"), ("Fixed prior mean", "latent/fixed_prior_mean"),
           ("Target-space persistence", "latent/persistence"), ("Wrong command", "latent/wrong_action"),
           ("Wrong source", "latent/wrong_source"), ("Encode--decode oracle", "latent/encode_decode_oracle")]
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r"Forecast & " + " & ".join(TEX[r] for r in ROLES) + r" \\", r"\midrule",
             r"\multicolumn{5}{@{}l}{\emph{Physical Energy Score " + DOWN + r" (Euclidean grid indices)}} \\"]
    for label, key in phys:
        vals = diag("mpi3d", P + key + "/energy_score_euclidean/mean")
        cells = bold_best([None if ("fixed_prior_mean" in key and r in ("R0", "S0")) else st.mean(vals[r]) for r in ROLES],
                          "min" if key == "physical/prior" else None, lambda v: es_text("mpi3d", v))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    for label, key in (("Coordinate persistence (privileged)", "physical/coordinate_persistence"),
                       ("Commanded move (privileged)", "physical/command_success"),
                       ("Two-outcome oracle (privileged)", "physical/physical_oracle")):
        vals = diag("mpi3d", P + key + "/energy_score_euclidean/mean")
        flat = {v for r in ROLES for v in vals[r]}
        if len(flat) != 1:
            raise ValueError("Privileged MPI3D reference is not constant")
        lines.append(f"{label} & \\multicolumn{{4}}{{c}}{{{fmt(flat.pop(), 3)}}} \\\\")
    lines += [r"\midrule", r"\multicolumn{5}{@{}l}{\emph{Gap-normalized latent Energy Score " + DOWN + r" (within each model's own target space)}} \\"]
    for label, key in lat:
        vals = diag("mpi3d", P + key + "/normalized_energy_score/mean")
        cells = bold_best([None if ("fixed_prior_mean" in key and r in ("R0", "S0")) else st.mean(vals[r]) for r in ROLES],
                          "min" if key == "latent/prior" else None, lambda v: f"{v:.3f}")
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "mpi3d_controls.tex").write_text("\n".join(lines) + "\n")
    record("tables/mpi3d_controls.tex", keys=[k for _, k in phys + lat], aggregation="mean over five replications",
           bold="lowest mean in the two sampled-prior rows only")
    for role in ROLES:
        w = ROLE_WORD[role]
        v = diag("mpi3d", P + "latent/prior/normalized_energy_score/mean", (role,))[role]
        num(f"MP{w}LatentES", fmt(st.mean(v), 3))
        num(f"MP{w}LatentESMin", fmt(min(v), 3))
        num(f"MP{w}LatentESMax", fmt(max(v), 3))
        p = diag("mpi3d", P + "physical/persistence/energy_score_euclidean/mean", (role,))[role]
        num(f"MP{w}PhysPersist", fmt(st.mean(p), 2))
        o = diag("mpi3d", P + "physical/encode_decode_oracle/energy_score_euclidean/mean", (role,))[role]
        num(f"MP{w}OracleED", fmt(st.mean(o), 2))
        wa = diag("mpi3d", P + "physical/wrong_action/energy_score_euclidean/mean", (role,))[role]
        ws = diag("mpi3d", P + "physical/wrong_source/energy_score_euclidean/mean", (role,))[role]
        num(f"MP{w}WrongAction", es_text("mpi3d", st.mean(wa)))
        num(f"MP{w}WrongSource", es_text("mpi3d", st.mean(ws)))
        la = diag("mpi3d", P + "latent/wrong_action/normalized_energy_score/mean", (role,))[role]
        num(f"MP{w}LatentWrongAction", fmt(st.mean(la), 3))
        num(f"MP{w}LatentWrongActionWorse", str(sum(a > b for a, b in zip(la, v))))
    s1 = diag("mpi3d", P + "latent/prior/normalized_energy_score/mean", ("S1",))["S1"]
    num("MPSOneLatentESList", ", ".join(fmt(x, 3) for x in s1))
    per = diag("mpi3d", P + "latent/persistence/normalized_energy_score/mean")
    assert all(abs(x - 0.5) < 1e-9 for r in ROLES for x in per[r])
    lose = sum(a > b for r in ROLES for a, b in zip(diag("mpi3d", P + "latent/prior/normalized_energy_score/mean")[r], per[r]))
    num("MPLoseToPersistence", str(lose))
    valid = diag("mpi3d", P + "latent/prior/valid_gap/valid")
    invalid = diag("mpi3d", P + "latent/prior/valid_gap/invalid")
    assert all(x == 2048 for r in ROLES for x in valid[r]) and all(x == 0 for r in ROLES for x in invalid[r])
    s1p = diag("mpi3d", P + "physical/prior/energy_score_euclidean/mean", ("S1",))["S1"]
    s1pp = diag("mpi3d", P + "physical/persistence/energy_score_euclidean/mean", ("S1",))["S1"]
    num("MPSOneBeatsOwnPhysPersist", str(sum(a < b for a, b in zip(s1p, s1pp))))
    fx = diag("mpi3d", P + "latent/fixed_prior_mean/normalized_energy_score/mean", ("S1",))["S1"]
    num("MPSOneLatentFixed", fmt(st.mean(fx), 3))
    num("MPSOneLatentSamplingWins", str(sum(f > s for f, s in zip(fx, s1))))
    fxp = diag("mpi3d", P + "physical/fixed_prior_mean/energy_score_euclidean/mean", ("S1",))["S1"]
    num("MPSOnePhysFixed", fmt(st.mean(fxp), 2))


def table_mpi3d_main(analysis):
    """Main-text MPI3D summary: physical score with its persistence decode, and the latent score."""
    P = "forecasts/metrics/"
    body = analysis["datasets"]["mpi3d"]["metrics"]["physical_forecast_energy_score"]["roles"]
    lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Physical ES " + DOWN + r" (grid indices)} & Latent ES " + DOWN + r" (gap-normalized) \\",
             r"\cmidrule(lr){2-3}\cmidrule(l){4-4}",
             r"Model & Sampled prior & Own persistence & Sampled prior \\", r"\midrule"]
    for role in ROLES:
        rec = body[role]
        own = st.mean(diag("mpi3d", P + "physical/persistence/energy_score_euclidean/mean", (role,))[role])
        lat = diag("mpi3d", P + "latent/prior/normalized_energy_score/mean", (role,))[role]
        lines.append(f"{TEX[role]} & {es_text('mpi3d', rec['mean'], rec['sample_sd'])} & {es_text('mpi3d', own)} & "
                     f"{fmt(st.mean(lat), 3)} ({fmt(min(lat), 3)}--{fmt(max(lat), 3)}) \\\\")
    refs = {}
    for key in ("physical/coordinate_persistence", "physical/physical_oracle"):
        vals = diag("mpi3d", P + key + "/energy_score_euclidean/mean")
        flat = {v for r in ROLES for v in vals[r]}
        if len(flat) != 1:
            raise ValueError("Privileged MPI3D reference is not constant")
        refs[key] = flat.pop()
    oracle_latent = {v for r in ROLES for v in diag("mpi3d", P + "latent/encode_decode_oracle/normalized_energy_score/mean")[r]}
    if len(oracle_latent) != 1:
        raise ValueError("Latent two-outcome reference is not constant")
    lines += [r"\midrule",
              r"Coordinate persistence (privileged) & \multicolumn{2}{c}{" + fmt(refs["physical/coordinate_persistence"], 2) + r"} & -- \\",
              r"Two-outcome oracle (privileged) & \multicolumn{2}{c}{" + fmt(refs["physical/physical_oracle"], 2) + r"} & " +
              fmt(oracle_latent.pop(), 3) + r" \\", r"\bottomrule", r"\end{tabular}"]
    (TABLES / "mpi3d_main.tex").write_text("\n".join(lines) + "\n")
    record("tables/mpi3d_main.tex", source="analysis.json (physical) and diagnostics.csv (persistence decode, latent score)",
           aggregation="mean ± sample SD (physical), mean and range (latent) over five replications", bold="none")
    num("MPRefCoordPersist", fmt(refs["physical/coordinate_persistence"], 2))
    num("MPRefOracle", fmt(refs["physical/physical_oracle"], 2))


def table_mp_support():
    lines = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r"Diagnostic & " + " & ".join(TEX[r] for r in ROLES) + r" \\", r"\midrule"]
    fit_p99 = {r: [DIAG[("mpi3d", r, k)][f"readout_selection/candidates/{mp_selected(r, k)}/train_standardized_norm_p99"] for k in REPS] for r in ROLES}
    pred = diag("mpi3d", "forecasts/predicted_feature_support/prior/standardized_norm_p99")
    orac = diag("mpi3d", "forecasts/predicted_feature_support/encode_decode_oracle/standardized_norm_p99")
    frac = diag("mpi3d", "forecasts/predicted_feature_support/prior/fraction_above_fit_p99")
    ofrac = diag("mpi3d", "forecasts/predicted_feature_support/encode_decode_oracle/fraction_above_fit_p99")
    rt = diag("mpi3d", "forecasts/real_target_readout/r2_mean")
    ho = diag("mpi3d", "readout_in_support_report/r2_mean")
    def cell(v, d=2):
        return es_text("mpi3d", v) if v >= 100 else f"{v:.{d}f}"
    rows = [("Fit features, 99th-percentile standardized norm", fit_p99, None, cell),
            ("Predicted features, 99th-percentile standardized norm", pred, None, cell),
            ("Real test targets, 99th-percentile standardized norm", orac, None, cell),
            ("Predicted atoms above fit 99th percentile (fraction)", frac, "min", lambda v: f"{v:.4f}"),
            ("Real test targets above fit 99th percentile (fraction)", ofrac, "min", lambda v: f"{v:.4f}"),
            ("Readout $R^2$, train-attribute hold-out", ho, "max", lambda v: f"{v:.3f}"),
            ("Readout $R^2$, real test-attribute targets", rt, "max", lambda v: f"{v:.3f}")]
    arrow = {None: NODIR, "min": DOWN, "max": UP}
    for label, data, direction, render in rows:
        cells = bold_best([st.mean(data[r]) for r in ROLES], direction, render)
        lines.append(f"{label} {arrow[direction]} & " + " & ".join(cells) + r" \\")
    sel = {role: "/".join(sorted({mp_selected(role, r) for r in REPS})) for role in ROLES}
    lines.append(f"Selected readout transform {NODIR} & " + " & ".join(sel[r] for r in ROLES) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "mpi3d_support.tex").write_text("\n".join(lines) + "\n")
    record("tables/mpi3d_support.tex", aggregation="mean over five replications", bold="best mean in directional rows",
           note="fit p99 uses the candidate with minimum real-feature selection MSE (raw tie-break), as the evaluator does")
    for r in ("R0", "R1"):
        assert all(v == 1.0 for v in frac[r]), "Expected every R0/R1 predicted atom above fit p99"
    num("MPROneFracAbove", "1")
    num("MPRZeroPredNorm", es_text("mpi3d", st.mean(pred["R0"])))
    num("MPROnePredNorm", es_text("mpi3d", st.mean(pred["R1"])))
    num("MPSOnePredNorm", fmt(st.mean(pred["S1"]), 1))
    num("MPFitNormLo", fmt(min(st.mean(fit_p99[r]) for r in ROLES), 1))
    num("MPFitNormHi", fmt(max(st.mean(fit_p99[r]) for r in ROLES), 1))
    for r in ROLES:
        num(f"MP{ROLE_WORD[r]}ReadoutTestRtwo", fmt(st.mean(rt[r]), 3))
        num(f"MP{ROLE_WORD[r]}ReadoutHoldRtwo", fmt(st.mean(ho[r]), 3))
    num("MPSOneReadoutTestRtwoWeak", fmt(rt["S1"][0], 3))


def table_mp_probes():
    keys = [("Position $R^2$ (mean of axes)", "probes/{v}/probes/position/mean_position_r2"),
            ("Colour, balanced accuracy", "probes/{v}/probes/color_id/balanced_accuracy"),
            ("Shape, balanced accuracy", "probes/{v}/probes/shape_id/balanced_accuracy"),
            ("Size, balanced accuracy", "probes/{v}/probes/size_id/balanced_accuracy"),
            ("Camera height, balanced accuracy", "probes/{v}/probes/camera_height/balanced_accuracy")]
    lines = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule", r"Features & Probe (" + UP + r") & " + " & ".join(TEX[r] for r in ROLES) + r" \\", r"\midrule"]
    for view, name in (("pooled", "Pooled backbone"), ("projector", "Projector")):
        for i, (label, key) in enumerate(keys):
            vals = diag("mpi3d", key.format(v=view))
            cells = bold_best([st.mean(vals[r]) for r in ROLES], "max", lambda v: f"{v:.3f}")
            lines.append(f"{name if i == 0 else ''} & {label} & " + " & ".join(cells) + r" \\")
        if view == "pooled":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "mpi3d_probes.tex").write_text("\n".join(lines) + "\n")
    record("tables/mpi3d_probes.tex", keys=[k for _, k in keys], population="2,048 test-attribute clean images (final report bank)",
           bold="highest mean per row")


# ----------------------------------------------------------------------------- cost
def cost_data(review):
    cpu = {(r["dataset"], r["role"]): r for r in load_json(COST / "benchmark_local_cpu_flops.json")["rows"]}
    mps_rows = load_json(COST / "benchmark_apple_mps.json")["rows"]
    recount = {(r["dataset"], r["role"]): r for r in load_json(DATA / "flop_recount.json")["rows"]}
    mps = defaultdict(list)
    for r in mps_rows:
        mps[(r["dataset"], r["role"])].append(r)
    out = {}
    for ds in DATASETS:
        for role in ROLES:
            sweeps = sorted(mps[(ds, role)], key=lambda x: x["repeat"])
            if [s["repeat"] for s in sweeps] != [0, 1]:
                raise ValueError("Expected exactly two MPS sweeps")
            c = cpu[(ds, role)]
            if any(s["stored_parameters"] != c["stored_parameters"] for s in sweeps):
                raise ValueError("Parameter counts differ between benchmarks")
            rc = recount[(ds, role)]
            math_flops = rc["train_flops_per_update"] if ds == "moving_mnist" else rc["math_attention_backend"]["train_flops_per_update"]
            default_flops = rc["train_flops_per_update"] if ds == "moving_mnist" else rc["default_backend"]["train_flops_per_update"]
            if default_flops != c["train_flops_per_update"]:
                raise ValueError("Recount does not reproduce the benchmark FLOP count")
            draws = 64 if ds == "moving_mnist" else 16
            stochastic = role.endswith("1")
            infer = (rc["inference_encoder_flops_per_source"] + rc["inference_prior_flops_per_source"]
                     + (draws if stochastic else 1) * rc["inference_predictor_flops_per_draw"])
            out[(ds, role)] = {
                "passes": {"moving_mnist": {"R0": "2 / 1", "R1": "3 / 2", "S0": "2 / 2", "S1": "3 / 3"},
                           "mpi3d": {"R0": "2 / 1", "R1": "3 / 2", "S0": "2 / 2", "S1": "2 / 2"}}[ds][role],
                "encoder_copies": 2 if role.startswith("R") else 1,
                "stored": c["stored_parameters"], "trainable": c["trainable_parameters"],
                "gflops": c["train_flops_per_update"] / 1e9, "gflops_math": math_flops / 1e9,
                "act_mib": st.mean(s["saved_activation_bytes_per_update"] for s in sweeps) / 2 ** 20,
                "act_mib_cpu": (rc["saved_activation_bytes_per_update"] if ds == "moving_mnist"
                                else rc["math_attention_backend"]["saved_activation_bytes_per_update"]) / 2 ** 20,
                "ms": st.mean(s["median_update_seconds"] for s in sweeps) * 1e3,
                "ms_sweeps": [s["median_update_seconds"] * 1e3 for s in sweeps],
                "dev_s": review["datasets"][ds]["measured_training_seconds_per_role"][role],
                "infer_mflops": infer / 1e6, "infer_draws": draws if stochastic else 1,
            }
    return out


def pct(a, b):
    return 100 * (a / b - 1)


def pct_text(a, b, digits=None):
    p = pct(a, b)
    if digits is None:
        digits = 1 if abs(p) < 5 else 0
    s = f"{p:+.{digits}f}\\%"
    return s.replace("-", "$-$").replace("+", "$+$")


def table_cost(cost):
    """Compact main-text cost table; the full record is table_cost_full."""
    cols = [("stored", lambda v: f"{v / 1e6:.2f}"), ("gflops", lambda v: f"{v:.1f}"),
            ("ms", lambda v: f"{v:.0f}"), ("dev_s", lambda v: f"{v:,.0f}".replace(",", "{,}"))]
    lines = [r"\begin{tabular}{@{}lccccc@{}}", r"\toprule",
             r"Model & Enc.\ passes & Params (M) " + DOWN + r" & GFLOPs " + DOWN +
             r" & ms / update " + DOWN + r" & Wall-clock (s) " + DOWN + r" \\",
             r"\midrule"]
    for ds in DATASETS:
        lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{DSNAME[ds]}}}}} \\")
        columns = [bold_best([cost[(ds, r)][key] for r in ROLES], "min", render) for key, render in cols]
        for i, role in enumerate(ROLES):
            lines.append(f"{TEX[role]} & {cost[(ds, role)]['passes']} & " + " & ".join(c[i] for c in columns) + r" \\")
        s1, r1 = cost[(ds, "S1")], cost[(ds, "R1")]
        lines.append(r"\quad \shvar{} vs.\ \emavar{} & & " + " & ".join(pct_text(s1[k], r1[k]) for k, _ in cols) + r" \\")
        if ds == "moving_mnist":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "training_cost.tex").write_text("\n".join(lines) + "\n")
    record("tables/training_cost.tex", tiers={"analytic": "results/compute_cost/benchmark_local_cpu_flops.json (FLOPs, params)",
           "controlled": "benchmark_apple_mps.json median ms/update, mean of two sweeps",
           "development": "frozen_review.json measured_training_seconds_per_role (Slurm ElapsedRaw, n=1)"},
           status="post hoc, secondary", bold="lowest value per column and dataset")
    for ds in DATASETS:
        s1, r1, s0 = cost[(ds, "S1")], cost[(ds, "R1")], cost[(ds, "S0")]
        d = DS_WORD[ds]
        num(f"{d}CostFlopsSOne", f"{s1['gflops']:.1f}")
        num(f"{d}CostFlopsROne", f"{r1['gflops']:.1f}")
        num(f"{d}CostFlopsSZero", f"{s0['gflops']:.1f}")
        num(f"{d}CostFlopsVsROne", f"{abs(pct(s1['gflops'], r1['gflops'])):.0f}")
        num(f"{d}CostFlopsVsSZero", f"{abs(pct(s1['gflops'], s0['gflops'])):.1f}" if ds == "mpi3d" else f"{abs(pct(s1['gflops'], s0['gflops'])):.0f}")
        num(f"{d}CostParamsVsROne", f"{abs(pct(s1['stored'], r1['stored'])):.0f}")
        num(f"{d}CostParamsVsSZero", f"{abs(pct(s1['stored'], s0['stored'])):.0f}")
        num(f"{d}CostMsSOne", f"{s1['ms']:.0f}")
        num(f"{d}CostMsROne", f"{r1['ms']:.0f}")
        num(f"{d}CostMsSZero", f"{s0['ms']:.0f}")
        num(f"{d}CostMsVsROne", f"{abs(pct(s1['ms'], r1['ms'])):.0f}")
        num(f"{d}CostMsVsSZero", f"{abs(pct(s1['ms'], s0['ms'])):.0f}")
        num(f"{d}CostActSOne", f"{s1['act_mib']:.0f}")
        num(f"{d}CostActROne", f"{r1['act_mib']:.0f}")
        num(f"{d}CostActVsROne", f"{abs(pct(s1['act_mib'], r1['act_mib'])):.1f}" if ds == "mpi3d" else f"{abs(pct(s1['act_mib'], r1['act_mib'])):.0f}")
        num(f"{d}CostDevVsROne", f"{abs(pct(s1['dev_s'], r1['dev_s'])):.0f}")
        num(f"{d}CostStoredSOne", f"{s1['stored'] / 1e6:.2f}")
        num(f"{d}CostStoredROne", f"{r1['stored'] / 1e6:.2f}")
        num(f"{d}CostMathFlopsRatio", f"{s1['gflops_math'] / r1['gflops_math']:.3f}")
        num(f"{d}CostFlopsRatio", f"{s1['gflops'] / r1['gflops']:.3f}")
    extra = cost[("mpi3d", "S1")]["trainable"] - cost[("mpi3d", "S0")]["trainable"]
    num("MPHeadParams", f"{extra:,}".replace(",", "{,}"))
    num("MMHeadParams", f"{cost[('moving_mnist', 'S1')]['trainable'] - cost[('moving_mnist', 'S0')]['trainable']:,}".replace(",", "{,}"))
    sweep_gap = max(abs(c["ms_sweeps"][0] / c["ms_sweeps"][1] - 1) for c in cost.values())
    num("CostSweepGap", f"{100 * sweep_gap:.1f}")
    math_gain = max(cost[("mpi3d", r)]["gflops_math"] / cost[("mpi3d", r)]["gflops"] - 1 for r in ROLES)
    num("CostMathGain", f"{100 * math_gain:.0f}")


def table_cost_full(cost):
    lines = [r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
             r" & & Trainable & \multicolumn{2}{c}{Training GFLOPs} & Saved act. & ms / update & Inference \\",
             r"\cmidrule(lr){4-5}",
             r"Dataset & Model & params (M) & bench. & math attn. & (MiB) & sweep 1 / 2 & MFLOPs \\",
             r"\midrule"]
    for ds in DATASETS:
        spec = [("trainable", lambda v: f"{v / 1e6:.3f}"), ("gflops", lambda v: f"{v:.1f}"), ("gflops_math", lambda v: f"{v:.1f}"),
                ("act_mib", lambda v: f"{v:.0f}"), ("ms", None), ("infer_mflops", lambda v: f"{v:.0f}")]
        columns = {}
        for key, render in spec:
            if key == "ms":
                heads = bold_best([cost[(ds, r)]["ms"] for r in ROLES], "min", lambda v: f"{v:.1f}")
                columns[key] = [(r"\best{" if h.startswith(r"\best{") else "{") +
                                f"{cost[(ds, r)]['ms_sweeps'][0]:.0f} / {cost[(ds, r)]['ms_sweeps'][1]:.0f}" + "}"
                                for h, r in zip(heads, ROLES)]
            else:
                columns[key] = bold_best([cost[(ds, r)][key] for r in ROLES], "min", render)
        for i, role in enumerate(ROLES):
            c = cost[(ds, role)]
            cells = [columns[key][i] for key, _ in spec]
            cells[-1] = f"{cells[-1]} ({c['infer_draws']})"
            lines.append(f"{DSNAME[ds] if role == 'R0' else ''} & {TEX[role]} & " + " & ".join(cells) + r" \\")
        if ds == "moving_mnist":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "training_cost_full.tex").write_text("\n".join(lines) + "\n")
    record("tables/training_cost_full.tex", source=["benchmark_local_cpu_flops.json", "benchmark_apple_mps.json", "data/flop_recount.json"],
           bold="lowest value per column and dataset")
    with open(DATA / "training_cost_record.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "role", "model_name", "encoder_passes_fwd_bwd", "stored_parameters", "trainable_parameters",
                         "train_gflops_benchmark_cpu", "train_gflops_math_attention", "saved_activation_mib_mps",
                         "saved_activation_mib_cpu_math", "mps_ms_per_update_mean_of_sweeps", "mps_ms_sweep1",
                         "mps_ms_sweep2", "development_wallclock_s", "inference_mflops_per_forecast", "inference_draws"])
        for (ds, role), c in cost.items():
            writer.writerow([ds, role, NAME[role], c["passes"], c["stored"], c["trainable"], c["gflops"], c["gflops_math"], c["act_mib"],
                             c["act_mib_cpu"], c["ms"], *c["ms_sweeps"], c["dev_s"], c["infer_mflops"], c["infer_draws"]])


def figure_cost(cost):
    groups = [("gflops", "FLOPs"), ("act_mib", "activ."), ("stored", "params"), ("ms", "time/upd."), ("dev_s", "wall-clock")]
    tiers = ["analytic", "count", "analytic", "MPS", "V100 dev."]
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 1.95), sharey=True)
    width = 0.19
    for ax, ds in zip(axes, DATASETS):
        for gi, (key, label) in enumerate(groups):
            base = cost[(ds, "R1")][key]
            for ri, role in enumerate(ROLES):
                value = cost[(ds, role)][key] / base
                x = gi + (ri - 1.5) * (width + 0.015)
                face = COLOR[role] if FILLED[role] else "white"
                ax.bar(x, value, width=width, color=face, edgecolor=COLOR[role], linewidth=0.9, zorder=3,
                       hatch=None if FILLED[role] else "////")
                PLOT_ROWS["training_cost"].append({"figure": "training_cost", "panel": ds, "series": key, "dataset": ds,
                                                    "role": role, "replication": "", "metric_key": key,
                                                    "transformation": "divided by AdaSSL reference (R1)", "value": repr(value)})
                if role == "S1":
                    ax.text(x, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=5.5, color=INK, rotation=90)
        ax.axhline(1.0, color=MUTED, linewidth=0.6, linestyle=":", zorder=2)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{g[1]}\n({t})" for g, t in zip(groups, tiers)], fontsize=6.3)
        ax.set_title(f"{'(a) Moving-MNIST: posterior sees a separate clip' if ds == 'moving_mnist' else '(b) MPI3D: posterior reuses the target encoding'}",
                     loc="left", fontsize=7.3)
        ax.set_ylim(0, 1.85)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("relative to AdaSSL reference")
    handles = [matplotlib.patches.Patch(facecolor=COLOR[r] if FILLED[r] else "white", edgecolor=COLOR[r],
                                        hatch=None if FILLED[r] else "////", label=NAME[r]) for r in ROLES]
    axes[1].legend(handles=handles, ncol=2, frameon=False, loc="upper right", fontsize=6.0, handlelength=1.2, columnspacing=0.8)
    fig.tight_layout(w_pad=0.6)
    save(fig, "training_cost", question="What does Shared-Var (S1) cost relative to AdaSSL reference (R1) and Shared-Det (S0)?", status="post hoc, secondary",
         normalization="each quantity divided by the R1 value of the same dataset", tiers="analytic / controlled / development")


# ----------------------------------------------------------------------------- core figures
def figure_physical(analysis):
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.35))
    body = {ds: analysis["datasets"][ds]["metrics"]["physical_forecast_energy_score"]["roles"] for ds in DATASETS}
    ax = axes[0]
    for i, role in enumerate(ROLES):
        vals = body["moving_mnist"][role]["values"]
        scatter_role(ax, i, vals, role)
        mean_bar(ax, i, vals)
    role_axis(ax)
    ax.set_ylim(0, 3.8)
    ax.set_ylabel("physical Energy Score (pixels/frame)")
    ax.set_title("(a) Moving-MNIST", loc="left")
    pix = diag("moving_mnist", "forecasts/pixel_constant_velocity/strata/all/energy_score")["R0"][0]
    ora = diag("moving_mnist", "forecasts/independent_privileged_oracle/strata/all/energy_score")["R0"][0]
    ax.axhline(pix, linestyle="--", color=MUTED, linewidth=0.7, zorder=1)
    ax.axhline(ora, linestyle=":", color=MUTED, linewidth=0.8, zorder=1)
    ax.add_artist(ax.legend(handles=[Line2D([], [], linestyle="--", color=MUTED, label="pixel constant velocity (source pixels)"),
                                     Line2D([], [], linestyle=":", color=MUTED, label="simulator oracle (privileged)")],
                            frameon=False, loc="center right", fontsize=6.3, bbox_to_anchor=(1.0, 0.62)))
    plot_rows("physical_forecast_results", "a", "model_prior", "moving_mnist", "physical_forecast_energy_score",
              {r: body["moving_mnist"][r]["values"] for r in ROLES})
    ax = axes[1]
    for i, role in enumerate(ROLES):
        vals = body["mpi3d"][role]["values"]
        scatter_role(ax, i, vals, role)
        mean_bar(ax, i, vals, log=True)
    role_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(0.7, 3e7)
    ax.set_ylabel("physical Energy Score (grid indices)")
    ax.set_title("(b) MPI3D", loc="left")
    ax.axhline(2.0, linestyle="--", color=MUTED, linewidth=0.7, zorder=1)
    ax.axhline(1.0, linestyle=":", color=MUTED, linewidth=0.8, zorder=1)
    ax.legend(handles=[Line2D([], [], linestyle="--", color=MUTED, label="coordinate persistence (privileged)"),
                       Line2D([], [], linestyle=":", color=MUTED, label="two-outcome oracle (privileged)")],
              frameon=False, loc="upper right", fontsize=6.3)
    plot_rows("physical_forecast_results", "b", "model_prior", "mpi3d", "physical_forecast_energy_score",
              {r: body["mpi3d"][r]["values"] for r in ROLES})
    fig.tight_layout(w_pad=1.2)
    save(fig, "physical_forecast_results", question="How do the four roles compare on the primary metric?",
         metric="physical_forecast_energy_score", inclusion="all 40 runs", aggregation="points are runs (left to right replications 1-5); "
         "black bar/line: arithmetic mean ± sample SD (MPI3D: mean marker and ±SD line on log axis, lower end clipped at half the minimum run)",
         baselines="MM pixel constant velocity and simulator oracle; MPI3D privileged coordinate persistence and two-outcome oracle; "
         "each one fixed-bank calculation", direction="lower is better")


def figure_mpi3d_failure():
    P = "forecasts/metrics/latent/"
    prior = diag("mpi3d", P + "prior/normalized_energy_score/mean")
    persist = diag("mpi3d", P + "persistence/normalized_energy_score/mean")
    fixed = diag("mpi3d", P + "fixed_prior_mean/normalized_energy_score/mean")
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.3))
    ax = axes[0]
    excess = {r: [a - b for a, b in zip(prior[r], persist[r])] for r in ROLES}
    for i, role in enumerate(ROLES):
        scatter_role(ax, i, excess[role], role)
        mean_bar(ax, i, excess[role])
    role_axis(ax)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.text(-0.55, 0.03, "own persistence control", ha="left", va="bottom", fontsize=6.5, color=MUTED)
    ax.set_ylim(-0.1, 2.6)
    ax.set_ylabel("latent ES $-$ persistence ES\n(gap-normalized; $>0$ is worse)")
    ax.set_title("(a) Loss to own persistence, all 20 runs", loc="left")
    plot_rows("mpi3d_forecast_failure", "a", "prior_minus_persistence", "mpi3d",
              P + "prior/normalized_energy_score/mean minus " + P + "persistence/normalized_energy_score/mean", excess, "difference")
    ax = axes[1]
    fit = {r: [DIAG[("mpi3d", r, k)][f"readout_selection/candidates/{mp_selected(r, k)}/train_standardized_norm_p99"] for k in REPS] for r in ROLES}
    pred = diag("mpi3d", "forecasts/predicted_feature_support/prior/standardized_norm_p99")
    real = diag("mpi3d", "forecasts/predicted_feature_support/encode_decode_oracle/standardized_norm_p99")
    ratio = {r: [p / f for p, f in zip(pred[r], fit[r])] for r in ROLES}
    rratio = {r: [p / f for p, f in zip(real[r], fit[r])] for r in ROLES}
    for i, role in enumerate(ROLES):
        scatter_role(ax, i, ratio[role], role)
        xs = [i + REP_OFFSET[k] + 0.30 for k in REPS]
        ax.scatter(xs, rratio[role], s=9, marker="x", color=MUTED, linewidths=0.7, zorder=3)
    role_axis(ax)
    ax.set_yscale("log")
    ax.set_ylim(0.3, 3e5)
    ax.axhline(1.0, color=INK, linewidth=0.8)
    ax.set_ylabel("p99 standardized norm / fit p99\n(log scale)")
    ax.set_title("(b) Readout support of predicted features", loc="left")
    ax.legend(handles=[Line2D([], [], marker="s", linestyle="", markerfacecolor="white", markeredgecolor=MUTED, label="predicted atoms"),
                       Line2D([], [], marker="x", linestyle="", color=MUTED, label="real test targets")],
              frameon=False, loc="upper right", fontsize=6.5)
    plot_rows("mpi3d_forecast_failure", "b", "predicted_norm_ratio", "mpi3d",
              "forecasts/predicted_feature_support/prior/standardized_norm_p99 / readout fit p99", ratio, "ratio")
    plot_rows("mpi3d_forecast_failure", "b", "real_target_norm_ratio", "mpi3d",
              "forecasts/predicted_feature_support/encode_decode_oracle/standardized_norm_p99 / readout fit p99", rratio, "ratio")
    fig.tight_layout(w_pad=1.2)
    save(fig, "mpi3d_forecast_failure", question="Is the MPI3D problem only the physical decoder?",
         metric="gap-normalized latent ES minus own persistence; predicted-feature support ratios",
         inclusion="all 20 MPI3D runs; all 2,048 queries have a defined (non-degenerate) gap",
         warning="latent spaces are separately learned; values compare each run only with its own control")
    num("MPExcessSOne", fmt(st.mean(excess["S1"]), 3))


def figure_residual_usage():
    kl = diag("moving_mnist", "latent/gaussian_diagnostics/mean_training_information_kl", ("R1", "S1"))
    gain = diag("moving_mnist", "diagnostics/delta_fixed_es", ("R1", "S1"))
    pstd = diag("moving_mnist", "latent/gaussian_diagnostics/prior_std_mean", ("R1", "S1"))
    qstd = diag("moving_mnist", "latent/gaussian_diagnostics/posterior_std_mean", ("R1", "S1"))
    P = "forecasts/metrics/"
    mkl = {r: [(a + b) / 2 for a, b in zip(diag("mpi3d", P + "diagnostic_posterior_kl_0/mean", (r,))[r],
                                           diag("mpi3d", P + "diagnostic_posterior_kl_1/mean", (r,))[r])] for r in ("R1", "S1")}
    mgain = {r: [f - s for f, s in zip(diag("mpi3d", P + "latent/fixed_prior_mean/normalized_energy_score/mean", (r,))[r],
                                       diag("mpi3d", P + "latent/prior/normalized_energy_score/mean", (r,))[r])] for r in ("R1", "S1")}
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95))
    for ax, x, y, title, xl, yl in (
            (axes[0], kl, gain, "(a) Moving-MNIST", "posterior–prior KL (nats)", "fixed-mean ES $-$ sampled ES"),
            (axes[1], pstd, qstd, "(b) Moving-MNIST scales", "mean prior std", "mean posterior std"),
            (axes[2], mkl, mgain, "(c) MPI3D", "posterior–prior KL (nats)", "fixed-mean $-$ sampled\n(gap-norm. latent ES)")):
        for role in ("R1", "S1"):
            face = COLOR[role]
            ax.scatter(x[role], y[role], s=16, marker=MARKER[role], facecolors=face, edgecolors=face, zorder=3, label=NAME[role])
        ax.set_title(title, loc="left")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].axhline(0, color=INK, linewidth=0.7)
    axes[2].axhline(0, color=INK, linewidth=0.7)
    axes[0].set_ylim(-0.02, 0.4)
    axes[2].set_ylim(-0.05, 0.35)
    axes[1].set_xlim(0, 10.5)
    axes[1].set_ylim(0, 0.8)
    axes[0].legend(frameon=False, loc="lower right", fontsize=6.5, handletextpad=0.1)
    fig.tight_layout(w_pad=0.8)
    for ax, x, y in ((axes[0], kl, gain), (axes[1], pstd, qstd), (axes[2], mkl, mgain)):
        label_points(fig, ax, [(xi, yi, str(k)) for role in ("R1", "S1")
                               for k, (xi, yi) in enumerate(zip(x[role], y[role]), start=1)])
    for name, data in (("kl", kl), ("delta_fixed_es", gain), ("prior_std", pstd), ("posterior_std", qstd)):
        plot_rows("residual_usage", "ab", name, "moving_mnist", name, data)
    plot_rows("residual_usage", "c", "posterior_kl_mean_of_outcomes", "mpi3d",
              "mean of forecasts/metrics/diagnostic_posterior_kl_{0,1}/mean", mkl, "mean of two outcomes")
    plot_rows("residual_usage", "c", "latent_sampling_gain", "mpi3d",
              "latent fixed_prior_mean minus prior normalized_energy_score/mean", mgain, "difference")
    save(fig, "residual_usage", question="Is the Gaussian residual used at the end of training?",
         inclusion="R1 and S1, five runs each; numbers next to points are replication IDs",
         note="endpoint values only; posterior evaluated on query trajectories for diagnosis, never used to forecast; "
              "MPI3D KL is defined for all 2,048 queries per outcome")
    for r in ("R1", "S1"):
        num(f"MP{ROLE_WORD[r]}PostKL", fmt(st.mean(mkl[r]), 2))


def figure_paired(analysis):
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 1.9))
    panels = [("moving_mnist", "S1-R1"), ("moving_mnist", "S1-S0"), ("mpi3d", "S1-R1"), ("mpi3d", "S1-S0")]
    for ax, (ds, name) in zip(axes, panels):
        rec = analysis["datasets"][ds]["metrics"]["physical_forecast_energy_score"]["contrasts"][name]
        d = rec["paired_differences"]
        xs = [REP_OFFSET[k] * 2 for k in REPS]
        ax.scatter(xs, d, s=14, marker="s", color=COLOR["S1"], zorder=3)
        lo, hi = rec["interval_95"]
        ax.errorbar([0.45], [rec["mean"]], yerr=[[rec["mean"] - lo], [hi - rec["mean"]]], fmt="D", color=INK,
                    markersize=3.5, elinewidth=0.9, capsize=2, zorder=4)
        ax.axhline(0, color=MUTED, linewidth=0.7, linestyle="--")
        ax.set_xlim(-0.4, 0.7)
        ax.set_xticks([])
        ax.set_title(DSNAME[ds] + "\n" + CONTRAST_PLOT[name].replace(" − ", " −\n"), fontsize=6.8)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        PLOT_ROWS["paired_physical_effects"] += [{"figure": "paired_physical_effects", "panel": f"{ds}/{name}", "series": "paired difference",
                                                  "dataset": ds, "role": name, "replication": k, "metric_key": "physical_forecast_energy_score",
                                                  "transformation": "S1 minus reference, same replication", "value": repr(v)}
                                                 for k, v in zip(REPS, d)]
    axes[0].set_ylabel("Shared-Var $-$ reference\n($<0$ favours Shared-Var)")
    for ax in axes:
        lo, hi = ax.get_ylim()
        pad = 0.06 * (hi - lo)
        ax.set_ylim(lo - pad, hi + pad)
    fig.tight_layout(w_pad=0.5)
    save(fig, "paired_physical_effects", question="Primary paired contrasts", interval="paired t, df=4, individual 95%",
         note="squares are replications 1-5 left to right; diamond and bar: mean and interval; no significance marks")


def figure_sampling_control():
    fig, axes = plt.subplots(1, 2, figsize=(3.6, 2.0), sharey=True)
    for ax, role in zip(axes, ("R1", "S1")):
        fixed = diag("moving_mnist", "forecasts/model_fixed_prior_mean/strata/all/energy_score", (role,))[role]
        sampled = diag("moving_mnist", "forecasts/model_prior/strata/all/energy_score", (role,))[role]
        for k, (a, b) in enumerate(zip(fixed, sampled), start=1):
            ax.plot([0, 1], [a, b], color=COLOR[role], linewidth=0.9, marker=MARKER[role], markersize=3.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["prior mean", "sampled prior"])
        ax.set_xlim(-0.3, 1.3)
        ax.set_title(NAME[role], loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        plot_rows("moving_mnist_sampling_control", role, "fixed_prior_mean", "moving_mnist",
                  "forecasts/model_fixed_prior_mean/strata/all/energy_score", {role: fixed})
        plot_rows("moving_mnist_sampling_control", role, "sampled", "moving_mnist",
                  "forecasts/model_prior/strata/all/energy_score", {role: sampled})
    axes[0].set_ylabel("physical ES (pixels/frame)")
    fig.tight_layout()
    save(fig, "moving_mnist_sampling_control", question="Does sampling the prior improve on its mean in the same trained model?",
         note="training identical within each pair; five lines per panel, one per replication; identities are in the plot-data CSV")


def figure_representation():
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.05))
    specs = [("(a) Source digit accuracy", [("probes/source/online_backbone/digit/accuracy", "backbone", "o"),
                                           ("probes/source/online_projector/digit/accuracy", "projector", "s")], (0.3, 1.0)),
             ("(b) Source projector velocity $R^2$", [("probes/source/online_projector/velocity/r2_xy/0", "$x$", "o"),
                                                    ("probes/source/online_projector/velocity/r2_xy/1", "$y$", "s")], (0, 1.05)),
             ("(c) Forecast readout velocity $R^2$", [("forecast_readout/r2_xy/0", "$x$", "o"),
                                                    ("forecast_readout/r2_xy/1", "$y$", "s")], (0, 1.05))]
    for ax, (title, series, ylim) in zip(axes, specs):
        for si, (key, label, marker) in enumerate(series):
            vals = diag("moving_mnist", key)
            for i, role in enumerate(ROLES):
                xs = [i + (si - 0.5) * 0.36 + REP_OFFSET[k] * 0.6 for k in REPS]
                ax.scatter(xs, vals[role], s=8, marker=marker, facecolors=COLOR[role] if si == 0 else "white",
                           edgecolors=COLOR[role], linewidths=0.8, zorder=3)
            plot_rows("moving_mnist_representation_tradeoffs", title, label, "moving_mnist", key, vals)
        role_axis(ax, two_line=True)
        ax.set_ylim(*ylim)
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.legend(handles=[Line2D([], [], marker=m, linestyle="", markerfacecolor=MUTED if j == 0 else "white",
                                  markeredgecolor=MUTED, markersize=4, label=l) for j, (_, l, m) in enumerate(series)],
                  frameon=False, loc="lower left", fontsize=6)
    fig.tight_layout(w_pad=0.6)
    save(fig, "moving_mnist_representation_tradeoffs",
         question="Do forecasting gains coincide with better representations?",
         populations="(a,b) online source-clip probes on final report clips; (c) target-space readout on real target clips",
         note="left/right of each role: first/second series; within series replications 1-5 left to right")


def figure_readout_fidelity():
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.0))
    mm = diag("moving_mnist", "forecasts/encoded_decoded_oracle/strata/all/energy_score")
    ora = diag("moving_mnist", "forecasts/independent_privileged_oracle/strata/all/energy_score")["R0"][0]
    for i, role in enumerate(ROLES):
        scatter_role(axes[0], i, mm[role], role, size=12)
    role_axis(axes[0], two_line=True)
    axes[0].axhline(ora, color=MUTED, linestyle=":", linewidth=0.7)
    axes[0].set_ylim(0.5, 1.4)
    axes[0].set_title("(a) Moving-MNIST", loc="left", fontsize=7.5)
    axes[0].set_ylabel("ES of decoded real futures")
    plot_rows("readout_fidelity", "a", "encode_decode_oracle", "moving_mnist", "forecasts/encoded_decoded_oracle/strata/all/energy_score", mm)
    mp = diag("mpi3d", "forecasts/metrics/physical/encode_decode_oracle/energy_score_euclidean/mean")
    for i, role in enumerate(ROLES):
        scatter_role(axes[1], i, mp[role], role, size=12)
    role_axis(axes[1], two_line=True)
    axes[1].axhline(1.0, color=MUTED, linestyle=":", linewidth=0.7)
    axes[1].set_ylim(0, 8.5)
    axes[1].set_title("(b) MPI3D", loc="left", fontsize=7.5)
    plot_rows("readout_fidelity", "b", "encode_decode_oracle", "mpi3d",
              "forecasts/metrics/physical/encode_decode_oracle/energy_score_euclidean/mean", mp)
    ho = diag("mpi3d", "readout_in_support_report/r2_mean")
    rt = diag("mpi3d", "forecasts/real_target_readout/r2_mean")
    for i, role in enumerate(ROLES):
        xs = [i - 0.17 + REP_OFFSET[k] * 0.6 for k in REPS]
        axes[2].scatter(xs, ho[role], s=8, marker="o", facecolors=COLOR[role], edgecolors=COLOR[role], zorder=3)
        xs = [i + 0.17 + REP_OFFSET[k] * 0.6 for k in REPS]
        axes[2].scatter(xs, rt[role], s=8, marker="s", facecolors="white", edgecolors=COLOR[role], linewidths=0.8, zorder=3)
    role_axis(axes[2], two_line=True)
    axes[2].set_ylim(0.3, 1.0)
    axes[2].set_title("(c) MPI3D readout $R^2$", loc="left", fontsize=7.5)
    axes[2].legend(handles=[Line2D([], [], marker="o", linestyle="", color=MUTED, markersize=4, label="train-attr. hold-out"),
                            Line2D([], [], marker="s", linestyle="", markerfacecolor="white", markeredgecolor=MUTED, markersize=4,
                                   label="test-attr. targets")], frameon=False, loc="lower left", fontsize=6)
    plot_rows("readout_fidelity", "c", "in_support_holdout_r2", "mpi3d", "readout_in_support_report/r2_mean", ho)
    plot_rows("readout_fidelity", "c", "real_test_target_r2", "mpi3d", "forecasts/real_target_readout/r2_mean", rt)
    fig.tight_layout(w_pad=0.6)
    save(fig, "readout_fidelity", question="How faithfully does each fitted readout decode real target features?",
         baselines="dotted: privileged oracle (MM 0.610, MPI3D 1.0)", note="aggregate statistics only; no invented histograms")


def figure_interval_coverage():
    fig, axes = plt.subplots(1, 2, figsize=(4.0, 2.0), sharey=True)
    labels = []
    for ax, i, axis in ((axes[0], 0, "$x$"), (axes[1], 1, "$y$")):
        cov = diag("moving_mnist", f"diagnostics/marginal_90_interval/coverage_xy/{i}", ("R1", "S1"))
        wid = diag("moving_mnist", f"diagnostics/marginal_90_interval/mean_width_xy/{i}", ("R1", "S1"))
        for role in ("R1", "S1"):
            ax.scatter(wid[role], cov[role], s=16, marker=MARKER[role], color=COLOR[role], zorder=3, label=NAME[role])
        labels.append((ax, [(xw, yc, str(k)) for role in ("R1", "S1")
                            for k, (xw, yc) in enumerate(zip(wid[role], cov[role]), start=1)]))
        ax.axhline(0.9, color=MUTED, linestyle="--", linewidth=0.7)
        ax.set_xlabel("mean interval width (px/frame)")
        ax.set_title(f"{axis} velocity", loc="left")
        ax.grid(color=GRID, linewidth=0.5)
        ax.set_xlim(0, 3.2)
        ax.set_ylim(0.3, 1.0)
        plot_rows("moving_mnist_interval_coverage", axis, "coverage", "moving_mnist", f"diagnostics/marginal_90_interval/coverage_xy/{i}", cov)
        plot_rows("moving_mnist_interval_coverage", axis, "width", "moving_mnist", f"diagnostics/marginal_90_interval/mean_width_xy/{i}", wid)
    axes[0].set_ylabel("empirical coverage")
    axes[0].legend(frameon=False, loc="lower right", fontsize=6.5)
    fig.tight_layout()
    for ax, points in labels:
        label_points(fig, ax, points)
    save(fig, "moving_mnist_interval_coverage", question="Do stochastic forecasts carry useful spread?",
         note="empirical marginal 5-95% intervals from 64 draws against 128 truth draws; deterministic roles have zero width and zero coverage; not a calibration proof")


def figure_speed_strata():
    strata = [("speed_lt_1", "$|v|<1$"), ("speed_1_to_2", "$1\\leq|v|<2$"), ("speed_ge_2", "$|v|\\geq2$")]
    fig, ax = plt.subplots(figsize=(5.2, 2.2))
    for si, (s, label) in enumerate(strata):
        vals = diag("moving_mnist", f"forecasts/model_prior/strata/{s}/energy_score")
        q = diag("moving_mnist", f"forecasts/model_prior/strata/{s}/queries")
        n = {int(v) for r in ROLES for v in q[r]}
        if len(n) != 1:
            raise ValueError("Stratum query counts differ across runs")
        for ri, role in enumerate(ROLES):
            x = si + (ri - 1.5) * 0.2
            xs = [x + REP_OFFSET[k] * 0.35 for k in REPS]
            ax.scatter(xs, vals[role], s=7, marker=MARKER[role], facecolors=COLOR[role] if FILLED[role] else "white",
                       edgecolors=COLOR[role], linewidths=0.8, zorder=3)
        for key, style, lab in (("pixel_constant_velocity", "--", "pixel constant velocity"), ("independent_privileged_oracle", ":", "simulator oracle")):
            v = diag("moving_mnist", f"forecasts/{key}/strata/{s}/energy_score")["R0"][0]
            ax.plot([si - 0.42, si + 0.42], [v, v], linestyle=style, color=MUTED, linewidth=0.8, zorder=2,
                    label=lab if si == 0 else None)
            PLOT_ROWS["moving_mnist_speed_strata"].append({"figure": "moving_mnist_speed_strata", "panel": s, "series": key,
                                                            "dataset": "moving_mnist", "role": "shared", "replication": "",
                                                            "metric_key": f"forecasts/{key}/strata/{s}/energy_score",
                                                            "transformation": "none", "value": repr(v)})
        plot_rows("moving_mnist_speed_strata", s, "model_prior", "moving_mnist", f"forecasts/model_prior/strata/{s}/energy_score", vals)
        strata[si] = (s, f"{label}\n({n.pop()} queries)")
    ax.set_xticks(range(3))
    ax.set_xticklabels([l for _, l in strata])
    ax.set_ylabel("physical ES (pixels/frame)")
    ax.set_xlabel("initial speed stratum")
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    handles = [Line2D([], [], linestyle="", marker=MARKER[r], markersize=4, markerfacecolor=COLOR[r] if FILLED[r] else "white",
                      markeredgecolor=COLOR[r], label=NAME[r]) for r in ROLES]
    handles += [Line2D([], [], linestyle="--", color=MUTED, label="pixel constant velocity"),
                Line2D([], [], linestyle=":", color=MUTED, label="simulator oracle (privileged)")]
    ax.legend(handles=handles, frameon=False, fontsize=6, loc="upper left", bbox_to_anchor=(1.01, 1.0), ncol=1)
    fig.tight_layout()
    save(fig, "moving_mnist_speed_strata", question="Where does the S1-S0 gain arise? (post hoc, secondary)",
         note="strata by initial speed |v| of the source state; counts are identical fixed-bank query counts")


def regenerate_dataset_protocols():
    from iwm_replication.moving_mnist import GeneratorConfig, render_centers, sample_future_velocities, sample_source, trajectory_centers
    from iwm_replication.moving_mnist_data import load_digits, load_identity_manifest
    from iwm_replication.mpi3d_data import MPI3DTaskState, mpi3d_attribute_split
    data_dir = MNIST_DIR
    manifest = load_identity_manifest(data_dir / "concept2-identities-v1-seed0.json")
    images, labels, ids = load_digits(data_dir, manifest, "train")
    index = 0  # first identity of the training split, fixed before rendering
    cfg = GeneratorConfig()
    rule = ("first s = 0, 1, 2, ... with SeedSequence([20260930, s]) such that both initial velocity components are >= 1.2 "
            "px/frame, the first x-changed and first y-changed of 16 future draws each change velocity by >= 1 px/frame, and "
            "all nine digit centres of both futures lie in [9, 55] (digit fully visible)")
    for s in range(1000):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20260930, s])))
        state = sample_source(ids[index], int(labels[index]), rng, cfg)
        v = np.asarray(state.velocity)
        if v.min() < 1.2:
            continue
        velocities, axes_changed = sample_future_velocities(state, 16, rng, cfg)
        if not ((axes_changed == 0).any() and (axes_changed == 1).any()):
            continue
        picks = [int(np.flatnonzero(axes_changed == 0)[0]), int(np.flatnonzero(axes_changed == 1)[0])]
        ok = all(abs(velocities[p] - v).max() >= 1.0 and trajectory_centers(state, velocities[p]).min() >= 9
                 and trajectory_centers(state, velocities[p]).max() <= 55 for p in picks)
        if ok:
            seed = s
            break
    frames = [render_centers(images[index], trajectory_centers(state, velocities[p]), cfg).numpy() for p in picks]
    W, H = 5.5, 3.45
    fig = plt.figure(figsize=(W, H))
    def axes(left, bottom, width, height):
        return fig.add_axes([left / W, bottom / H, width / W, height / H])
    seg_color = ("#5a5a5a", COLOR["R1"], COLOR["S1"])
    size, gap, left0 = 0.5, 0.04, 0.64
    fig.text(0.02 / W, 3.36 / H, "(a) Moving-MNIST: one observed clip, two sampled futures", fontsize=7.5, va="top")
    names = ["$x$: frames 1–3 (observed)", "$y$: frames 4–6 (target)", "$u$: frames 7–9 (posterior input)"]
    for seg, name in enumerate(names):
        x_mid = left0 + (3 * seg + 1.5) * (size + gap) - gap / 2
        fig.text(x_mid / W, 3.12 / H, name, ha="center", va="bottom", fontsize=6.5, color=seg_color[seg])
    for row, (clip, pick) in enumerate(zip(frames, picks)):
        bottom = 2.47 - row * (size + 0.1)
        for t in range(9):
            ax = axes(left0 + t * (size + gap), bottom, size, size)
            ax.imshow(clip[t], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(seg_color[t // 3])
                spine.set_linewidth(1.2)
            if row == 0:
                ax.set_title(str(t + 1), fontsize=6.5, pad=1.5)
            if t == 0:
                changed = "$x$" if axes_changed[pick] == 0 else "$y$"
                ax.set_ylabel(f"future {'A' if row == 0 else 'B'}\n({changed} changes)", fontsize=6.5, rotation=0,
                              ha="right", va="center", labelpad=3)
    fig.text(0.02 / W, 1.8 / H, "(b) MPI3D-S: source image and command right; the command fails or succeeds with probability 1/2",
             fontsize=7.5, va="top")
    asset = ROOT / "notebooks/assets/mpi3d_one_to_many_scenarios.png"
    image = plt.imread(register_input(asset))
    crop = (1015, 1496, 392, 1616)  # rows, then columns, of the stochastic-execution row in the original pixels
    sub = image[crop[0]:crop[1], crop[2]:crop[3]]
    height = 1.58
    width = height * sub.shape[1] / sub.shape[0]
    bottom_ax = axes((W - width) / 2, 0.04, width, height)
    bottom_ax.imshow(sub, interpolation="lanczos")
    bottom_ax.set_xticks([])
    bottom_ax.set_yticks([])
    for spine in bottom_ax.spines.values():
        spine.set_visible(False)
    fig.savefig(FIGURES / "dataset_protocols.pdf", pad_inches=0.02)
    fig.savefig(FIGURES / "dataset_protocols.png", pad_inches=0.02, dpi=300)
    plt.close(fig)
    info = {"mm_identity": ids[index], "mm_digit_label": int(labels[index]), "mm_seed_sequence": [20260930, seed],
            "mm_selection_rule": rule,
            "mm_source_state": {"center": list(state.center), "velocity": list(state.velocity)},
            "mm_future_draw_indices": picks, "mm_future_velocities": [velocities[p].tolist() for p in picks],
            "mm_split": "training identity (illustration only; not an evaluation example)",
            "mpi3d_asset": rel(asset), "mpi3d_crop_rows_cols": list(crop),
            "mpi3d_source_factors": {"colour": 4, "shape": 3, "size": 1, "camera": 1, "position": [20, 20], "command": "right"},
            "mpi3d_split": "validation attribute combination: (4 + 3 + 3*1) mod 6 = 4"}
    if mpi3d_attribute_split(MPI3DTaskState(4, 3, 1, 20, 20)) != "validation":
        raise ValueError("MPI3D illustration is expected to be a validation-split object")
    record("figures/dataset_protocols.pdf", png="figures/dataset_protocols.png", **info)
    with open(DATA / "dataset_protocols.json", "w") as handle:
        json.dump(info, handle, indent=2)
    num("FigMMIdentity", ids[index])
    num("FigMMDigit", str(int(labels[index])))
    num("FigMMSeed", str(seed))
    num("FigMMVelocity", f"({state.velocity[0]:.2f}, {state.velocity[1]:.2f})")
    num("FigMMFutureA", f"({velocities[picks[0]][0]:.2f}, {velocities[picks[0]][1]:.2f})")
    num("FigMMFutureB", f"({velocities[picks[1]][0]:.2f}, {velocities[picks[1]][1]:.2f})")


def figure_dataset_protocols():
    if REGENERATE_ILLUSTRATION:
        return regenerate_dataset_protocols()
    info = load_json(DATA / "dataset_protocols.json")
    register_input(FIGURES / "dataset_protocols.pdf")
    record("figures/dataset_protocols.pdf", png="figures/dataset_protocols.png", **info)
    num("FigMMIdentity", info["mm_identity"])
    num("FigMMDigit", str(info["mm_digit_label"]))
    num("FigMMSeed", str(info["mm_seed_sequence"][1]))
    num("FigMMVelocity", "({:.2f}, {:.2f})".format(*info["mm_source_state"]["velocity"]))
    num("FigMMFutureA", "({:.2f}, {:.2f})".format(*info["mm_future_velocities"][0]))
    num("FigMMFutureB", "({:.2f}, {:.2f})".format(*info["mm_future_velocities"][1]))


def figure_information_flow():
    """Keep the reviewed vector export from the editable Figma architecture."""
    source = FIGURES / "src" / "model_information_flow.figma.json"
    design = load_json(source)
    exported = FIGURES / "model_information_flow.pdf"
    if sha256(exported) != design["pdf_sha256"]:
        raise ValueError("Architecture PDF changed: re-export the Figma frame and update its provenance")
    if shutil.which("gs"):
        subprocess.run(["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-sDEVICE=png16m", "-r150",
                        f"-sOutputFile={FIGURES / 'model_information_flow.png'}", str(exported)], check=True)
    record("figures/model_information_flow.pdf", source=rel(source), png="figures/model_information_flow.png",
           figma_url=design["url"], pdf_sha256=design["pdf_sha256"],
           note="reviewed Figma vector export; retained by the asset generator; no data")



def table_mm_primary(analysis):
    body = analysis["datasets"]["moving_mnist"]["metrics"]["physical_forecast_energy_score"]["roles"]
    lowest = min(body[role]["mean"] for role in ROLES)
    lines = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
             r"Model & Energy Score $\downarrow$ (mean $\pm$ SD) & Range \\", r"\midrule"]
    for role in ROLES:
        rec = body[role]
        text = es_text("moving_mnist", rec["mean"], rec["sample_sd"])
        if rec["mean"] == lowest:
            text = r"\best{" + text + "}"
        lines.append(TEX[role] + " & " + text + " & " +
                     es_text("moving_mnist", min(rec["values"])) + "--" +
                     es_text("moving_mnist", max(rec["values"])) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "moving_mnist_primary.tex").write_text("\n".join(lines) + "\n")
    record("tables/moving_mnist_primary.tex", source="analysis.json (recomputed and asserted equal)",
           dataset="moving_mnist", metric="physical_forecast_energy_score", roles=list(ROLES),
           uncertainty="sample SD over five training replications", bold="lowest observed mean")


def table_cost_summary(cost):
    lines = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
             r"Dataset & Stored parameters & Training FLOPs / update \\", r"\midrule"]
    for ds in DATASETS:
        s1, r1 = cost[(ds, "S1")], cost[(ds, "R1")]
        for shared_role, ema_role in (("S0", "R0"), ("S1", "R1")):
            if cost[(ds, shared_role)]["trainable"] != cost[(ds, ema_role)]["trainable"]:
                raise ValueError("Expected equal trainable parameters within each prediction family")
        cells = [f"${pct(s1[key], r1[key]):+.0f}\\%$" for key in ("stored", "gflops")]
        lines.append(DSNAME[ds] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (TABLES / "cost_summary.tex").write_text("\n".join(lines) + "\n")
    record("tables/cost_summary.tex", source="same cost_data as tables/training_cost.tex",
           transformation="100 * (Shared-Var / AdaSSL reference - 1), by dataset",
           status="post hoc, secondary", tier="analytic", units="percent")


def figure_mm_paired_comparisons(analysis):
    body = analysis["datasets"]["moving_mnist"]["metrics"]["physical_forecast_energy_score"]
    fig, ax = plt.subplots(figsize=(5.5, 1.8))
    bounds = [0.0]
    for y, reference, name in ((1, "S0", "S1-S0"), (0, "R1", "S1-R1")):
        rec = body["contrasts"][name]
        differences = rec["paired_differences"]
        lo, hi = rec["interval_95"]
        bounds.extend([lo, hi, *differences])
        ys = [y + 0.13 + REP_OFFSET[rep] * 0.25 for rep in REPS]
        ax.scatter(differences, ys, s=16, color=MUTED, zorder=3)
        ax.errorbar(rec["mean"], y - 0.10,
                    xerr=[[rec["mean"] - lo], [hi - rec["mean"]]],
                    fmt="D", color=COLOR["S1"], markersize=4,
                    elinewidth=1.1, capsize=3, zorder=4)
        for rep, value in zip(REPS, differences):
            PLOT_ROWS["moving_mnist_paired_comparisons"].append({
                "figure": "moving_mnist_paired_comparisons", "panel": name,
                "series": "paired difference", "dataset": "moving_mnist", "role": name,
                "replication": rep, "metric_key": "physical_forecast_energy_score",
                "transformation": "S1 minus reference, matched training replication", "value": repr(value)})
        for series, value in (("mean difference", rec["mean"]), ("interval lower", lo), ("interval upper", hi)):
            PLOT_ROWS["moving_mnist_paired_comparisons"].append({
                "figure": "moving_mnist_paired_comparisons", "panel": name,
                "series": series, "dataset": "moving_mnist", "role": name,
                "replication": "", "metric_key": "physical_forecast_energy_score",
                "transformation": "paired t, df=4, individual 95% interval", "value": repr(value)})
    span = max(bounds) - min(bounds)
    ax.set_xticks([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25])
    ax.set_xlim(min(bounds) - 0.08 * span, max(bounds) + 0.08 * span)
    ax.set_ylim(-0.40, 1.42)
    ax.set_yticks([1, 0], ["vs Shared-Det", "vs AdaSSL reference"])
    ax.set_xlabel("Shared-Var minus reference (Energy Score, px/frame)")
    ax.axvline(0, color=INK, linestyle="--", linewidth=0.8, zorder=2)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [Line2D([0], [0], marker="o", linestyle="none", color=MUTED,
                      markersize=4, label="Training replications"),
               Line2D([0], [0], marker="D", color=COLOR["S1"], linewidth=1.1,
                      markersize=4, label="Mean and 95% interval")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.04),
              ncol=2, frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.3)
    fig.tight_layout(pad=0.8)
    save(fig, "moving_mnist_paired_comparisons",
         question="How does Shared-Var compare with Shared-Det and AdaSSL reference on Moving-MNIST?",
         source="analysis.json (recomputed and asserted equal)", sample_size="five paired replications per contrast",
         metric="physical_forecast_energy_score", units="px/frame", uncertainty="individual paired t 95% interval, df=4",
         note="single common scale; circles are matched-run differences; diamond and horizontal interval summarize each contrast; zero denotes equal scores")

# ----------------------------------------------------------------------------- numbers and manifest
def misc_numbers(analysis, cost):
    body = analysis["datasets"]["mpi3d"]["metrics"]["physical_forecast_energy_score"]
    num("MPSOneSZeroWinsText", str(sum(d < 0 for d in body["contrasts"]["S1-S0"]["paired_differences"])))
    mm = analysis["datasets"]["moving_mnist"]["metrics"]["physical_forecast_energy_score"]
    num("MMROneOverSOneRatio", f"{mm['roles']['R1']['mean'] / mm['roles']['S1']['mean']:.2f}")


def write_numbers():
    lines = []
    for name in sorted(NUMBERS):
        if not name.isalpha():
            raise ValueError(f"Macro name must be alphabetic: {name}")
        lines.append(f"\\newcommand{{\\V{name}}}{{{NUMBERS[name]}}}")
    (TABLES / "numbers.tex").write_text("\n".join(lines) + "\n")
    record("tables/numbers.tex", count=len(NUMBERS))


def write_plot_data():
    for figure, rows in PLOT_ROWS.items():
        with open(DATA / f"{figure}.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["figure", "panel", "series", "dataset", "role", "replication",
                                                         "metric_key", "transformation", "value"])
            writer.writeheader()
            writer.writerows(rows)


def main():
    global DIAG, REGENERATE_ILLUSTRATION, MNIST_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate-illustration", action="store_true")
    parser.add_argument("--mnist-dir", type=Path, default=ROOT / "data/concept2-mnist")
    args = parser.parse_args()
    REGENERATE_ILLUSTRATION, MNIST_DIR = args.regenerate_illustration, args.mnist_dir.resolve()
    for folder in (TABLES, FIGURES, DATA):
        folder.mkdir(parents=True, exist_ok=True)
    DIAG = load_diagnostics()
    analysis = verified_analysis()
    import yaml
    mm_cfg = yaml.safe_load(Path(register_input(ROOT / "config/campaigns/five_seed_v1/moving_mnist.yaml")).read_text())
    mpi_cfg = yaml.safe_load(Path(register_input(ROOT / "config/campaigns/five_seed_v1/mpi3d.yaml")).read_text())
    protocol = load_json(E / "frozen_protocol.json")
    review = load_json(E / "frozen_review.json")
    counts = split_counts()
    table_role_matrix(mm_cfg, mpi_cfg)
    table_splits(counts, protocol, mpi_cfg)
    table_training(mm_cfg, mpi_cfg, None)
    table_primary(analysis)
    table_mm_primary(analysis)
    table_paired(analysis)
    table_per_replication(analysis)
    table_mm_controls()
    table_mm_representation()
    table_mm_residual()
    table_mp_controls()
    table_mpi3d_main(analysis)
    table_mp_support()
    table_mp_probes()
    cost = cost_data(review)
    table_cost(cost)
    table_cost_summary(cost)
    table_cost_full(cost)
    figure_physical(analysis)
    figure_mm_paired_comparisons(analysis)
    figure_mpi3d_failure()
    figure_residual_usage()
    figure_cost(cost)
    figure_paired(analysis)
    figure_sampling_control()
    figure_representation()
    figure_readout_fidelity()
    figure_interval_coverage()
    figure_speed_strata()
    figure_dataset_protocols()
    figure_information_flow()
    misc_numbers(analysis, cost)
    write_numbers()
    write_plot_data()
    for figure in PLOT_ROWS:
        register = MANIFEST["outputs"].get(f"figures/{figure}.pdf")
        if register is not None:
            register["rows"] = len(PLOT_ROWS[figure])
    MANIFEST["inputs"] = dict(sorted(MANIFEST["inputs"].items()))
    (DATA / "manifest.json").write_text(json.dumps(MANIFEST, indent=2, sort_keys=True) + "\n")
    print(f"tables: {len(list(TABLES.glob('*.tex')))}, figures: {len(list(FIGURES.glob('*.pdf')))}, numbers: {len(NUMBERS)}")


if __name__ == "__main__":
    main()
