"""
Energy minimization plots for the four membranes (GROMACS .xvg files).

Usage:
    python plot_minimization.py [data_dir] [output_dir]

data_dir   folder containing the .xvg files (default: current folder)
output_dir folder for the PNG files (default: current folder)

Requires: numpy, matplotlib
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (label, short name for legend, .xvg file, colour, output name)
MEMBRANES = [
    ("Membrane - 1 (POPC)",                 "POPC",              "MINIMISATION_MEM1.XVG",                 "#1f77b4", "membrane1_POPC"),
    ("Membrane - 2 (Ox-POPC / CPS09A)",     "Ox-POPC / CPS09A",  "Membrane2_step6_0_ox_minimization.xvg", "#2ca02c", "membrane2_ox_POPC"),
    ("Membrane - 3 (POPE)",                 "POPE",              "minimization_membrane3.xvg",            "#9467bd", "membrane3_POPE"),
    ("Membrane - 4 (Ox-POPE / EPS09A)",     "Ox-POPE / EPS09A",  "minimization_membrane4.xvg",            "#d62728", "membrane4_ox_POPE"),
]


def load_xvg(path):
    """Read an .xvg file; return array of (step, potential energy).
    Lines starting with # or @ are comments/metadata and are skipped."""
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line[0] in "#@":
            continue
        parts = line.split()
        rows.append((float(parts[0]), float(parts[1])))
    return np.array(rows)


def style_axes(ax, title):
    ax.set_title(title, fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Minimization step", fontsize=14)
    ax.set_ylabel("Potential energy (kJ mol$^{-1}$)", fontsize=14)
    ax.grid(True, ls="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda v, _: f"{int(v)}".replace("-", "\u2212")))
    ax.tick_params(labelsize=12)


BOX = dict(boxstyle="round,pad=0.5", fc="white", ec="#bbb", alpha=0.9)

data = {m[0]: load_xvg(DATA_DIR / m[2]) for m in MEMBRANES}

# ---- one plot per membrane ----
for label, short, _, colour, name in MEMBRANES:
    d = data[label]
    initial, final = d[0, 1], d[-1, 1]
    reduction = initial - final            # energy reduction = initial - final

    fig, ax = plt.subplots(figsize=(11.45, 7.37), dpi=100)
    ax.plot(d[:, 0], d[:, 1], color=colour, lw=2.6, label=short)
    style_axes(ax, f"Energy Minimization Profile: {label}")
    ax.legend(loc="center right", fontsize=12)
    ax.text(0.985, 0.96,
            f"Energy range = {initial:,.1f} to {final:,.1f} kJ/mol\n"
            f"Energy reduction = {reduction:,.1f} kJ/mol",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=11, bbox=BOX, linespacing=1.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"minimization_{name}.png")
    plt.close(fig)
    print(f"{label}: initial {initial:,.1f}, final {final:,.1f}, reduction {reduction:,.1f}")

# ---- all membranes in one plot ----
# The text box sits outside the axes (to the right) so it never covers a curve.
fig, ax = plt.subplots(figsize=(15, 7.37), dpi=100)
fig.subplots_adjust(left=0.095, right=0.645, top=0.92, bottom=0.12)
blocks = []
for label, short, _, colour, name in MEMBRANES:
    d = data[label]
    ax.plot(d[:, 0], d[:, 1], color=colour, lw=2.4, label=label)
    initial, final = d[0, 1], d[-1, 1]
    blocks.append(f"{label}\n"
                  f"  - Energy range: {initial:,.1f} to {final:,.1f} kJ/mol\n"
                  f"  - Energy reduction: {initial - final:,.1f} kJ/mol")
style_axes(ax, "Energy Minimization Profile: All Membranes")
ax.legend(loc="upper right", fontsize=11)
fig.text(0.655, 0.52, "\n\n".join(blocks), ha="left", va="center",
         fontsize=10.5, bbox=BOX, linespacing=1.35)
fig.savefig(OUT_DIR / "minimization_all_membranes.png")
plt.close(fig)
print("Saved plots to", OUT_DIR.resolve())
