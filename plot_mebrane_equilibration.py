"""
Temperature, pressure and area-per-lipid plot for each membrane (single plot, 3 y-axes).
Input : GROMACS .xvg files from `gmx energy` (must contain Temperature, Pressure, Box-X, Box-Y)
Output: <key>_T_P_Area_single.png for every membrane
Usage : python membrane_plots.py   (edit the CONFIG block below)
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ---------------- CONFIG ----------------
DATA_DIR = "."          # folder containing the .xvg files
OUT_DIR = "."           # folder for the PNGs
T_MAX = 500             # ps, plot 0..T_MAX
TARGET_T = 310          # K (only sets the temperature axis range)
PRESSURE_WINDOW = 25    # points for running average (25 points x 2 ps = 50 ps)

MEMBRANES = {
    # key: (xvg file, title, lipids in ONE (upper) leaflet)
    "M1": ("step7_10__PRF-1_Membrane-1.xvg",  "Membrane 1 (POPC)",          128),
    "M2": ("step7_10_ox__membrane-2.xvg",     "Membrane 2 (oxidised POPC)", 128),  # 26 ox + 102 POPC
    "M3": ("productionstep7_10_membrane3.xvg", "Membrane 3 (POPE)",          102),
    "M4": ("equilibration_membrane4.xvg",     "Membrane 4 (oxidised POPE)", 102),
}
# ----------------------------------------


def read_xvg(path):
    """Read an .xvg file into a DataFrame (column names from the @ s# legend lines)."""
    legends = []
    with open(path) as f:
        for line in f:
            m = re.match(r'@ s\d+ legend "(.*)"', line)
            if m:
                legends.append(m.group(1))
    data = np.loadtxt(path, comments=("#", "@"))
    return pd.DataFrame(data[:, 1:], index=data[:, 0], columns=legends)


def make_plot(key, fname, title, n_leaflet):
    df = read_xvg(f"{DATA_DIR}/{fname}")
    df = df[df.index <= T_MAX]
    t = df.index

    # Area per lipid (A^2) = Box-X * Box-Y (nm^2) * 100 / lipids in one leaflet
    apl = df["Box-X"] * df["Box-Y"] * 100 / n_leaflet
    pressure = df["Pressure"].rolling(PRESSURE_WINDOW, center=True, min_periods=1).mean()

    c_t, c_p, c_a = "#d62728", "#2ca02c", "#1f77b4"
    fig, ax_t = plt.subplots(figsize=(10, 5.5))
    ax_p = ax_t.twinx()
    ax_a = ax_t.twinx()
    ax_a.spines["right"].set_position(("outward", 60))

    ax_t.plot(t, df["Temperature"], color=c_t, lw=0.6, alpha=0.8)
    ax_t.set_ylabel("Temperature (K)", color=c_t)
    ax_t.set_ylim(TARGET_T - 10, TARGET_T + 10)

    ax_p.plot(t, pressure, color=c_p, lw=1.3)
    ax_p.set_ylabel("Pressure (bar), 50 ps running avg", color=c_p)
    ax_p.set_ylim(-300, 300)

    ax_a.plot(t, apl, color=c_a, lw=1.5)
    ax_a.set_ylabel("Area per lipid (Å²)", color=c_a)
    mid, half = (apl.max() + apl.min()) / 2, (apl.max() - apl.min()) * 1.5 + 0.5
    ax_a.set_ylim(mid - half, mid + half)

    ax_t.set_xlim(0, T_MAX)
    ax_t.set_xlabel("Time (ps)")
    ax_t.set_title(title, fontsize=11)
    ax_t.grid(alpha=0.3)
    ax_t.legend(
        handles=[Line2D([], [], color=c, label=l) for c, l in
                 [(c_t, "Temperature"), (c_p, "Pressure (50 ps avg)"), (c_a, "Area per lipid (Å²)")]],
        fontsize=8, loc="lower right", ncol=3)

    fig.tight_layout()
    out = f"{OUT_DIR}/{key}_T_P_Area_single.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"{key}: saved {out} | mean APL = {apl.mean():.1f} Å²")


if __name__ == "__main__":
    for key, (fname, title, n) in MEMBRANES.items():
        make_plot(key, fname, title, n)
