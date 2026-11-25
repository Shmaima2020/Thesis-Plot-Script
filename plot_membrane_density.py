"""
Density profile plots (lipid / peptide / water) for GROMACS `gmx density` .xvg files.

Usage:
    python plot_membrane_density.py                 # files in current folder
    python plot_membrane_density.py path/to/data    # files in another folder
Edit the MEMBRANES dictionary below if your file names or labels change.
Outputs (per membrane, 300 dpi PNG): *_full.png, *_dual_axis.png, *_two_panel.png
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

DATA_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT_DIR = Path("plots"); OUT_DIR.mkdir(exist_ok=True)

# name -> (title, lipid label, lipid file, peptide file, water file)
MEMBRANES = {
    "membrane1": ("Membrane 1 (POPC)", "Lipid (POPC)",
                  "POPC_LIPID_Density-membrane-1.xvg", "PEPTIDE_MEMBRANE1.xvg", "water_MEMBRANE1.xvg"),
    "membrane2": ("Membrane 2 (Oxidised POPC)", "Oxidised lipid (CPS09A)",
                  "lipid_ox_membrane2.xvg", "PEPTIDE_OXIDISED_POPC__membrane2__1_.xvg", "water_OX_membrane2.xvg"),
    "membrane3": ("Membrane 3 (POPE)", "Lipid (POPE)",
                  "POPE_LIPID_POPE__membrane3.xvg", "PEPTIDE_POPE_membrane3.xvg", "water_POPE__membrane3.xvg"),
    "membrane4": ("Membrane 4 (Oxidised POPE)", "Oxidised lipid (EPS09A)",
                  "OX_POPE_EPSO09A_LIPID_membrane4.xvg", "PEPTIDE_0X_POPE_EPS09A_membrane4.xvg",
                  "water_OX_POPE_EPSO9A_membrane4.xvg"),
}

COL = dict(lipid="#3274a1", pep="#e1812c", water="#3a923a")
PEPTIDE_PEAK_PROMINENCE = 0.15   # fraction of the peptide maximum; raise to label fewer peaks
plt.rcParams.update({"font.size": 12})


def load_xvg(path):
    """Read an .xvg file, skipping the # and @ header lines. Returns an (N, 2) array."""
    return np.loadtxt(path, comments=["#", "@"])[:, :2]


def leaflet_peaks(lip):
    """Highest lipid point on each side of the bilayer center (two leaflets)."""
    x, y = lip[:, 0], lip[:, 1]
    inside = x[y > 0.3 * y.max()]
    mid = (inside[0] + inside[-1]) / 2
    peaks = []
    for mask in (x < mid, x >= mid):
        i = y[mask].argmax()
        peaks.append((x[mask][i], y[mask][i]))
    return peaks


def peptide_peaks(pep):
    idx, _ = find_peaks(pep[:, 1], prominence=PEPTIDE_PEAK_PROMINENCE * pep[:, 1].max())
    return [(pep[i, 0], pep[i, 1]) for i in idx]


def water_half_crossings(wat):
    """x positions where water density crosses half of its bulk value."""
    half = 0.5 * np.percentile(wat[:, 1], 95)
    above = wat[:, 1] > half
    return wat[:-1, 0][np.diff(above.astype(int)) != 0]


def make_plots(key, title, lip_label, lip, pep, wat):
    xmax = lip[-1, 0] + 0.1
    lpk, ppk = leaflet_peaks(lip), peptide_peaks(pep)
    ymax = 1450

    # 1) Full range, all three curves on one axis, lipid peaks labeled
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot(lip[:, 0], lip[:, 1], COL["lipid"], lw=2.2, label=lip_label)
    ax.plot(pep[:, 0], pep[:, 1], COL["pep"], lw=2.2, label="Peptide (PG-1)")
    ax.plot(wat[:, 0], wat[:, 1], COL["water"], lw=2.2, label="Water")
    for x, y in lpk:
        ax.annotate(f"Lipid peak\n{x:.2f} nm, {y:.0f}", (x, y), xytext=(x, y + 170),
                    ha="center", arrowprops=dict(arrowstyle="->"))
    ax.set(xlabel="Bilayer Normal (nm)", ylabel="Density (g/L)", xlim=(0, xmax), ylim=(0, ymax))
    ax.set_title(f"Density Distribution – {title}\nFull range", fontweight="bold", fontsize=16)
    ax.grid(alpha=.3); ax.legend(loc="upper center", ncol=3)
    fig.tight_layout(); fig.savefig(OUT_DIR / f"{key}_density_full.png", dpi=300); plt.close(fig)

    # 2) Dual axis: lipid + water on the left, peptide on the right, peptide peaks labeled
    fig, ax = plt.subplots(figsize=(10, 7)); ax2 = ax.twinx()
    h1 = ax.plot(lip[:, 0], lip[:, 1], COL["lipid"], lw=2.2, label=f"{lip_label} (left)")
    h2 = ax.plot(wat[:, 0], wat[:, 1], COL["water"], lw=2.2, label="Water (left)")
    h3 = ax2.plot(pep[:, 0], pep[:, 1], COL["pep"], lw=2.5, marker="o", ms=4, label="Peptide PG-1 (right)")
    for x, y in ppk:
        ax2.annotate(f"{x:.2f} nm", (x, y), xytext=(x, y + 0.2 * pep[:, 1].max()), ha="center",
                     fontsize=10, color=COL["pep"], arrowprops=dict(arrowstyle="-", color=COL["pep"]))
    ax.set(xlabel="Bilayer Normal (nm)", ylabel="Lipid / Water Density (g/L)", xlim=(0, xmax), ylim=(0, ymax))
    ax2.set_ylabel("Peptide Density (g/L)", color=COL["pep"])
    ax2.set_ylim(0, 2.2 * pep[:, 1].max()); ax2.tick_params(axis="y", colors=COL["pep"])
    ax.set_title(f"Density Distribution – {title}\nDual axis", fontweight="bold", fontsize=16)
    ax.grid(alpha=.3)
    hs = h1 + h2 + h3
    ax.legend(hs, [h.get_label() for h in hs], loc="upper center", ncol=3)
    fig.tight_layout(); fig.savefig(OUT_DIR / f"{key}_density_dual_axis.png", dpi=300); plt.close(fig)

    # 3) Two panels: full range and 0-100 g/L zoom
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, zoom in zip(axes, (False, True)):
        ax.plot(lip[:, 0], lip[:, 1], COL["lipid"], lw=2.2, label=lip_label)
        ax.plot(pep[:, 0], pep[:, 1], COL["pep"], lw=2.2, label="Peptide (PG-1)")
        ax.plot(wat[:, 0], wat[:, 1], COL["water"], lw=2.2, label="Water")
        ax.set(xlabel="Bilayer Normal (nm)", ylabel="Density (g/L)", xlim=(0, xmax)); ax.grid(alpha=.3)
        ax.set_ylim(0, 100 if zoom else 1250)
        ax.set_title("Zoom (0–100 g/L)" if zoom else "Full range", fontweight="bold")
    axes[0].legend(loc="upper center", ncol=3)
    fig.suptitle(f"Density Distribution – {title}", fontweight="bold", fontsize=17)
    fig.tight_layout(); fig.savefig(OUT_DIR / f"{key}_density_two_panel.png", dpi=300); plt.close(fig)

    return lpk, ppk


def main():
    print(f"{'system':<10}{'lipid peaks (nm, g/L)':<34}{'peptide max (nm, g/L)':<24}{'water-free width (nm)'}")
    for key, (title, lip_label, f_lip, f_pep, f_wat) in MEMBRANES.items():
        lip, pep, wat = (load_xvg(DATA_DIR / f) for f in (f_lip, f_pep, f_wat))
        lpk, ppk = make_plots(key, title, lip_label, lip, pep, wat)
        cross = water_half_crossings(wat)
        pmax = pep[pep[:, 1].argmax()]
        lp = ", ".join(f"({x:.2f}, {y:.0f})" for x, y in lpk)
        pm = f"({pmax[0]:.2f}, {pmax[1]:.1f})"
        print(f"{key:<10}{lp:<34}{pm:<24}{cross[-1] - cross[0]:.2f}")
    print(f"\nPlots saved in: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
