"""
Plot hydrogen-bond counts vs time from GROMACS .xvg files (gmx hbond -num).

Usage:
    python plot_hbonds.py                 # uses the FILES list below
    python plot_hbonds.py -o my_plot.png  # custom output name

Edit the FILES list below: (path to .xvg, legend label, colour).
Requires: numpy, matplotlib
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# EDIT THIS LIST: (file, legend label, colour). Remove a row to drop a line.
# ----------------------------------------------------------------------
FILES = [
    ("hbond_membrane1.xvg",         "Membrane 1",                   "tab:blue"),
    ("hbond_ox__membrane2.xvg",     "Oxidised Membrane 2 / CPS09A", "tab:green"),
    ("hbond_membrane3_1_.xvg",      "POPE / Membrane 3",            "tab:purple"),
    ("hbondEPS09A_membrane4.xvg",   "Oxidised Membrane 4 / EPS09A", "tab:red"),
]


def load_xvg(path):
    """Read an .xvg file, skipping '#' and '@' header lines. Returns (time_ps, values)."""
    data = np.loadtxt(path, comments=["#", "@"])
    return data[:, 0], data[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", default="hbond_plot_4_membranes.png")
    ap.add_argument("--ps", action="store_true", help="keep time in ps instead of ns")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=200)
    tmax = 0
    ymax = 0

    print(f"{'Label':<32}{'min':>5}{'max':>5}{'mean':>8}{'final':>7}")
    for path, label, colour in FILES:
        t, y = load_xvg(path)
        if not args.ps:
            t = t / 1000.0
        ax.plot(t, y, "-o", ms=4, lw=1.8, color=colour, label=label)
        tmax, ymax = max(tmax, t.max()), max(ymax, y.max())
        print(f"{label:<32}{y.min():>5.0f}{y.max():>5.0f}{y.mean():>8.2f}{y[-1]:>7.0f}")

    ax.set_xlabel("Time (ps)" if args.ps else "Time (ns)")
    ax.set_ylabel("Number of hydrogen bonds")
    ax.set_title("Hydrogen bonds over time")
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, ymax + 1)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
