import numpy as np
import matplotlib.pyplot as plt

files = {
    "POPC": "rmsd_membrane1.xvg",
    "Ox-POPC": "rmsd_membrane2.xvg",
    "POPE": "rmsd_membrane3.xvg",
    "Ox-POPE": "rmsd_membrane4.xvg",
}


colors = {
    "POPC": "#2C7DA0",   # blue
    "Ox-POPC": "#B33A3A",   # red
    "POPE": "#3A9145",   # green
    "Ox-POPE": "#7A5A9E",   # purple
}

def read_xvg(filename):
    """
    Read a GROMACS .xvg file.

    Returns:
        time : numpy array
            Time in ns
        rmsd : numpy array
            RMSD in nm
    """

    time = []
    rmsd = []
    with open(filename, "r") as file:
        for line in file:
            if line.startswith("#") or line.startswith("@"):
                continue
            if not line.strip():
                continue
            columns = line.split()
            if len(columns) >= 2:
                time.append(float(columns[0]))
                rmsd.append(float(columns[1]))
    return np.array(time), np.array(rmsd)

plt.figure(figsize=(12, 7))

for membrane, filename in files.items():
    time, rmsd = read_xvg(filename)
    print(f"{membrane}:")
    print(f"    File: {filename}")
    print(f"    Number of points: {len(time)}")
    print(f"    Time range: {time.min():.2f} - {time.max():.2f} ns")
    print(f"    RMSD range: {rmsd.min():.4f} - {rmsd.max():.4f} nm")
    print()

    plt.plot(time, rmsd, label=membrane, color=colors[membrane], linewidth=2.2)

plt.xlabel("Time (ns)", fontsize=16, fontweight="bold")
plt.ylabel("RMSD (nm)", fontsize=16, fontweight="bold")
plt.title("RMSD Comparison of All Four Membrane Systems", fontsize=20, fontweight="bold", pad=15)
plt.legend(fontsize=13, frameon=False, loc="upper left")
plt.xticks(np.arange(0, 1.01, 0.2), fontsize=13)
plt.yticks(fontsize=13)
plt.grid(True, alpha=0.25, linewidth=0.8)
ax = plt.gca()
ax.spines["top"].set_linewidth(1.5)
ax.spines["bottom"].set_linewidth(1.5)
ax.spines["left"].set_linewidth(1.5)
ax.spines["right"].set_linewidth(1.5)
plt.tight_layout()
plt.savefig("RMSD_comparison_all_four_membranes.png", dpi=600, bbox_inches="tight")
plt.show()