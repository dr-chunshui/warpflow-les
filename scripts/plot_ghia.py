"""Compare the steady cavity centreline profiles against Ghia et al. (1982).

Extracts u along the vertical centreline (x = y = 0.5) and w along the
horizontal centreline (y = z = 0.5) from a warpflow snapshot, plots them
against the benchmark data, and writes the extracted profiles to CSV.

Usage:
    python scripts/plot_ghia.py [--snapshot output_cavity/snap_0050000_r000.npz]
                                [--ghia-dir <Benchmark_CavityFlow>]
                                [--out output_cavity]
"""
import argparse
import csv
import os

import numpy as np

# dataviz palette (validated: CVD dE 24.7, contrast >= 3:1 on #fcfcfb)
COLOR_MINE = "#2a78d6"
COLOR_REF = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#1a1a17"
INK_2 = "#6e6c66"
GRID = "#e7e5e1"

N = 80
DX = 1.0 / N


def read_ghia(path):
    data = np.loadtxt(path, skiprows=1)
    return data[:, 0], data[:, 1]


def centreline_profiles(npz_path):
    """u(z) on the vertical centreline, w(x) on the horizontal one.

    Staggered-grid geometry (full-array indexing, NPL offset included in
    the +2): cell centres sit at (c+0.5)*dx (full index c+2), u x-faces at
    f*dx (full index f+2), w z-faces likewise.  x = 0.5 is exactly u face
    f = 40; y = z = 0.5 lie between cell centres 39 and 40 -- take 40.
    """
    z = np.load(npz_path)
    u = z["u"]
    w = z["w"]
    zc = (np.arange(N) + 0.5) * DX            # cell-centre coordinates
    u_prof = u[42, 42, 2:2 + N]                # x face 40, y centre 40
    w_prof = w[2:2 + N, 42, 42]                # z face 40, y centre 40
    return zc, u_prof, zc, w_prof


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.8)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default="output_cavity/snap_0050000_r000.npz")
    ap.add_argument("--ghia-dir",
                    default="/home/chunshui/Workspace/hydro3D/"
                            "hydro3D-7_Hydro3D-STABLE/Benchmark_CavityFlow")
    ap.add_argument("--out", default="output_cavity")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    zc, u_prof, xc, w_prof = centreline_profiles(args.snapshot)
    gx_u, gu = read_ghia(os.path.join(args.ghia_dir, "Ghia_U.dat"))
    gx_w, gw = read_ghia(os.path.join(args.ghia_dir, "Ghia_W.dat"))

    # numeric agreement summary
    u_err = np.sqrt(np.mean((np.interp(gx_u, zc, u_prof) - gu) ** 2))
    w_err = np.sqrt(np.mean((np.interp(gx_w, xc, w_prof) - gw) ** 2))
    print("u(z) vs Ghia Re=400: rms = %.4f" % u_err)
    print("w(x) vs Ghia Re=400: rms = %.4f" % w_err)

    # CSV of the extracted profiles
    with open(os.path.join(args.out, "centreline.csv"), "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["z", "u_centerline", "x", "w_centerline"])
        for row in zip(zc, u_prof, xc, w_prof):
            wr.writerow(["%.6f" % row[0], "%.8f" % row[1],
                         "%.6f" % row[2], "%.8f" % row[3]])

    # figure: two panels, shared styling, one legend on the first panel
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)

    ax1.plot(u_prof, zc, color=COLOR_MINE, linewidth=2.0,
             label="warpflow-les (GPU)", zorder=3)
    ax1.plot(gu, gx_u, linestyle="none", marker="o", markersize=8,
             markerfacecolor=SURFACE, markeredgecolor=COLOR_REF,
             markeredgewidth=2.0, label="Ghia et al. (Re = 400)", zorder=4)
    ax1.set_xlabel("u", color=INK, fontsize=11)
    ax1.set_ylabel("z", color=INK, fontsize=11)
    ax1.set_title("vertical centreline (x = 0.5)", color=INK,
                  fontsize=11, pad=10)

    ax2.plot(xc, w_prof, color=COLOR_MINE, linewidth=2.0, zorder=3)
    ax2.plot(gx_w, gw, linestyle="none", marker="o", markersize=8,
             markerfacecolor=SURFACE, markeredgecolor=COLOR_REF,
             markeredgewidth=2.0, zorder=4)
    ax2.set_xlabel("x", color=INK, fontsize=11)
    ax2.set_ylabel("w", color=INK, fontsize=11)
    ax2.set_title("horizontal centreline (z = 0.5)", color=INK,
                  fontsize=11, pad=10)

    for ax in (ax1, ax2):
        style_axis(ax)
    ax1.legend(loc="lower right", frameon=False, fontsize=9,
               labelcolor=INK)

    fig.suptitle("Lid-driven cavity, 80$^3$, Re = 400, t = 15.0",
                 color=INK, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(args.out, "ghia_comparison.png")
    fig.savefig(path, facecolor=SURFACE)
    print("wrote", path)
    print("wrote", os.path.join(args.out, "centreline.csv"))


if __name__ == "__main__":
    main()
