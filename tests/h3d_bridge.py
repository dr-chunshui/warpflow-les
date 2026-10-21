"""Bridge between warpflow and Hydro3D: export an identical restart field,
then compare the two codes step by step.

export <outdir>              write final_ctime.dat + tecbin0000.bin from
                             warpflow's freshly initialised cavity case
                             (rest + deterministic noise), ghosts rebuilt
                             to Hydro3D's 4 layers (LMR=2, pl_ex=2 -> pl=4).
compare <h3ddir> <snapdir>   load Hydro3D tecbin dumps (one per 25 steps)
                             and warpflow snapshots, print per-field max
                             differences.
"""
import struct
import sys

import numpy as np

sys.path.insert(0, ".")

from warpflow import load_case, build_topology, FlowSolver, NPL

PL_H3D = 4  # Hydro3D ghost layers with LMR=2, pl_ex=2

_NORMAL = {"west": "u", "east": "u", "south": "v", "north": "v",
           "bottom": "w", "top": "w"}


def fill_ghosts_numpy(real, name, bnd, npl, ulid):
    """Rebuild npl ghost layers around the real cells exactly like
    Hydro3D's bounds.for for the benchmark configuration (walls at
    west/east/bottom, slip at south/north, moving lid on top)."""
    n = real.shape[0]
    g = np.zeros((n + 2 * npl,) * 3)
    g[npl:npl + n, npl:npl + n, npl:npl + n] = real
    axes = {"west": 0, "east": 0, "south": 1, "north": 1,
            "bottom": 2, "top": 2}
    for side, kind in bnd.items():
        ax = axes[side]
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        for m in range(npl):
            if side in ("west", "south", "bottom"):
                gi, ri = npl - 1 - m, npl + m          # ghost, real index
            else:
                gi, ri = npl + n + m, npl + n - 1 - m
            lo[ax], hi[ax] = gi, ri
            if name == "p":
                val = g[tuple(hi)]                    # Neumann mirror
            elif kind == "moving_lid" and name == "u":
                val = ulid                            # constant lid velocity
            elif name == _NORMAL[side]:
                val = 0.0                             # solid face, no flow
            else:
                mirror = +1 if kind in ("slip", "moving_lid") else -1
                val = mirror * g[tuple(hi)]
            g[tuple(lo)] = val
    return g


def export(outdir, case="cases/cavity_serial.yaml"):
    import os

    cfg = load_case(case)
    topo = build_topology(cfg)
    sol = FlowSolver(cfg, topo, device="cpu")
    ni, nj, nk = sol.ncells
    n = ni
    dx = sol.dx

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "final_ctime.dat"), "w") as fh:
        fh.write("       0%18.12f%18.12f%18.12f\n" % (0.0, 0.0, 1.0))

    fields = {}
    for name in ("u", "v", "w", "p"):
        real = sol.f[name].numpy()[NPL:NPL + n, NPL:NPL + n, NPL:NPL + n]
        fields[name] = fill_ghosts_numpy(real, name, sol.bnd, PL_H3D,
                                         sol.ulid)
    vis = np.full_like(fields["u"], sol.rey)

    ttc = n + 2 * PL_H3D
    path = os.path.join(outdir, "tecbin0000.bin")
    with open(path, "wb") as fh:
        # Fortran unformatted sequential records: 4-byte length marker
        # before and after each record
        fh.write(struct.pack("<i", 12) + struct.pack("<3i", ttc, ttc, ttc)
                 + struct.pack("<i", 12))
        fh.write(struct.pack("<ii", 4, PL_H3D) + struct.pack("<i", 4))
        fh.write(struct.pack("<i", 12) + struct.pack("<3i", -1, -1, -1)
                 + struct.pack("<i", 12))
        xs = (np.arange(ttc) - PL_H3D + 0.5) * dx
        buf = np.empty(23)
        for k in range(ttc):
            for j in range(ttc):
                for i in range(ttc):
                    buf[:] = 0.0
                    buf[0], buf[1], buf[2] = xs[i], xs[j], xs[k]
                    buf[3] = fields["p"][i, j, k]
                    buf[6] = vis[i, j, k]
                    buf[8] = fields["u"][i, j, k]
                    buf[11] = fields["v"][i, j, k]
                    buf[14] = fields["w"][i, j, k]
                    fh.write(struct.pack("<i", 184))
                    fh.write(struct.pack("<23d", *buf))
                    fh.write(struct.pack("<i", 184))
    print("wrote", path, "ttc =", ttc)


def read_tecbin(path, n, raw=False):
    """Return (u, v, w, p) arrays from a Hydro3D tecbin dump.

    raw=False: the classic [pl:pl+n] slice.  raw=True: the full (n+2*pl)^3
    plane set with NO slicing -- tecbin plane i sits at x=(i-3)*dx, the same
    coordinate law as warpflow's full local array."""
    with open(path, "rb") as fh:
        fh.read(20)   # record 1: marker + (tti,ttj,ttk) + marker
        fh.read(4)    # record 2 leading marker
        pl = struct.unpack("<i", fh.read(4))[0]
        fh.read(4)    # record 2 trailing marker
        fh.read(20)   # record 3: marker + (inind,jnind,knind) + marker
        tti = n + 2 * pl
        assert pl == PL_H3D, pl
        arr = np.empty((tti, tti, tti, 23))
        for k in range(tti):
            for j in range(tti):
                for i in range(tti):
                    fh.read(4)
                    arr[i, j, k] = struct.unpack("<23d", fh.read(8 * 23))
                    fh.read(4)
        if raw:
            return {name: arr[..., col].copy()
                    for name, col in (("u", 8), ("v", 11), ("w", 14), ("p", 3))}
        s = (slice(pl, pl + n),) * 3
        return {name: arr[s[0], s[1], s[2], col].copy()
                for name, col in (("u", 8), ("v", 11), ("w", 14), ("p", 3))}


def compare(h3ddir, snapdir, n=80, step=25, upto=100):
    import os

    # Plane alignment (established empirically, see tests notes): the tecbin
    # Fortran law is x_F(i) = (i-4)*dx, so 0-based file plane s sits at
    # x=(s-3)*dx.  Hydro3D's real velocity points are file planes 4..82
    # (F5..F83, the 79 interior faces; the wall faces are the ghost planes
    # 3 and 83, zeroed every bound call).  Warpflow's full local array keeps
    # the same physical planes at indices 3..81 (full i at x=(i-2)*dx).
    # Pairing: h3d[s] <-> mine[s-1].
    lo, hi = 4, 83   # h3d planes 4..82 = warpflow full 3..81
    for it in range(step, upto + 1, step):
        hp = os.path.join(h3ddir, "h3d_it%03d.bin" % it)
        sp = os.path.join(snapdir, "snap_%06d_r000.npz" % it)
        if not (os.path.exists(hp) and os.path.exists(sp)):
            print("it %3d: missing (%s / %s)" % (it, hp, sp))
            continue
        h3d = read_tecbin(hp, n, raw=True)
        z = np.load(sp)
        print("it %3d" % it)
        for name in ("u", "v", "w", "p"):
            mine = z[name][lo - 1:hi - 1, lo - 1:hi - 1, lo - 1:hi - 1].copy()
            href = h3d[name][lo:hi, lo:hi, lo:hi].copy()
            if name == "p":
                # the Poisson problem with a periodic side is singular: the
                # additive constant is iteration-path dependent, so compare
                # the pressure structure with both fields demeaned
                mine -= mine.mean()
                href -= href.mean()
            d = np.abs(mine - href)
            i, j, k = np.unravel_index(np.argmax(d), d.shape)
            print("  %s  max|d| %.3e at (%3d,%3d,%3d)"
                  "  mine %.6f  h3d %.6f"
                  % (name, d.max(), i + lo, j + lo, k + lo,
                     mine[i, j, k], href[i, j, k]))
            print("      rms %.3e" % np.sqrt((d ** 2).mean()))


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(sys.argv[2])
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3])
