"""Snapshot output: one .npz per rank holding the local block (float64)."""
from pathlib import Path

import numpy as np

from .topology import NPL


def write_snapshot(sol, itime):
    outdir = Path(sol.cfg.output.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ni, nj, nk = sol.ncells
    data = {}
    for name in ("u", "v", "w", "p"):
        # full local block incl. ghost layers; with a single rank, plane
        # index i sits at x=(i-NPL)*dx (the west wall face is i=NPL).
        # Saving everything keeps post-processing flexible.
        data[name] = sol.f[name].numpy()
    path = outdir / ("snap_%06d_r%03d.npz" % (itime, sol.topo.rank))
    np.savez_compressed(
        path, **data,
        ctime=np.float64(sol.ctime), dt=np.float64(sol.dt),
        offsets=np.array(sol.topo.global_offsets, dtype=np.int64),
        ncells=np.array(sol.ncells, dtype=np.int64),
        gshape=np.array((sol.cfg.grid.nx, sol.cfg.grid.ny,
                         sol.cfg.grid.nz), dtype=np.int64))
    return path
