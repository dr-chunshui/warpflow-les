"""Standalone multigrid convergence test: random Neumann rhs, print the
max-residual after each V-cycle.  A healthy geometric MG drops the residual
by 1-2 orders per cycle on these grids."""
import sys

import numpy as np
import warp as wp

sys.path.insert(0, ".")

from warpflow.config import load_case
from warpflow.topology import build_topology
from warpflow.multigrid import Multigrid


def residual(lev):
    phi = lev.phi.numpy()
    rhs = lev.rhs.numpy()
    aw, ae = lev.aw.numpy(), lev.ae.numpy()
    as_, an = lev.as_.numpy(), lev.an.numpy()
    ab, at, ap = lev.ab.numpy(), lev.at.numpy(), lev.ap.numpy()
    n = lev.n3[0]
    s = slice(1, n + 1)
    r = (rhs[s, s, s]
         - (ap[s, s, s] * phi[s, s, s]
            + aw[s, s, s] * phi[0:n, s, s]
            + ae[s, s, s] * phi[2:n + 2, s, s]
            + as_[s, s, s] * phi[s, 0:n, s]
            + an[s, s, s] * phi[s, 2:n + 2, s]
            + ab[s, s, s] * phi[s, s, 0:n]
            + at[s, s, s] * phi[s, s, 2:n + 2]))
    return np.abs(r).max()


def make_rhs(lev0):
    rng = np.random.default_rng(0)
    h = rng.standard_normal(lev0.rhs.shape)
    h[0, :, :] = 0.0
    h[-1, :, :] = 0.0
    h[:, 0, :] = 0.0
    h[:, -1, :] = 0.0
    h[:, :, 0] = 0.0
    h[:, :, -1] = 0.0
    h[1:-1, 1:-1, 1:-1] -= h[1:-1, 1:-1, 1:-1].mean()
    lev0.rhs.assign(h.astype(np.float64))


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "cases/cavity_serial.yaml"
    ncyc = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    cfg = load_case(case)
    topo = build_topology(cfg)
    dev = "cuda:0" if wp.is_cuda_available() else "cpu"
    n = topo.local_ncells[0]
    mg = Multigrid(cfg, topo, dev, 1.0 / cfg.grid.nx, 1.0 / cfg.grid.ny,
                   1.0 / cfg.grid.nz)
    lev0 = mg.levels[0]

    make_rhs(lev0)
    lev0.phi.zero_()
    mg._refill(lev0)

    print("grid %d^3, levels %d" % (n, len(mg.levels)))
    for it in range(ncyc):
        mg._vcycle()
        print("cycle %2d  res0 %.6e" % (it, residual(mg.levels[0])))


if __name__ == "__main__":
    main()
