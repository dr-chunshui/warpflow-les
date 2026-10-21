"""Diagnose which part of the multigrid cycle fails: smoother-only
convergence vs 2-level vs 3-level V-cycles."""
import sys

import numpy as np
import warp as wp

sys.path.insert(0, ".")

from warpflow.config import load_case
from warpflow.topology import build_topology
from warpflow.multigrid import Multigrid
from tests.test_mg import residual, make_rhs


def run(mg, ncyc, label):
    lev0 = mg.levels[0]
    lev0.phi.zero_()
    mg._refill(lev0)
    r0 = residual(lev0)
    for it in range(ncyc):
        mg._vcycle()
        r = residual(lev0)
        if it < 3 or it == ncyc - 1:
            print("%-10s cycle %2d  res %.3e  factor %.2f"
                  % (label, it, r, r / max(residual_prev[0], 1e-300)
                     if residual_prev else 0), end="")
            print()
        residual_prev.clear()
        residual_prev.append(r)


residual_prev = []


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "cases/cavity_serial.yaml"
    cfg = load_case(case)
    topo = build_topology(cfg)
    dev = "cuda:0" if wp.is_cuda_available() else "cpu"
    mg = Multigrid(cfg, topo, dev, 1.0 / cfg.grid.nx, 1.0 / cfg.grid.ny,
                   1.0 / cfg.grid.nz)
    lev0 = mg.levels[0]
    make_rhs(lev0)

    # 1) smoother only
    lev0.phi.zero_()
    mg._refill(lev0)
    r_prev = residual(lev0)
    print("smoother-only (RBGS sweeps), res0 %.3e" % r_prev)
    for it in range(6):
        mg._relax(0, 10)
        r = residual(lev0)
        print("  sweeps %3d  res %.3e  factor %.3f"
              % ((it + 1) * 10, r, r / r_prev))
        r_prev = r

    # 2) V-cycles at 3 levels (as configured)
    run(mg, 6, "3-level")

    # 3) V-cycles at 2 levels
    mg.levels = mg.levels[:2]
    make_rhs(lev0)
    run(mg, 6, "2-level")


if __name__ == "__main__":
    main()
