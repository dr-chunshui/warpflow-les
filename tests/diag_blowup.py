"""Locate the blow-up: run the cavity case step by step, report where the
velocity maximum sits and how strong the 2dx sawtooth content is."""
import sys

import numpy as np
import warp as wp

sys.path.insert(0, ".")

from warpflow import load_case, build_topology, FlowSolver, NPL


def stats(sol):
    ni, nj, nk = sol.ncells
    u = sol.f["u"].numpy()[NPL:NPL + ni, NPL:NPL + nj, NPL:NPL + nk]
    v = sol.f["v"].numpy()[NPL:NPL + ni, NPL:NPL + nj, NPL:NPL + nk]
    w = sol.f["w"].numpy()[NPL:NPL + ni, NPL:NPL + nj, NPL:NPL + nk]
    au = np.abs(u)
    i, j, k = np.unravel_index(np.argmax(au), au.shape)
    # sawtooth measure: curvature energy vs gradient energy
    curv = np.abs(u[2:, 1:-1, 1:-1] - 2 * u[1:-1, 1:-1, 1:-1]
                  + u[:-2, 1:-1, 1:-1]).sum()
    grad = np.abs(np.diff(u, axis=0)).sum() + 1e-300
    return (au.max(), (i, j, k), curv / grad,
            np.abs(v).max(), np.abs(w).max())


def main():
    cfg = load_case(sys.argv[1] if len(sys.argv) > 1
                    else "cases/cavity_serial.yaml")
    nsteps = int(sys.argv[2]) if len(sys.argv) > 2 else 215
    topo = build_topology(cfg)
    dev = "cuda:0" if wp.is_cuda_available() else "cpu"
    sol = FlowSolver(cfg, topo, device=dev)
    dx = sol.dx
    for it in range(1, nsteps + 1):
        info = sol.step()
        if it % 10 == 0 or (it > 180 and it % 2 == 0):
            umax, pos, saw, vmax, wmax = stats(sol)
            print("it %3d umax %.3f at (%3d,%3d,%3d) x=%.3f y=%.3f z=%.3f"
                  "  saw %.3f  vmax %.3f wmax %.3f  dt %.2e cfl %.2f"
                  % (it, umax, pos[0], pos[1], pos[2],
                     (pos[0] + 0.5) * dx, (pos[1] + 0.5) * dx,
                     (pos[2] + 0.5) * dx, saw, vmax, wmax,
                     info["dt"], info["cfl"]), flush=True)


if __name__ == "__main__":
    main()
