"""CLI entry point.

Usage:
    python run.py --case cases/cavity.yaml [--device auto] [--steps N]

Serial (single MPI rank) or `mpirun -np 8 python run.py --case ...` for a
decomposed run matching mpi.domains in the case file.
"""
import argparse
import time

import warp as wp

from mpi4py import MPI

from warpflow import load_case, build_topology, FlowSolver
from warpflow.io import write_snapshot


def main():
    ap = argparse.ArgumentParser(description="warpflow-les solver")
    ap.add_argument("--case", required=True, help="case YAML file")
    ap.add_argument("--device", default="auto",
                    help="warp device: auto | cpu | cuda:0")
    ap.add_argument("--steps", type=int, default=None,
                    help="override time.n_steps from the case file")
    ap.add_argument("--outdir", default=None,
                    help="override output.outdir from the case file")
    args = ap.parse_args()

    cfg = load_case(args.case)
    if args.outdir:
        cfg.output.outdir = args.outdir
    nsteps = args.steps if args.steps is not None else cfg.time.n_steps

    topo = build_topology(cfg)
    device = args.device
    if device == "auto":
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
    if topo.rank == 0:
        print("warpflow-les %s: case=%s grid=%dx%dx%d domains=%s device=%s"
              % ("0.1.0", cfg.name, cfg.grid.nx, cfg.grid.ny, cfg.grid.nz,
                 list(cfg.mpi.domains), device), flush=True)

    sol = FlowSolver(cfg, topo, device=device)
    comm = topo.comm
    t0 = time.time()

    for itime in range(1, nsteps + 1):
        info = sol.step()
        if itime % cfg.output.log_interval == 0 and topo.rank == 0:
            print("it %6d  ctime %.6f  dt %.3e  rmax %.3e  outer %2d  cfl %.2f"
                  % (itime, info["ctime"], info["dt"], info["rmax"],
                     info["iters"], info["cfl"]), flush=True)
        if itime % cfg.output.n_out == 0:
            write_snapshot(sol, itime)

    if nsteps % cfg.output.n_out != 0:
        write_snapshot(sol, nsteps)

    if topo.rank == 0:
        wall = time.time() - t0
        print("done: itime=%d ctime=%.6f wall=%.1fs"
              % (nsteps, sol.ctime, wall), flush=True)
    comm.Barrier()


if __name__ == "__main__":
    main()
