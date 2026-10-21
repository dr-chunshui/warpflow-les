"""MPI domain decomposition (port of Hydro3D infodom/mdmap logic).

Splits the global grid into `sd_x x sd_y x sd_z` sub-domains; each MPI rank
owns exactly one sub-domain. Rank 0 of a cartesian layout maps to
sub-domain (0,0,0), matching Hydro3D's dom ordering.
"""
from __future__ import annotations

from dataclasses import dataclass

from mpi4py import MPI

from .config import CaseConfig

NPL = 3  # ghost layers on each side (Hydro3D pl = pl_ex + 1 with pl_ex = 2)


@dataclass
class Topology:
    rank: int
    nranks: int
    comm: MPI.Comm
    sd: tuple            # (sd_x, sd_y, sd_z) decomposition
    coords: tuple        # cartesian coordinates of this rank's sub-domain
    neighbors: dict      # axis -> (rank_low, rank_high); -1 means physical boundary
    local_ncells: tuple  # (ni, nj, nk) cells owned by this rank
    global_offsets: tuple


def build_topology(cfg: CaseConfig) -> Topology:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nranks = comm.Get_size()
    sd = tuple(int(m) for m in cfg.mpi.domains)
    if nranks != sd[0] * sd[1] * sd[2]:
        raise ValueError(
            f"mpi.domains {sd} implies {sd[0]*sd[1]*sd[2]} ranks, "
            f"but mpirun launched {nranks}")

    coords = _rank_to_coords(rank, sd)

    cart = comm.Create_cart(sd, periods=(False, False, False), reorder=False)
    neighbors = {}
    for axis, name in enumerate(("x", "y", "z")):
        lo, hi = cart.Shift(axis, 1)
        neighbors[name] = (lo, hi)

    gn = (cfg.grid.nx, cfg.grid.ny, cfg.grid.nz)
    local = tuple(g // s for g, s in zip(gn, sd))
    offsets = tuple(c * n for c, n in zip(coords, local))
    return Topology(rank=rank, nranks=nranks, comm=comm, sd=sd,
                    coords=coords, neighbors=neighbors,
                    local_ncells=local, global_offsets=offsets)


def _rank_to_coords(rank: int, sd: tuple) -> tuple:
    """Hydro3D orders sub-domains x-fastest (i fastest, then j, then k)."""
    i = rank % sd[0]
    j = (rank // sd[0]) % sd[1]
    k = rank // (sd[0] * sd[1])
    return (i, j, k)
