"""warpflow-les: GPU (NVIDIA Warp) + MPI incompressible LES solver.

A Python port of Hydro3D v7.0 numerics: staggered-grid fractional-step
incompressible LES with 4th-order CDS convection, explicit 4th-order
diffusion, WALE sub-grid model and a geometric multigrid Poisson solver.
MPI between nodes (mpi4py), NVIDIA Warp GPU kernels within a node, all
floating point in float64.
"""
from .config import load_case
from .topology import build_topology, NPL
from .solver import FlowSolver

__version__ = "0.1.0"
__all__ = ["load_case", "build_topology", "FlowSolver", "NPL", "__version__"]
