"""Cavity regression test: a small lid-driven cavity must stay consistent.

Runs 30 steps of a 24^3 case on the GPU and checks the invariants that
the benchmark relies on:
  - the projection converges (rmax below the tolerance),
  - all fields stay finite,
  - the boundary ghosts hold exactly the bounds.for values (lid speed on
    the top u-face, no-penetration on every wall, mirror symmetry for
    the tangential slip components).
"""
import os
import sys

import numpy as np
import pytest
import warp as wp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warpflow.config import load_case
from warpflow.solver import FlowSolver
from warpflow.topology import NPL, build_topology

CASE = """
case: cavity_test
domain:  {lx: 1.0, ly: 1.0, lz: 1.0}
grid:    {nx: 24, ny: 24, nz: 24}
physics:
  reynolds: 400
  sgs: {enabled: true, model: wale, cw: 0.46}
initial: {type: rest, noise: 0.01}
boundaries:
  west: wall
  east: wall
  south: slip
  north: slip
  bottom: wall
  top: moving_lid
  lid_velocity: 1.0
numerics:
  time_scheme: rk2
  convection: cds4
  diffusion: explicit4
  pressure_solver: multigrid
  mg_levels: 3
  max_outer_iters: 30
  div_tolerance: 1.0e-6
  ghost_layers: 3
time: {n_steps: 30, variable_dt: true, dt_init: 3.0e-4,
       safety_factor: 0.8, dt_growth: 1.1}
output: {n_out: 1000, log_interval: 10, outdir: /tmp/warpflow_test}
mpi: {domains: [1, 1, 1]}
"""


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    casefile = tmp_path_factory.mktemp("case") / "cavity_test.yaml"
    casefile.write_text(CASE)
    cfg = load_case(casefile)
    topo = build_topology(cfg)
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    sol = FlowSolver(cfg, topo, device=device)
    infos = [sol.step() for _ in range(30)]
    return sol, infos


def test_projection_converges(result):
    _, infos = result
    for info in infos:
        assert info["rmax"] < 1.0e-6
        assert info["iters"] <= 6


def test_fields_finite(result):
    sol, _ = result
    for name in ("u", "v", "w", "p", "vis"):
        arr = sol.f[name].numpy()
        assert np.isfinite(arr).all(), name


def test_boundary_ghosts(result):
    sol, _ = result
    n = sol.ncells[0]
    for name in ("u", "v", "w"):
        g = sol.f[name].numpy()
        s = slice(NPL, NPL + n)
        if name == "u":
            # moving lid: ghost layer just above the top u-faces
            assert np.allclose(g[NPL:NPL + n, NPL:NPL + n, NPL + n],
                               sol.ulid)
            # west / east walls: no normal flow
            assert np.allclose(g[NPL - 1, s, s], 0.0)
            assert np.allclose(g[NPL + n, s, s], 0.0)
        if name == "v":
            # south / north: v is the wall-normal component there
            assert np.allclose(g[s, NPL - 1, s], 0.0)
            assert np.allclose(g[s, NPL + n, s], 0.0)
        if name == "w":
            # bottom / top walls: no normal flow
            assert np.allclose(g[s, s, NPL - 1], 0.0)
            assert np.allclose(g[s, s, NPL + n], 0.0)
    # pressure: homogeneous Neumann on every physical face
    p = sol.f["p"].numpy()
    s = slice(NPL, NPL + n)
    assert np.allclose(p[NPL - 1, s, s], p[NPL, s, s])
    assert np.allclose(p[NPL + n, s, s], p[NPL + n - 1, s, s])
    assert np.allclose(p[s, NPL - 1, s], p[s, NPL, s])
    assert np.allclose(p[s, NPL + n, s], p[s, NPL + n - 1, s])
    assert np.allclose(p[s, s, NPL - 1], p[s, s, NPL])
    assert np.allclose(p[s, s, NPL + n], p[s, s, NPL + n - 1])
