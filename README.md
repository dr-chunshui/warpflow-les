# warpflow-les

A Python port of [Hydro3D](https://github.com/hydro3d) v7.0 (large-eddy
simulation of incompressible flow on staggered Cartesian grids) for
NVIDIA GPUs, built on the [Warp](https://github.com/NVIDIA/warp) kernel
framework with MPI for multi-node decomposition.

Everything computes in **float64**.  Within a node the fields live in
device memory and every kernel is a Warp kernel launched on the GPU;
between nodes the classic MPI halo exchange (mpi4py) couples the
sub-domains, mirroring Hydro3D's domain-decomposition layout.

## Numerics (mirroring Hydro3D v7.0)

| Piece | Method | Hydro3D source |
|---|---|---|
| Time integration | 2-stage Heun RK2 (`conv_sch = 3`) | `rungek.for` |
| Convection | 4th-order central differences | `conv4th` in `rungek.for` |
| Diffusion | explicit 4th-order, variable viscosity | `diff4th` in `rungek.for` |
| Sub-grid model | WALE (`cw = 0.46`) | `eddyvis_wale.for` |
| Projection | fractional step, outer loop on the 4th-order divergence residual | `press.for`, `newsolv_mg.for` |
| Poisson solver | geometric multigrid, 3 levels | `mgsolver.for` |
| Time step | advective + viscous limits (`checkdt.for`) | `checkdt.for` |
| Boundaries | wall / slip / moving lid / periodic, 3 ghost layers | `bounds.for` |

## Layout

```
warpflow/
  config.py     YAML case loader (dataclass tree)
  topology.py   MPI domain decomposition, ghost-layer bookkeeping (NPL = 3)
  halo.py       MPI halo exchange (GPU-pinned staging buffers)
  kernels.py    WARP kernels: convection, diffusion, WALE, dt control,
                projection pieces
  bcs.py        boundary fills (bounds.for): wall / slip / moving lid /
                Neumann pressure ghosts
  multigrid.py  geometric multigrid Poisson solver (mgsolver.for)
  solver.py     FlowSolver: the flosol.for time loop
  io.py         npz snapshots, log lines
cases/          YAML case files (cavity benchmark variants)
tests/          multigrid unit test, Hydro3D bridge (restart export +
                step-by-step field comparison)
scripts/        post-processing (Ghia centreline comparison)
run.py          CLI entry point
```

## Usage

```bash
pip install warp mpi4py pyyaml numpy

# single GPU
python run.py --case cases/cavity_serial.yaml --device cuda:0

# multi-node / multi-GPU (one rank per GPU)
mpirun -np 8 python run.py --case cases/cavity_mpi.yaml
```

Case files are YAML (see `cases/cavity_serial.yaml`):

```yaml
case: cavity_serial
domain:  {lx: 1.0, ly: 1.0, lz: 1.0}
grid:    {nx: 80, ny: 80, nz: 80}
physics:
  reynolds: 400
  sgs: {enabled: true, model: wale, cw: 0.46}
initial: {type: rest, noise: 0.01}
boundaries:
  west: wall       # wall | slip | periodic
  east: wall
  south: slip
  north: slip
  bottom: wall
  top: moving_lid  # + lid_velocity: 1.0
numerics:
  time_scheme: rk2
  convection: cds4
  diffusion: explicit4
  pressure_solver: multigrid
  mg_levels: 3
  max_outer_iters: 30
  div_tolerance: 1.0e-6
  ghost_layers: 3
time:    {n_steps: 50000, variable_dt: false, dt_init: 3.0e-4}
output:  {n_out: 1000, log_interval: 20, outdir: output_cavity}
mpi:     {domains: [1, 1, 1]}   # sub-domain count per axis
```

Snapshots land in `outdir` as `snap_%06d_r%03d.npz` (full 3-D arrays with
3 ghost layers, float64).

## Verification: lid-driven cavity vs Ghia et al.

`cases/cavity_serial.yaml` reproduces Hydro3D's `Benchmark_CavityFlow`
exactly: 80^3 grid, Re = 400, fixed dt = 3e-4, 50 000 steps (t = 15),
rest + 1 % noise initial condition, WALE SGS on.

Against the native Fortran code, driven from an identical restart field,
per-field differences stay at ~1e-4 rms (velocity scale 1) over the
first 100 steps, with matching divergence-residual trajectories; the
residual difference is traceable to the documented multigrid deviation
below.  The converged flow reproduces the Ghia, Tuann & Olson centreline
profiles (Re = 400):

```bash
python scripts/plot_ghia.py --snapshot output_cavity/snap_0050000_r000.npz
# -> output_cavity/ghia_comparison.png, centreline.csv
```

## Documented deviations from Hydro3D

1. **Multigrid smoother / cycle.**  Hydro3D uses TDMA line smoothers
   (`mg_itrsch = 2`) inside a K-cycle.  warpflow uses red-black
   Gauss-Seidel (6 pre + 6 post sweeps, 300 coarse sweeps) inside a
   plain V-cycle, iterated until the same outer convergence criterion.
   The Poisson problems are solved to the same tolerance, but the
   iteration path (and therefore the additive pressure constant and
   tiny per-point pressure differences) is not bit-identical; velocity
   fields agree to ~1e-4 rms over 100 steps.
2. **`correctoutflux` / `pressure_forcing`.**  Both are structurally
   inert for the cavity benchmark (they act only on inflow/outflow or
   fully periodic faces) and are therefore omitted; the guarded code
   paths exist only in Hydro3D configurations with through-flow.
3. **Restart files.**  Hydro3D's Fortran unformatted `tecbin` restarts
   are not read natively; `tests/h3d_bridge.py` can export an
   equivalent restart for cross-validation purposes.
