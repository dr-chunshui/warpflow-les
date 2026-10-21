"""Case configuration for warpflow-les.

Loads a YAML case file into a typed configuration object. The layout mirrors
the Hydro3D `control.cin` semantics (numeric parameters, boundary conditions,
modelling options) so cases port over 1:1.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class DomainCfg:
    lx: float = 1.0
    ly: float = 1.0
    lz: float = 1.0


@dataclass
class GridCfg:
    nx: int = 80
    ny: int = 80
    nz: int = 80


@dataclass
class PhysicsCfg:
    reynolds: float = 400.0
    sgs_enabled: bool = True
    sgs_model: str = "wale"
    sgs_cw: float = 0.46


@dataclass
class InitialCfg:
    type: str = "rest"
    noise: float = 0.01


@dataclass
class BoundaryCfg:
    west: str = "wall"
    east: str = "wall"
    south: str = "slip"
    north: str = "slip"
    bottom: str = "wall"
    top: str = "moving_lid"
    lid_velocity: float = 1.0


@dataclass
class NumericsCfg:
    time_scheme: str = "rk2"
    convection: str = "cds4"
    diffusion: str = "explicit4"
    pressure_solver: str = "multigrid"
    mg_levels: int = 3
    max_outer_iters: int = 30
    div_tolerance: float = 1.0e-6
    ghost_layers: int = 3


@dataclass
class TimeCfg:
    n_steps: int = 1000
    variable_dt: bool = True
    dt_init: float = 3.0e-4
    safety_factor: float = 0.8
    dt_growth: float = 1.1


@dataclass
class OutputCfg:
    n_out: int = 1500
    log_interval: int = 20
    outdir: str = "output"


@dataclass
class MpiCfg:
    domains: tuple = (1, 1, 1)


@dataclass
class CaseConfig:
    name: str = "cavity"
    domain: DomainCfg = field(default_factory=DomainCfg)
    grid: GridCfg = field(default_factory=GridCfg)
    physics: PhysicsCfg = field(default_factory=PhysicsCfg)
    initial: InitialCfg = field(default_factory=InitialCfg)
    boundaries: BoundaryCfg = field(default_factory=BoundaryCfg)
    numerics: NumericsCfg = field(default_factory=NumericsCfg)
    time: TimeCfg = field(default_factory=TimeCfg)
    output: OutputCfg = field(default_factory=OutputCfg)
    mpi: MpiCfg = field(default_factory=MpiCfg)


def load_case(path: str | Path) -> CaseConfig:
    raw = yaml.safe_load(Path(path).read_text())
    cfg = CaseConfig(name=raw.get("case", "cavity"))
    cfg.domain = DomainCfg(**raw.get("domain", {}))
    cfg.grid = GridCfg(**raw.get("grid", {}))
    p = raw.get("physics", {})
    sgs = p.pop("sgs", {})
    cfg.physics = PhysicsCfg(
        reynolds=p.get("reynolds", 400.0),
        sgs_enabled=sgs.get("enabled", True),
        sgs_model=sgs.get("model", "wale"),
        sgs_cw=sgs.get("cw", 0.46),
    )
    cfg.initial = InitialCfg(**raw.get("initial", {}))
    cfg.boundaries = BoundaryCfg(**raw.get("boundaries", {}))
    cfg.numerics = NumericsCfg(**raw.get("numerics", {}))
    cfg.time = TimeCfg(**raw.get("time", {}))
    cfg.output = OutputCfg(**raw.get("output", {}))
    cfg.mpi = MpiCfg(**raw.get("mpi", {}))

    # sanity: global grid must divide evenly across sub-domains
    for axis, (n, m) in enumerate(zip(
            (cfg.grid.nx, cfg.grid.ny, cfg.grid.nz), cfg.mpi.domains)):
        if n % m != 0:
            raise ValueError(
                f"grid axis {axis} ({n} cells) not divisible by "
                f"mpi domains ({m})")
        if (n // m) % (2 ** (cfg.numerics.mg_levels - 1)) != 0:
            raise ValueError(
                f"sub-domain cells on axis {axis} ({n // m}) must be "
                f"divisible by 2**(mg_levels-1)={2 ** (cfg.numerics.mg_levels - 1)}")
    return cfg
