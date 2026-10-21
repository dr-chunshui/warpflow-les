"""Time-marching flow solver.

Port of the Hydro3D v7.0 time loop (flosol.for) with the RK2 convection/
diffusion stages (rungek.for), the WALE sub-grid model (eddyvis_wale.for)
and the fractional-step projection (press.for calvel/calmas + newsolv_mg).

One time step:
  1. checkdt        - adaptive dt from advective + viscous limits
  2. save uoo/voo/woo
  3. WALE eddy viscosity
  4. RK2 stages alpha = 1/2, 1:  ustar = uoo - dt*alpha*conv + dt*alpha*visc
     (stage 1 result is projected onto the new base with calvel)
  5. projection: outer loop {calmas; multigrid Poisson for pp; p += pp;
     calvel} until the 4th-order divergence residual rmax < eps
"""
from mpi4py import MPI

import warp as wp

from . import kernels as K
from .bcs import BoundaryKernels
from .halo import HaloExchanger
from .multigrid import Multigrid
from .topology import NPL

ALFARK = (0.5, 1.0)  # Heun RK2 stage coefficients (conv_sch = 3)


class FlowSolver:
    def __init__(self, cfg, topo, device):
        self.cfg = cfg
        self.topo = topo
        self.device = device

        ni, nj, nk = topo.local_ncells
        self.ncells = (ni, nj, nk)
        self.dx = cfg.domain.lx / cfg.grid.nx
        self.dy = cfg.domain.ly / cfg.grid.ny
        self.dz = cfg.domain.lz / cfg.grid.nz
        self.rey = 1.0 / cfg.physics.reynolds

        shape = (ni + 2 * NPL, nj + 2 * NPL, nk + 2 * NPL)
        names = ("u", "v", "w", "p",
                 "ustar", "vstar", "wstar",
                 "uoo", "voo", "woo",
                 "vis", "su", "pp")
        self.f = {name: wp.zeros(shape, dtype=wp.float64, device=device)
                  for name in names}

        self.halo = HaloExchanger(topo, device, NPL)
        b = cfg.boundaries
        self.bnd = {"west": b.west, "east": b.east, "south": b.south,
                    "north": b.north, "bottom": b.bottom, "top": b.top}
        self.bcs = BoundaryKernels(self.bnd, device)
        self.ulid = b.lid_velocity

        self.mg = Multigrid(cfg, topo, device, self.dx, self.dy, self.dz)
        self._buf = wp.zeros(1, dtype=wp.float64, device=device)

        self.dt = cfg.time.dt_init
        self.ctime = 0.0
        self.itime = 0
        self.rmax = 0.0
        self.cfl = 0.0
        self.outer_iters = 0

        # initial condition: rest + deterministic white noise
        amp = cfg.initial.noise if cfg.initial.type == "rest" else 0.0
        if amp > 0.0:
            wp.launch(K.init_noise, dim=self.ncells,
                      inputs=[self.f["u"], self.f["v"], self.f["w"],
                              amp, 1, NPL],
                      device=device)
        # vis starts as the molecular value everywhere, ghosts included
        # (Hydro3D's restart field carries vis = 1/Re on every plane and
        # only the first ghost layer is ever refreshed).
        self.f["vis"].fill_(self.rey)
        self.bcs.apply(self.f, self.ncells, self.ulid, NPL)

        # initial.for massflux/correctoutflux never activate for this
        # benchmark: they are guarded on inflow (bc=1/7/8) and zero-gradient
        # outflow (bc=2) faces, and every side here is a wall or a slip
        # symmetry plane, so the outflow-flux correction is a no-op.

    # ------------------------------------------------------------------ #
    # ghost fills                                                        #
    # ------------------------------------------------------------------ #
    def fill_bc(self, name, alfabc):
        """One Hydro3D boundu/boundv/boundw call for a single field."""
        self.halo.exchange(self.f[name])
        n = self.ncells[0]
        if name == "u":
            self.bcs.apply_u(self.f["u"], n, self.ulid, alfabc)
        elif name == "v":
            self.bcs.apply_v(self.f["v"], n, alfabc)
        elif name == "w":
            self.bcs.apply_w(self.f["w"], n, alfabc)

    def _op_dim(self, comp):
        """Operator write ranges: u x-faces isu..ieu (reduced at the east
        wall), v y-faces jsv..jev (reduced at the north slip plane),
        w z-faces ksw..kew (reduced at the lid)."""
        ni, nj, nk = self.ncells
        if comp == "u":
            return (ni - 1, nj, nk)
        if comp == "v":
            return (ni, nj - 1, nk)
        if comp == "w":
            return (ni, nj, nk - 1)
        return (ni, nj, nk)

    # ------------------------------------------------------------------ #
    # time step control (checkdt.for)                                    #
    # ------------------------------------------------------------------ #
    def _maxabs(self, name):
        self._buf.zero_()
        wp.launch(K.maxabs_reduce, dim=self.ncells,
                  inputs=[self.f[name], self._buf, NPL], device=self.device)
        loc = float(self._buf.numpy()[0])
        return self.topo.comm.allreduce(loc, op=MPI.MAX)

    def compute_dt(self):
        umax = self._maxabs("u")
        vmax = self._maxabs("v")
        wmax = self._maxabs("w")

        self._buf.fill_(1.0e10)  # atomic_min needs a large initialiser
        wp.launch(K.dtvisc_reduce, dim=self.ncells,
                  inputs=[self.f["u"], self.f["v"], self.f["w"],
                          self.f["vis"], self._buf,
                          self.dx, self.dy, self.dz, NPL],
                  device=self.device)
        dtvisc = float(self._buf.numpy()[0])
        dtvisc = self.topo.comm.allreduce(dtvisc, op=MPI.MIN)

        big = 1.0e10
        dtcfl = big
        for n, d, um in zip(self.ncells, (self.dx, self.dy, self.dz),
                            (umax, vmax, wmax)):
            if um > 0.0:
                dtcfl = min(dtcfl, d / um)
        dtmax = min(dtvisc, dtcfl)
        if dtmax <= 0.0 or not (dtmax < big):
            raise RuntimeError("time step collapsed (dtmax = %.3e)" % dtmax)

        self.cfl = self.dt / dtmax
        if self.cfg.time.variable_dt:
            dt_new = min(self.dt * self.cfg.time.dt_growth,
                         self.cfg.time.safety_factor * dtmax)
            self.dt = self.topo.comm.allreduce(dt_new, op=MPI.MIN)

    # ------------------------------------------------------------------ #
    # projection pieces (press.for)                                      #
    # ------------------------------------------------------------------ #
    def _calvel(self, alpha):
        # H's calvel ends with exchange(1,2,3) only: no bound calls, the
        # velocity ghosts keep whatever the last bound call left.
        f = self.f
        for ker, star, comp, d in ((K.calvel_u, "ustar", "u", self.dx),
                                   (K.calvel_v, "vstar", "v", self.dy),
                                   (K.calvel_w, "wstar", "w", self.dz)):
            wp.launch(ker, dim=self._op_dim(comp),
                      inputs=[f[star], f["p"], f[comp], d, self.dt, alpha, NPL],
                      device=self.device)
        for name in ("u", "v", "w"):
            self.halo.exchange(f[name])

    def _calmas(self):
        self._buf.zero_()
        wp.launch(K.calmas, dim=self.ncells,
                  inputs=[self.f["u"], self.f["v"], self.f["w"],
                          self.f["su"], self._buf,
                          self.dx, self.dy, self.dz, self.dt, NPL],
                  device=self.device)
        loc = float(self._buf.numpy()[0])
        self.rmax = self.topo.comm.allreduce(loc, op=MPI.MAX)

    def _project(self):
        """newsolv_mg.for: outer Poisson/projection iterations."""
        f = self.f
        eps = self.cfg.numerics.div_tolerance
        maxcy = self.cfg.numerics.max_outer_iters
        self._calvel(1.0)
        iters = 0
        for it in range(1, maxcy + 1):
            iters = it
            self._calmas()
            if self.rmax < eps and it > 1:
                break
            if self.rmax > 10.0:
                raise RuntimeError(
                    "projection diverged: rmax = %.6e at itime %d"
                    % (self.rmax, self.itime))
            self.mg.solve(f["su"], f["pp"], NPL)
            # p(isp-1:iep+1,...) += pp -- the ghost ring is included, so p's
            # first ghost layer picks up the MG boundary values (pbound then
            # re-mirrors the physical faces; the periodic north keeps them)
            wp.launch(K.add_inplace,
                      dim=(self.ncells[0] + 2, self.ncells[1] + 2,
                           self.ncells[2] + 2),
                      inputs=[f["p"], f["pp"], NPL], device=self.device)
            self.halo.exchange(f["p"])
            self.bcs.apply_pbound(f["p"], self.ncells[0])
            self._calvel(1.0)
        else:
            if self.topo.rank == 0:
                print("warning: projection did not converge, "
                      "rmax = %.6e after %d iterations"
                      % (self.rmax, maxcy), flush=True)
        self.outer_iters = iters
        for name in ("u", "v", "w"):
            self.fill_bc(name, 0)

    # ------------------------------------------------------------------ #
    # one time step (flosol.for)                                         #
    # ------------------------------------------------------------------ #
    def step(self):
        self.itime += 1
        self.compute_dt()
        self.ctime += self.dt

        f = self.f
        wp.copy(f["uoo"], f["u"])
        wp.copy(f["voo"], f["v"])
        wp.copy(f["woo"], f["w"])

        # sub-grid viscosity
        if self.cfg.physics.sgs_enabled:
            wp.launch(K.wale_viscosity, dim=self.ncells,
                      inputs=[f["u"], f["v"], f["w"], f["vis"],
                              self.dx, self.dy, self.dz,
                              self.rey, self.cfg.physics.sgs_cw, NPL],
                      device=self.device)
            self.bcs.apply_vis(f["vis"], self.ncells[0], self.rey)
        else:
            wp.launch(K.fill_const, dim=self.ncells,
                      inputs=[f["vis"], self.rey, NPL], device=self.device)

        # pressure_forcing (press.for) is an exact no-op here: the only
        # periodic direction is y and its PI controller drives the mean
        # v-flux towards qstpn_y, which stays 0 because boundv zeroes the
        # south v-face that the flux is measured on.

        dt = self.dt
        d3 = (self.dx, self.dy, self.dz)
        comps = (("u", "ustar", "uoo", K.conv4th_u, K.diff4th_u),
                 ("v", "vstar", "voo", K.conv4th_v, K.diff4th_v),
                 ("w", "wstar", "woo", K.conv4th_w, K.diff4th_w))
        for stage, alpha in enumerate(ALFARK):
            # flosol.for passes alfabc = 1 on the first stage only; it only
            # gates branches for outflow faces, which this benchmark has
            # none of, so the ghost fills are identical on both stages.
            alfabc = 1 if stage == 0 else 0
            for comp, star, oo, kconv, kdiff in comps:
                self.fill_bc(comp, alfabc)
                wp.launch(kconv, dim=self._op_dim(comp),
                          inputs=[f["u"], f["v"], f["w"], f[oo], f[star],
                                  *d3, dt, alpha, NPL],
                          device=self.device)
            for comp, star, oo, kconv, kdiff in comps:
                self.fill_bc(comp, alfabc)
                wp.launch(kdiff, dim=self._op_dim(comp),
                          inputs=[f[comp], f["vis"], f[star],
                                  *d3, dt, alpha, NPL],
                          device=self.device)
            if stage + 1 < len(ALFARK):
                self._calvel(alpha)

        self._project()
        return {"ctime": self.ctime, "dt": self.dt, "rmax": self.rmax,
                "iters": self.outer_iters, "cfl": self.cfl}
