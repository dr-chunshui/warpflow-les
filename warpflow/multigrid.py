"""Geometric multigrid Poisson solver (port of Hydro3D mgsolver.for).

Solves   sum_nb a_nb (p_p - p_nb) = -rhs   with a 7-point stencil on a
uniform staggered grid, homogeneous Neumann conditions at physical
boundaries (implemented exactly like the Fortran: the boundary-face
coefficient is zeroed at boundary cells), and a nozzle of 1 ghost layer per
side at every level (Hydro3D mgc_* = n/2^(l-1) + 2).

Smoothing is red-black Gauss-Seidel (mg_itrsch=3); restriction is 8-point
full weighting (mgrestr); prolongation is trilinear with the (1,3,9,27)/64
stencil of mgcorr.  A plain V-cycle replaces Hydro3D's K-cycle; with two
pre/post smoothing sweeps and enough coarsest sweeps this converges at
least as fast.
"""
import warp as wp

from .bcs import _mk_neumann
from .halo import HaloExchanger

NPL_MG = 1  # ghost layers per level (Hydro3D uses a single layer inside MG)


@wp.kernel
def _mgcoeff(aw: wp.array3d(dtype=wp.float64), ae: wp.array3d(dtype=wp.float64),
             as_: wp.array3d(dtype=wp.float64), an: wp.array3d(dtype=wp.float64),
             ab: wp.array3d(dtype=wp.float64), at: wp.array3d(dtype=wp.float64),
             ap: wp.array3d(dtype=wp.float64),
             n: wp.int32,
             bcw: wp.int32, bce: wp.int32, bcs: wp.int32, bcn: wp.int32,
             bcb: wp.int32, bct: wp.int32,
             dxl: wp.float64, dyl: wp.float64, dzl: wp.float64):
    """Face coefficients on real cells; zeroed across physical boundaries."""
    i, j, k = wp.tid()
    awv = wp.float64(1.0) / (dxl * dxl)
    aev = awv
    asv = wp.float64(1.0) / (dyl * dyl)
    anv = asv
    abv = wp.float64(1.0) / (dzl * dzl)
    atv = abv
    if bcw == 1:
        if i == 0:
            awv = wp.float64(0.0)
    if bce == 1:
        if i == n - 1:
            aev = wp.float64(0.0)
    if bcs == 1:
        if j == 0:
            asv = wp.float64(0.0)
    if bcn == 1:
        if j == n - 1:
            anv = wp.float64(0.0)
    if bcb == 1:
        if k == 0:
            abv = wp.float64(0.0)
    if bct == 1:
        if k == n - 1:
            atv = wp.float64(0.0)
    ii = i + 1
    jj = j + 1
    kk = k + 1
    aw[ii, jj, kk] = awv
    ae[ii, jj, kk] = aev
    as_[ii, jj, kk] = asv
    an[ii, jj, kk] = anv
    ab[ii, jj, kk] = abv
    at[ii, jj, kk] = atv
    ap[ii, jj, kk] = -(awv + aev + asv + anv + abv + atv)


@wp.kernel
def _rbgs(phi: wp.array3d(dtype=wp.float64),
          aw: wp.array3d(dtype=wp.float64), ae: wp.array3d(dtype=wp.float64),
          as_: wp.array3d(dtype=wp.float64), an: wp.array3d(dtype=wp.float64),
          ab: wp.array3d(dtype=wp.float64), at: wp.array3d(dtype=wp.float64),
          ap: wp.array3d(dtype=wp.float64),
          rhs: wp.array3d(dtype=wp.float64),
          color: wp.int32):
    """One half-sweep of red-black Gauss-Seidel over real cells."""
    i, j, k = wp.tid()
    if (i + j + k + color) % 2 == 1:  # matches Fortran MOD(i+j+k,2)==0 red
        ii = i + 1
        jj = j + 1
        kk = k + 1
        phi[ii, jj, kk] = (rhs[ii, jj, kk]
                           - (aw[ii, jj, kk] * phi[ii - 1, jj, kk]
                              + ae[ii, jj, kk] * phi[ii + 1, jj, kk]
                              + as_[ii, jj, kk] * phi[ii, jj - 1, kk]
                              + an[ii, jj, kk] * phi[ii, jj + 1, kk]
                              + ab[ii, jj, kk] * phi[ii, jj, kk - 1]
                              + at[ii, jj, kk] * phi[ii, jj, kk + 1])) / ap[ii, jj, kk]


@wp.kernel
def _restrict(phi_f: wp.array3d(dtype=wp.float64),
              rhs_f: wp.array3d(dtype=wp.float64),
              aw_f: wp.array3d(dtype=wp.float64), ae_f: wp.array3d(dtype=wp.float64),
              as_f: wp.array3d(dtype=wp.float64), an_f: wp.array3d(dtype=wp.float64),
              ab_f: wp.array3d(dtype=wp.float64), at_f: wp.array3d(dtype=wp.float64),
              ap_f: wp.array3d(dtype=wp.float64),
              rhs_c: wp.array3d(dtype=wp.float64)):
    """Full-weighting restriction of the fine-grid residual (mgrestr)."""
    ci, cj, ck = wp.tid()
    acc = wp.float64(0.0)
    for di in range(2):
        for dj in range(2):
            for dk in range(2):
                i = 2 * ci + di + 1
                j = 2 * cj + dj + 1
                k = 2 * ck + dk + 1
                res = (rhs_f[i, j, k]
                       - (aw_f[i, j, k] * phi_f[i - 1, j, k]
                          + ae_f[i, j, k] * phi_f[i + 1, j, k]
                          + as_f[i, j, k] * phi_f[i, j - 1, k]
                          + an_f[i, j, k] * phi_f[i, j + 1, k]
                          + ab_f[i, j, k] * phi_f[i, j, k - 1]
                          + at_f[i, j, k] * phi_f[i, j, k + 1]
                          + ap_f[i, j, k] * phi_f[i, j, k]))
                acc = acc + res
    rhs_c[ci + 1, cj + 1, ck + 1] = acc * wp.float64(0.125)


@wp.kernel
def _prolong(phi_c: wp.array3d(dtype=wp.float64),
             phi_f: wp.array3d(dtype=wp.float64)):
    """Trilinear prolongation, phi_f += P phi_c (mgcorr): each fine cell
    accumulates the weighted sum over its 8 surrounding coarse cells
    (weights 1/4 and 3/4 per axis, Hydro3D stencil (1,3,9,27)/64)."""
    i, j, k = wp.tid()  # fine real cells
    # per-axis: nearest coarse cell and its weight; the partner cell
    # (c0+1) takes the complementary weight
    c0x = wp.where(i % 2 == 0, i // 2 - 1, i // 2)
    wx0 = wp.where(i % 2 == 0, wp.float64(0.25), wp.float64(0.75))
    c0y = wp.where(j % 2 == 0, j // 2 - 1, j // 2)
    wy0 = wp.where(j % 2 == 0, wp.float64(0.25), wp.float64(0.75))
    c0z = wp.where(k % 2 == 0, k // 2 - 1, k // 2)
    wz0 = wp.where(k % 2 == 0, wp.float64(0.25), wp.float64(0.75))
    acc = wp.float64(0.0)
    for di in range(2):
        wxi = wp.where(di == 0, wx0, wp.float64(1.0) - wx0)
        for dj in range(2):
            wyj = wp.where(dj == 0, wy0, wp.float64(1.0) - wy0)
            for dk in range(2):
                wzk = wp.where(dk == 0, wz0, wp.float64(1.0) - wz0)
                acc = acc + wxi * wyj * wzk \
                    * phi_c[c0x + di + 1, c0y + dj + 1, c0z + dk + 1]
    phi_f[i + 1, j + 1, k + 1] = phi_f[i + 1, j + 1, k + 1] + acc


@wp.kernel
def _copy_in(src: wp.array3d(dtype=wp.float64), dst: wp.array3d(dtype=wp.float64),
             npl: wp.int32):
    """(npl-ghost) field -> (1-ghost) level-0 field, real cells only."""
    i, j, k = wp.tid()
    dst[i + 1, j + 1, k + 1] = src[i + npl, j + npl, k + npl]


@wp.kernel
def _copy_out_ring(src: wp.array3d(dtype=wp.float64),
                   dst: wp.array3d(dtype=wp.float64), npl: wp.int32):
    """(1-ghost) level-0 field -> (npl-ghost) field including the ring
    (mgkcyc's final cof -> pp copy runs over i = 1..mgc_i)."""
    i, j, k = wp.tid()
    dst[i + npl - 1, j + npl - 1, k + npl - 1] = src[i, j, k]


# ---------------------------------------------------------------------------
# exchangepp.for: after every relaxation sweep the ghost ring on a periodic
# face copies the OPPOSITE real edge (single-domain branch, jprev < 0 and
# jnext < 0): south ring <- last real row, north ring <- first real row.
# ---------------------------------------------------------------------------
@wp.kernel
def _wrap_x_low(phi: wp.array3d(dtype=wp.float64), n: wp.int32):
    j, k = wp.tid()
    phi[0, 1 + j, 1 + k] = phi[n, 1 + j, 1 + k]


@wp.kernel
def _wrap_x_high(phi: wp.array3d(dtype=wp.float64), n: wp.int32):
    j, k = wp.tid()
    phi[n + 1, 1 + j, 1 + k] = phi[1, 1 + j, 1 + k]


@wp.kernel
def _wrap_y_low(phi: wp.array3d(dtype=wp.float64), n: wp.int32):
    i, k = wp.tid()
    phi[1 + i, 0, 1 + k] = phi[1 + i, n, 1 + k]


@wp.kernel
def _wrap_y_high(phi: wp.array3d(dtype=wp.float64), n: wp.int32):
    i, k = wp.tid()
    phi[1 + i, n + 1, 1 + k] = phi[1 + i, 1, 1 + k]


@wp.kernel
def _wrap_z_low(phi: wp.array3d(dtype=wp.float64), n: wp.int32):
    i, j = wp.tid()
    phi[1 + i, 1 + j, 0] = phi[1 + i, 1 + j, n]


@wp.kernel
def _wrap_z_high(phi: wp.array3d(dtype=wp.float64), n: wp.int32):
    i, j = wp.tid()
    phi[1 + i, 1 + j, n + 1] = phi[1 + i, 1 + j, 1]


class _Level:
    def __init__(self, n3, device):
        shape = tuple(x + 2 for x in n3)
        self.n3 = n3
        self.shape = shape
        self.phi = wp.zeros(shape, dtype=wp.float64, device=device)
        self.rhs = wp.zeros(shape, dtype=wp.float64, device=device)
        self.aw = wp.zeros(shape, dtype=wp.float64, device=device)
        self.ae = wp.zeros(shape, dtype=wp.float64, device=device)
        self.as_ = wp.zeros(shape, dtype=wp.float64, device=device)
        self.an = wp.zeros(shape, dtype=wp.float64, device=device)
        self.ab = wp.zeros(shape, dtype=wp.float64, device=device)
        self.at = wp.zeros(shape, dtype=wp.float64, device=device)
        self.ap = wp.zeros(shape, dtype=wp.float64, device=device)


class Multigrid:
    def __init__(self, cfg, topo, device, dx, dy, dz):
        self.cfg = cfg
        self.topo = topo
        self.device = device
        self.halo = HaloExchanger(topo, device, NPL_MG)
        sides = ("west", "east", "south", "north", "bottom", "top")
        self.phys = {s: (topo.neighbors["xyz"[i // 2]][i % 2] < 0
                         and getattr(cfg.boundaries, s) != "periodic")
                     for i, s in enumerate(sides)}
        self.mirror = {s: _mk_neumann(s) for s in sides}
        wrapk = {"west": _wrap_x_low, "east": _wrap_x_high,
                 "south": _wrap_y_low, "north": _wrap_y_high,
                 "bottom": _wrap_z_low, "top": _wrap_z_high}
        self.wrap = {s: wrapk[s]
                     for i, s in enumerate(sides)
                     if topo.neighbors["xyz"[i // 2]][i % 2] < 0
                     and getattr(cfg.boundaries, s) == "periodic"}

        nb = topo.local_ncells
        self.levels = []
        for l in range(cfg.numerics.mg_levels):
            n3 = tuple(x >> l for x in nb)
            lev = _Level(n3, device)
            bflag = tuple(1 if self.phys[s] else 0
                          for s in ("west", "east", "south", "north",
                                    "bottom", "top"))
            wp.launch(_mgcoeff, dim=lev.n3,
                      inputs=[lev.aw, lev.ae, lev.as_, lev.an, lev.ab, lev.at,
                              lev.ap, lev.n3[0], *bflag,
                              dx * 2.0 ** l, dy * 2.0 ** l, dz * 2.0 ** l],
                      device=device)
            self.levels.append(lev)

        self.n_pre = 6       # smoothing sweeps on the way down
        self.n_post = 6      # smoothing sweeps on the way up
        self.n_coarse = 300  # sweeps at the coarsest level
        self.n_vcycles = 4   # V-cycles per solve (stands in for the K-cycle)

    # -- ghost refill: mirror at physical sides, wrap at periodic sides
    #    (mgbound then exchangepp, run after every full relaxation sweep) --
    def _refill(self, lev):
        for side in ("west", "east", "south", "north", "bottom", "top"):
            if self.phys[side]:
                n_of = {"west": lev.n3[0], "east": lev.n3[0],
                        "south": lev.n3[1], "north": lev.n3[1],
                        "bottom": lev.n3[2], "top": lev.n3[2]}[side]
                dim = {( "west", "east"): (lev.n3[1], lev.n3[2]),
                       ("south", "north"): (lev.n3[0], lev.n3[2]),
                       ("bottom", "top"): (lev.n3[0], lev.n3[1])}
                key = next(k for k in dim if side in k)
                wp.launch(self.mirror[side], dim=dim[key],
                          inputs=[lev.phi, n_of, NPL_MG], device=self.device)
        for side, ker in self.wrap.items():
            n_of = {"west": lev.n3[0], "east": lev.n3[0],
                    "south": lev.n3[1], "north": lev.n3[1],
                    "bottom": lev.n3[2], "top": lev.n3[2]}[side]
            dim = {"west": (lev.n3[1], lev.n3[2]),
                   "east": (lev.n3[1], lev.n3[2]),
                   "south": (lev.n3[0], lev.n3[2]),
                   "north": (lev.n3[0], lev.n3[2]),
                   "bottom": (lev.n3[0], lev.n3[1]),
                   "top": (lev.n3[0], lev.n3[1])}[side]
            wp.launch(ker, dim=dim, inputs=[lev.phi, n_of], device=self.device)
        self.halo.exchange(lev.phi)

    def _relax(self, l, nsweeps):
        lev = self.levels[l]
        for _ in range(nsweeps):
            # mgrelax itrsch=3: one red + one black half-sweep, then the
            # ring refresh happens once per sweep (mgbound + exchangepp)
            for color in (0, 1):
                wp.launch(_rbgs, dim=lev.n3,
                          inputs=[lev.phi, lev.aw, lev.ae, lev.as_, lev.an,
                                  lev.ab, lev.at, lev.ap, lev.rhs, color],
                          device=self.device)
            self._refill(lev)

    def _vcycle(self):
        nlev = len(self.levels)
        for l in range(nlev):
            if l > 0:
                f, c = self.levels[l - 1], self.levels[l]
                wp.launch(_restrict, dim=c.n3,
                          inputs=[f.phi, f.rhs, f.aw, f.ae, f.as_, f.an,
                                  f.ab, f.at, f.ap, c.rhs],
                          device=self.device)
                # mgrestr zeroes the whole coarse phi block, rings included;
                # the first coarse sweep reads those zero rings
                c.phi.zero_()
            if l < nlev - 1:
                self._relax(l, self.n_pre)
            else:
                self._relax(l, self.n_coarse)
        for l in range(nlev - 2, -1, -1):
            f, c = self.levels[l], self.levels[l + 1]
            wp.launch(_prolong, dim=f.n3,
                      inputs=[c.phi, f.phi], device=self.device)
            self._refill(f)
            self._relax(l, self.n_post)

    def solve(self, su3, pp3, npl):
        """Solve A pp = su3 (npl-ghost arrays).  newsolv_mg zeroes pp before
        every outer iteration, so the cycle always starts from phi = 0 with
        zero rings (mgbound/exchangepp only run after the first sweep); the
        result is copied back including the one-layer ring."""
        lev0 = self.levels[0]
        wp.launch(_copy_in, dim=lev0.n3, inputs=[su3, lev0.rhs, npl],
                  device=self.device)
        pp3.zero_()
        lev0.phi.zero_()
        for _ in range(self.n_vcycles):
            self._vcycle()
        wp.launch(_copy_out_ring, dim=lev0.shape,
                  inputs=[lev0.phi, pp3, npl], device=self.device)
