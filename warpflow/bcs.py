"""Boundary-condition ghost fills: exact port of Hydro3D's bounds.for,
pbound and the eddyvis_wale ghost section for the lid-driven cavity
benchmark configuration (Benchmark_CavityFlow/control.cin).  The values
that read_control consumes are the plain numbers at the start of each BC
line (the trailing text only documents the available options):

    bc_west=4 (wall), bc_east=4 (wall), bc_south=3 (slip),
    bc_north=3 (slip), bc_bottom=4 (wall),
    bc_top=1 + keyword='cavity' (moving lid).

PERIODIC stays false, so exchange_bc, pressure_forcing and the
correctoutflux scaling never activate (initial.for guards them on
bc=5 / bc=2 faces).

Fortran -> full-array index map (NPL = 3, n = 80 real cells, single rank;
Fortran index f corresponds to full index f-2):

    u: x faces isu..ieu = 5..83  -> 3..81, y/z centers -> 3..82
    v: y faces jsv..jev = 5..83  -> 3..81 (jev reduced: north is not
       periodic), x/z centers -> 3..82
    w: z faces ksw..kew = 5..83  -> 3..81, x/y centers -> 3..82
    p: cells 5..84               -> 3..82

Every boundu/boundv/boundw call performs, in this order (ly = 0..2):

  1. west side    (bc=4:  u face zero / v, w odd mirror)
  2. east side    (bc=4:  u face zero / v, w odd mirror)
  3. south side   (bc=3:  u, w even mirror / v zero)
  4. north side   (bc=3:  u, w even mirror / v zero)
  5. bottom side  (bc=4:  u, v odd mirror / w zero)
  6. top side     (bc_t=1 has no branch; the 'cavity' special cases set
                   u = ubulk, v even mirror (ke+1+ly), w = 0 (kew+1+ly))

pbound fills ONE Neumann ghost layer on every side (no periodic face).
The eddy-viscosity ghosts are handled by apply_vis.
"""
import warp as wp

from .topology import NPL


def _mk_lin(axis, high):
    """fld[ghost(m)] = sign * fld[src(m)] for m = 0..nlay-1.

    Low side:  ghost = g0-m, src = s0+m.  High side: ghost = g0+m, src = s0-m.
    The two perpendicular axes run over [a0, a0+na) x [b0, b0+nb).
    """

    if axis == 0:
        if high:
            @wp.kernel
            def f(fld: wp.array3d(dtype=wp.float64), g0: wp.int32,
                  s0: wp.int32, sign: wp.float64, a0: wp.int32,
                  na: wp.int32, b0: wp.int32, nb: wp.int32, nlay: wp.int32):
                aa, bb = wp.tid()
                for m in range(nlay):
                    fld[g0 + m, a0 + aa, b0 + bb] = \
                        sign * fld[s0 - m, a0 + aa, b0 + bb]
        else:
            @wp.kernel
            def f(fld: wp.array3d(dtype=wp.float64), g0: wp.int32,
                  s0: wp.int32, sign: wp.float64, a0: wp.int32,
                  na: wp.int32, b0: wp.int32, nb: wp.int32, nlay: wp.int32):
                aa, bb = wp.tid()
                for m in range(nlay):
                    fld[g0 - m, a0 + aa, b0 + bb] = \
                        sign * fld[s0 + m, a0 + aa, b0 + bb]
    elif axis == 1:
        if high:
            @wp.kernel
            def f(fld: wp.array3d(dtype=wp.float64), g0: wp.int32,
                  s0: wp.int32, sign: wp.float64, a0: wp.int32,
                  na: wp.int32, b0: wp.int32, nb: wp.int32, nlay: wp.int32):
                aa, bb = wp.tid()
                for m in range(nlay):
                    fld[a0 + aa, g0 + m, b0 + bb] = \
                        sign * fld[a0 + aa, s0 - m, b0 + bb]
        else:
            @wp.kernel
            def f(fld: wp.array3d(dtype=wp.float64), g0: wp.int32,
                  s0: wp.int32, sign: wp.float64, a0: wp.int32,
                  na: wp.int32, b0: wp.int32, nb: wp.int32, nlay: wp.int32):
                aa, bb = wp.tid()
                for m in range(nlay):
                    fld[a0 + aa, g0 - m, b0 + bb] = \
                        sign * fld[a0 + aa, s0 + m, b0 + bb]
    else:
        if high:
            @wp.kernel
            def f(fld: wp.array3d(dtype=wp.float64), g0: wp.int32,
                  s0: wp.int32, sign: wp.float64, a0: wp.int32,
                  na: wp.int32, b0: wp.int32, nb: wp.int32, nlay: wp.int32):
                aa, bb = wp.tid()
                for m in range(nlay):
                    fld[a0 + aa, b0 + bb, g0 + m] = \
                        sign * fld[a0 + aa, b0 + bb, s0 - m]
        else:
            @wp.kernel
            def f(fld: wp.array3d(dtype=wp.float64), g0: wp.int32,
                  s0: wp.int32, sign: wp.float64, a0: wp.int32,
                  na: wp.int32, b0: wp.int32, nb: wp.int32, nlay: wp.int32):
                aa, bb = wp.tid()
                for m in range(nlay):
                    fld[a0 + aa, b0 + bb, g0 - m] = \
                        sign * fld[a0 + aa, b0 + bb, s0 + m]
    return f


def _mk_const(axis, high):
    """fld[ghost(m)] = val for m = 0..nlay-1 (zero / lid velocity)."""

    if axis == 0:
        if high:
            @wp.kernel
            def f(fld: wp.array3d(dtype=wp.float64), g0: wp.int32,
                  val: wp.float64, a0: wp.int32, na: wp.int32,
                  b0: wp.int32, nb: wp.int32, nlay: wp.int32):
                aa, bb = wp.tid()
                for m in range(nlay):
                    fld[g0 + m, a0 + aa, b0 + bb] = val
        else:
            @wp.kernel
            def f(fld: wp.array3d(dtype=wp.float64), g0: wp.int32,
                  val: wp.float64, a0: wp.int32, na: wp.int32,
                  b0: wp.int32, nb: wp.int32, nlay: wp.int32):
                aa, bb = wp.tid()
                for m in range(nlay):
                    fld[g0 - m, a0 + aa, b0 + bb] = val
    elif axis == 1:
        if high:
            @wp.kernel
            def f(fld: wp.array3d(dtype=wp.float64), g0: wp.int32,
                  val: wp.float64, a0: wp.int32, na: wp.int32,
                  b0: wp.int32, nb: wp.int32, nlay: wp.int32):
                aa, bb = wp.tid()
                for m in range(nlay):
                    fld[a0 + aa, g0 + m, b0 + bb] = val
        else:
            @wp.kernel
            def f(fld: wp.array3d(dtype=wp.float64), g0: wp.int32,
                  val: wp.float64, a0: wp.int32, na: wp.int32,
                  b0: wp.int32, nb: wp.int32, nlay: wp.int32):
                aa, bb = wp.tid()
                for m in range(nlay):
                    fld[a0 + aa, g0 - m, b0 + bb] = val
    else:
        if high:
            @wp.kernel
            def f(fld: wp.array3d(dtype=wp.float64), g0: wp.int32,
                  val: wp.float64, a0: wp.int32, na: wp.int32,
                  b0: wp.int32, nb: wp.int32, nlay: wp.int32):
                aa, bb = wp.tid()
                for m in range(nlay):
                    fld[a0 + aa, b0 + bb, g0 + m] = val
        else:
            @wp.kernel
            def f(fld: wp.array3d(dtype=wp.float64), g0: wp.int32,
                  val: wp.float64, a0: wp.int32, na: wp.int32,
                  b0: wp.int32, nb: wp.int32, nlay: wp.int32):
                aa, bb = wp.tid()
                for m in range(nlay):
                    fld[a0 + aa, b0 + bb, g0 - m] = val
    return f


class BoundaryKernels:
    """Exact cavity-benchmark ghost fills for u, v, w, p and vis.

    Expected configuration: west/east/bottom = wall, south/north = slip,
    top = moving_lid; deviations raise.
    """

    def __init__(self, bnd, device):
        self.device = device
        self.bnd = dict(bnd)
        ok = (self.bnd["south"] == "slip" and self.bnd["north"] == "slip"
              and self.bnd["top"] == "moving_lid"
              and self.bnd["west"] == "wall"
              and self.bnd["east"] == "wall"
              and self.bnd["bottom"] == "wall")
        if not ok:
            raise ValueError(
                "unsupported boundary combination for the Hydro3D benchmark "
                "port: %s" % self.bnd)

    # -- helpers ------------------------------------------------------- #
    def _launch_lin(self, fld, axis, high, g0, s0, sign, a0, na, b0, nb,
                    nlay):
        wp.launch(_mk_lin(axis, high), dim=(na, nb),
                  inputs=[fld, g0, s0, wp.float64(sign), a0, na, b0, nb,
                          nlay], device=self.device)

    def _launch_const(self, fld, axis, high, g0, val, a0, na, b0, nb, nlay):
        wp.launch(_mk_const(axis, high), dim=(na, nb),
                  inputs=[fld, g0, wp.float64(val), a0, na, b0, nb, nlay],
                  device=self.device)

    # -- per-field bound calls ------------------------------------------ #
    # Ranges scale with the local cell count n: the first ghost plane is
    # NPL-1, the last real one NPL+n-1, the first high-side ghost NPL+n.
    def apply_u(self, fld, n, ulid, alfabc):
        # west (bc_west=4): u(is-1-ly) = 0, j,k in [js-1, je+1]x[ks-1, ke+1]
        self._launch_const(fld, 0, False, NPL - 1, 0.0,
                           NPL - 1, n + 2, NPL - 1, n + 2, 3)
        # east (bc_east=4): u(ie+1+ly) = 0, same perpendicular ranges
        self._launch_const(fld, 0, True, NPL + n, 0.0,
                           NPL - 1, n + 2, NPL - 1, n + 2, 3)
        # south (bc_south=3): u(i, js-1-ly) = +u(i, js+ly),
        # i in [is-1, ie], k in [ks-1, ke+1]
        self._launch_lin(fld, 1, False, NPL - 1, NPL, 1.0,
                         NPL - 1, n + 1, NPL - 1, n + 2, 3)
        # north (bc_north=3): u(i, je+1+ly) = +u(i, je-ly)
        self._launch_lin(fld, 1, True, NPL + n, NPL + n - 1, 1.0,
                         NPL - 1, n + 1, NPL - 1, n + 2, 3)
        # bottom (bc_bottom=4): u(i, j, ks-1-ly) = -u(i, j, ks+ly)
        self._launch_lin(fld, 2, False, NPL - 1, NPL, -1.0,
                         NPL - 1, n + 1, NPL - 1, n + 2, 3)
        # top 'cavity': u(i, j, ke+1+ly) = ubulk,
        # i in [is-1, ie], j in [js-1, je]
        self._launch_const(fld, 2, True, NPL + n, ulid,
                           NPL - 1, n + 1, NPL - 1, n + 2, 3)

    def apply_v(self, fld, n, alfabc):
        # west (bc_west=4): v(is-1-ly) = -v(is+ly),
        # j in [js-1, je], k in [ks-1, ke+1] (jev is reduced)
        self._launch_lin(fld, 0, False, NPL - 1, NPL, -1.0,
                         NPL - 1, n + 1, NPL - 1, n + 2, 3)
        # east (bc_east=4): v(ie+1+ly) = -v(ie-ly)
        self._launch_lin(fld, 0, True, NPL + n, NPL + n - 1, -1.0,
                         NPL - 1, n + 1, NPL - 1, n + 2, 3)
        # south (bc_south=3): v(i, js-1-ly) = 0
        self._launch_const(fld, 1, False, NPL - 1, 0.0,
                           NPL - 1, n + 2, NPL - 1, n + 2, 3)
        # north (bc_north=3): v(i, jev+1+ly) = 0 (3 layers)
        self._launch_const(fld, 1, True, NPL + n - 1, 0.0,
                           NPL - 1, n + 2, NPL - 1, n + 2, 3)
        # bottom (bc_bottom=4): v(i, j, ks-1-ly) = -v(i, j, ks+ly)
        self._launch_lin(fld, 2, False, NPL - 1, NPL, -1.0,
                         NPL - 1, n + 2, NPL - 1, n + 1, 3)
        # top 'cavity': v(i, j, kev+1+ly) = +v(i, j, kev-ly)
        self._launch_lin(fld, 2, True, NPL + n, NPL + n - 1, 1.0,
                         NPL - 1, n + 2, NPL - 1, n + 1, 3)

    def apply_w(self, fld, n, alfabc):
        # west (bc_west=4): w(is-1-ly) = -w(is+ly),
        # j in [js-1, je+1], k in [ks-1, ke] (kew is reduced)
        self._launch_lin(fld, 0, False, NPL - 1, NPL, -1.0,
                         NPL - 1, n + 2, NPL - 1, n + 1, 3)
        # east (bc_east=4): w(ie+1+ly) = -w(ie-ly)
        self._launch_lin(fld, 0, True, NPL + n, NPL + n - 1, -1.0,
                         NPL - 1, n + 2, NPL - 1, n + 1, 3)
        # south (bc_south=3): w(i, js-1-ly) = +w(i, js+ly)
        self._launch_lin(fld, 1, False, NPL - 1, NPL, 1.0,
                         NPL - 1, n + 2, NPL - 1, n + 1, 3)
        # north (bc_north=3): w(i, je+1+ly) = +w(i, je-ly)
        self._launch_lin(fld, 1, True, NPL + n, NPL + n - 1, 1.0,
                         NPL - 1, n + 2, NPL - 1, n + 1, 3)
        # bottom (bc_bottom=4): w(i, j, ks-1-ly) = 0
        self._launch_const(fld, 2, False, NPL - 1, 0.0,
                           NPL - 1, n + 2, NPL - 1, n + 2, 3)
        # top 'cavity' (kew reduced): w(i, j, kew+1+ly) = 0
        self._launch_const(fld, 2, True, NPL + n - 1, 0.0,
                           NPL - 1, n + 2, NPL - 1, n + 2, 3)

    def apply_vis(self, fld, n, rrey):
        """eddyvis_wale.for ghost section (one layer each side); at the
        west, east and bottom walls the first real plane is forced to rrey."""
        # west (bc=4): vis(is) = rrey; vis(is-1) = vis(is)
        self._launch_const(fld, 0, False, NPL, rrey,
                           NPL - 1, n + 2, NPL - 1, n + 2, 1)
        self._launch_lin(fld, 0, False, NPL - 1, NPL, 1.0,
                         NPL - 1, n + 2, NPL - 1, n + 2, 1)
        # east (bc=4): vis(ie) = rrey; vis(ie+1) = vis(ie)
        self._launch_const(fld, 0, True, NPL + n - 1, rrey,
                           NPL - 1, n + 2, NPL - 1, n + 2, 1)
        self._launch_lin(fld, 0, True, NPL + n, NPL + n - 1, 1.0,
                         NPL - 1, n + 2, NPL - 1, n + 2, 1)
        # south (bc=3): mirror
        self._launch_lin(fld, 1, False, NPL - 1, NPL, 1.0,
                         NPL - 1, n + 2, NPL - 1, n + 2, 1)
        # north (bc=3): mirror
        self._launch_lin(fld, 1, True, NPL + n, NPL + n - 1, 1.0,
                         NPL - 1, n + 2, NPL - 1, n + 2, 1)
        # bottom (bc=4): vis(ks) = rrey; vis(ks-1) = vis(ks)
        self._launch_const(fld, 2, False, NPL, rrey,
                           NPL - 1, n + 2, NPL - 1, n + 2, 1)
        self._launch_lin(fld, 2, False, NPL - 1, NPL, 1.0,
                         NPL - 1, n + 2, NPL - 1, n + 2, 1)
        # top (bc_t=1 -> else branch): mirror
        self._launch_lin(fld, 2, True, NPL + n, NPL + n - 1, 1.0,
                         NPL - 1, n + 2, NPL - 1, n + 2, 1)

    def apply_pbound(self, fld, n):
        """press.for pbound: ONE Neumann layer on every side."""
        perp = (NPL - 1, n + 3)
        for axis in (0, 1, 2):
            self._launch_lin(fld, axis, False, NPL - 1, NPL, 1.0,
                             *perp, *perp, 1)
            self._launch_lin(fld, axis, True, NPL + n, NPL + n - 1, 1.0,
                             *perp, *perp, 1)

    def apply_p_mirror(self, fld, n, npl):
        """Full-depth Neumann mirror on every side (used for the initial
        state, matching the restart field Hydro3D reads)."""
        for axis, high in ((0, False), (0, True), (1, False), (1, True),
                           (2, False), (2, True)):
            g0, s0 = (NPL - 1, NPL) if not high else (NPL + n, NPL + n - 1)
            self._launch_lin(fld, axis, high, g0, s0, 1.0,
                             0, n + 2 * NPL, 0, n + 2 * NPL, npl)

    def apply(self, fields, ncells, ulid, npl=3):
        """Initial full fill: boundu/v/w (alfabc=1) + mirrored p ghosts."""
        n = ncells[0]
        if "u" in fields:
            self.apply_u(fields["u"], n, ulid, 1)
        if "v" in fields:
            self.apply_v(fields["v"], n, 1)
        if "w" in fields:
            self.apply_w(fields["w"], n, 1)
        if "p" in fields:
            self.apply_p_mirror(fields["p"], n, npl)


def _mk_neumann(axis):
    """Generic one-sided Neumann ghost mirror (NPL-layer, cell-block form)
    used by the multigrid solver for its own work arrays."""
    sides = {"west": (0, False), "east": (0, True),
             "south": (1, False), "north": (1, True),
             "bottom": (2, False), "top": (2, True)}
    ax, high = sides[axis]

    def f(fld: wp.array3d(dtype=wp.float64), n: wp.int32, npl: wp.int32):
        a, b = wp.tid()
        for m in range(npl):
            if ax == 0:
                if high:
                    fld[npl + n + m, npl + a, npl + b] = \
                        fld[npl + n - 1 - m, npl + a, npl + b]
                else:
                    fld[npl - 1 - m, npl + a, npl + b] = \
                        fld[npl + m, npl + a, npl + b]
            elif ax == 1:
                if high:
                    fld[npl + a, npl + n + m, npl + b] = \
                        fld[npl + a, npl + n - 1 - m, npl + b]
                else:
                    fld[npl + a, npl - 1 - m, npl + b] = \
                        fld[npl + a, npl + m, npl + b]
            else:
                if high:
                    fld[npl + a, npl + b, npl + n + m] = \
                        fld[npl + a, npl + b, npl + n - 1 - m]
                else:
                    fld[npl + a, npl + b, npl - 1 - m] = \
                        fld[npl + a, npl + b, npl + m]

    f = wp.kernel(f)
    return f
