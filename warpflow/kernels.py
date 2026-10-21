"""NVIDIA Warp kernels for warpflow-les.

Numerical core ported from Hydro3D v7.0 (Fortran):
  - rungek.for  : 4th-order CDS convection (rungek_conv4th) and diffusion
                  (rungek_diff4th), Heun RK2 stage updates
  - press.for   : calvel (pressure correction), calmas (4th-order divergence)
  - eddyvis_wale.for : WALE sub-grid viscosity (cw = 0.46)

Grid layout (identical to Hydro3D): every field, including the staggered
velocity components, uses the same index range over cells plus NPL ghost
layers on each side.  u(i,j,k) is the +x face of cell (i,j,k), v(i,j,k) the
+y face, w(i,j,k) the +z face; p and vis are cell-centred.

All float literals are wrapped in wp.float64() explicitly: Warp's implicit
widening of bare float32 literals in mixed expressions is not reliable.
"""
import warp as wp

wp.init()

NPL = 3  # ghost layers (Hydro3D pl)

F = wp.float64  # shorthand for literal typing


# ---------------------------------------------------------------------------
# 4th-order interpolation helper mirrored from rungek_conv4th:
#   I(x) = (-f(x+1) + 9 f(x) + 9 f(x-1) - f(x-2)) / 16
# ---------------------------------------------------------------------------

@wp.func
def _ix(f: wp.array3d(dtype=wp.float64), i: wp.int32, j: wp.int32, k: wp.int32):
    return (-f[i + 1, j, k] + F(9.0) * f[i, j, k]
            + F(9.0) * f[i - 1, j, k] - f[i - 2, j, k]) / F(16.0)


@wp.func
def _iy(f: wp.array3d(dtype=wp.float64), i: wp.int32, j: wp.int32, k: wp.int32):
    return (-f[i, j + 1, k] + F(9.0) * f[i, j, k]
            + F(9.0) * f[i, j - 1, k] - f[i, j - 2, k]) / F(16.0)


@wp.func
def _iz(f: wp.array3d(dtype=wp.float64), i: wp.int32, j: wp.int32, k: wp.int32):
    return (-f[i, j, k + 1] + F(9.0) * f[i, j, k]
            + F(9.0) * f[i, j, k - 1] - f[i, j, k - 2]) / F(16.0)


# ---------------------------------------------------------------------------
# convection: ustar = uoo - dt*alpha*N(u), 4th-order CDS (rungek_conv4th)
# ---------------------------------------------------------------------------

@wp.kernel
def conv4th_u(u: wp.array3d(dtype=wp.float64), v: wp.array3d(dtype=wp.float64),
              w: wp.array3d(dtype=wp.float64), uoo: wp.array3d(dtype=wp.float64),
              ustar: wp.array3d(dtype=wp.float64),
              dx: wp.float64, dy: wp.float64, dz: wp.float64,
              dt: wp.float64, alpha: wp.float64, npl: wp.int32):
    i, j, k = wp.tid()
    i += npl
    j += npl
    k += npl

    # du2/dx
    sp2 = _ix(u, i + 2, j, k) * _ix(u, i + 2, j, k)
    sp1 = _ix(u, i + 1, j, k) * _ix(u, i + 1, j, k)
    sn = _ix(u, i, j, k) * _ix(u, i, j, k)
    sm1 = _ix(u, i - 1, j, k) * _ix(u, i - 1, j, k)
    du2dx = (-sp2 + F(27.0) * sp1 - F(27.0) * sn + sm1) / (F(24.0) * dx)

    # duv/dy
    sp1 = _iy(u, i, j + 1, k) * _ix(v, i + 1, j + 1, k)
    sn = _iy(u, i, j, k) * _ix(v, i + 1, j, k)
    sm1 = _iy(u, i, j - 1, k) * _ix(v, i + 1, j - 1, k)
    sm2 = _iy(u, i, j - 2, k) * _ix(v, i + 1, j - 2, k)
    duvdy = (-sp1 + F(27.0) * sn - F(27.0) * sm1 + sm2) / (F(24.0) * dy)

    # duw/dz
    sp1 = _iz(u, i, j, k + 1) * _ix(w, i + 1, j, k + 1)
    sn = _iz(u, i, j, k) * _ix(w, i + 1, j, k)
    sm1 = _iz(u, i, j, k - 1) * _ix(w, i + 1, j, k - 1)
    sm2 = _iz(u, i, j, k - 2) * _ix(w, i + 1, j, k - 2)
    duwdz = (-sp1 + F(27.0) * sn - F(27.0) * sm1 + sm2) / (F(24.0) * dz)

    ustar[i, j, k] = uoo[i, j, k] - dt * alpha * (du2dx + duvdy + duwdz)


@wp.kernel
def conv4th_v(u: wp.array3d(dtype=wp.float64), v: wp.array3d(dtype=wp.float64),
              w: wp.array3d(dtype=wp.float64), voo: wp.array3d(dtype=wp.float64),
              vstar: wp.array3d(dtype=wp.float64),
              dx: wp.float64, dy: wp.float64, dz: wp.float64,
              dt: wp.float64, alpha: wp.float64, npl: wp.int32):
    i, j, k = wp.tid()
    i += npl
    j += npl
    k += npl

    # dv2/dy
    sp2 = _iy(v, i, j + 2, k) * _iy(v, i, j + 2, k)
    sp1 = _iy(v, i, j + 1, k) * _iy(v, i, j + 1, k)
    sn = _iy(v, i, j, k) * _iy(v, i, j, k)
    sm1 = _iy(v, i, j - 1, k) * _iy(v, i, j - 1, k)
    dv2dy = (-sp2 + F(27.0) * sp1 - F(27.0) * sn + sm1) / (F(24.0) * dy)

    # duv/dx
    sp1 = _ix(v, i + 1, j, k) * _iy(u, i + 1, j + 1, k)
    sn = _ix(v, i, j, k) * _iy(u, i, j + 1, k)
    sm1 = _ix(v, i - 1, j, k) * _iy(u, i - 1, j + 1, k)
    sm2 = _ix(v, i - 2, j, k) * _iy(u, i - 2, j + 1, k)
    duvdx = (-sp1 + F(27.0) * sn - F(27.0) * sm1 + sm2) / (F(24.0) * dx)

    # dvw/dz
    sp1 = _iz(v, i, j, k + 1) * _iy(w, i, j + 1, k + 1)
    sn = _iz(v, i, j, k) * _iy(w, i, j + 1, k)
    sm1 = _iz(v, i, j, k - 1) * _iy(w, i, j, k - 1)
    sm2 = _iz(v, i, j, k - 2) * _iy(w, i, j + 1, k - 2)
    dvwdz = (-sp1 + F(27.0) * sn - F(27.0) * sm1 + sm2) / (F(24.0) * dz)

    vstar[i, j, k] = voo[i, j, k] - dt * alpha * (duvdx + dv2dy + dvwdz)


@wp.kernel
def conv4th_w(u: wp.array3d(dtype=wp.float64), v: wp.array3d(dtype=wp.float64),
              w: wp.array3d(dtype=wp.float64), woo: wp.array3d(dtype=wp.float64),
              wstar: wp.array3d(dtype=wp.float64),
              dx: wp.float64, dy: wp.float64, dz: wp.float64,
              dt: wp.float64, alpha: wp.float64, npl: wp.int32):
    i, j, k = wp.tid()
    i += npl
    j += npl
    k += npl

    # dw2/dz
    sp2 = _iz(w, i, j, k + 2) * _iz(w, i, j, k + 2)
    sp1 = _iz(w, i, j, k + 1) * _iz(w, i, j, k + 1)
    sn = _iz(w, i, j, k) * _iz(w, i, j, k)
    sm1 = _iz(w, i, j, k - 1) * _iz(w, i, j, k - 1)
    dw2dz = (-sp2 + F(27.0) * sp1 - F(27.0) * sn + sm1) / (F(24.0) * dz)

    # duw/dx
    sp1 = _ix(w, i + 1, j, k) * _iz(u, i + 1, j, k + 1)
    sn = _ix(w, i, j, k) * _iz(u, i, j, k + 1)
    sm1 = _ix(w, i - 1, j, k) * _iz(u, i - 1, j, k + 1)
    sm2 = _ix(w, i - 2, j, k) * _iz(u, i - 2, j, k + 1)
    duwdx = (-sp1 + F(27.0) * sn - F(27.0) * sm1 + sm2) / (F(24.0) * dx)

    # dvw/dy
    sp1 = _iy(w, i, j + 1, k) * _iz(v, i, j + 1, k + 1)
    sn = _iy(w, i, j, k) * _iz(v, i, j, k + 1)
    sm1 = _iy(w, i, j - 1, k) * _iz(v, i, j - 1, k + 1)
    sm2 = _iy(w, i, j - 2, k) * _iz(v, i, j - 2, k + 1)
    dvwdy = (-sp1 + F(27.0) * sn - F(27.0) * sm1 + sm2) / (F(24.0) * dy)

    wstar[i, j, k] = woo[i, j, k] - dt * alpha * (duwdx + dvwdy + dw2dz)


# ---------------------------------------------------------------------------
# diffusion: ustar += dt*alpha*vis*L4(u), 4th-order Laplacian
# (rungek_diff4th; viscosity sampled as in Hydro3D: u@vis(i,j,k),
#  v@vis(i,j+1,k), w@vis(i,j,k+1))
# ---------------------------------------------------------------------------

@wp.func
def _lap4(f: wp.array3d(dtype=wp.float64), i: wp.int32, j: wp.int32,
          k: wp.int32, dx: wp.float64, dy: wp.float64, dz: wp.float64):
    lx = (-f[i + 2, j, k] + F(16.0) * f[i + 1, j, k] - F(30.0) * f[i, j, k]
          + F(16.0) * f[i - 1, j, k] - f[i - 2, j, k]) / (F(12.0) * dx * dx)
    ly = (-f[i, j + 2, k] + F(16.0) * f[i, j + 1, k] - F(30.0) * f[i, j, k]
          + F(16.0) * f[i, j - 1, k] - f[i, j - 2, k]) / (F(12.0) * dy * dy)
    lz = (-f[i, j, k + 2] + F(16.0) * f[i, j, k + 1] - F(30.0) * f[i, j, k]
          + F(16.0) * f[i, j, k - 1] - f[i, j, k - 2]) / (F(12.0) * dz * dz)
    return lx + ly + lz


@wp.kernel
def diff4th_u(u: wp.array3d(dtype=wp.float64), vis: wp.array3d(dtype=wp.float64),
              ustar: wp.array3d(dtype=wp.float64),
              dx: wp.float64, dy: wp.float64, dz: wp.float64,
              dt: wp.float64, alpha: wp.float64, npl: wp.int32):
    i, j, k = wp.tid()
    i += npl
    j += npl
    k += npl
    diff = vis[i, j, k] * _lap4(u, i, j, k, dx, dy, dz)
    ustar[i, j, k] = ustar[i, j, k] + dt * alpha * diff


@wp.kernel
def diff4th_v(v: wp.array3d(dtype=wp.float64), vis: wp.array3d(dtype=wp.float64),
              vstar: wp.array3d(dtype=wp.float64),
              dx: wp.float64, dy: wp.float64, dz: wp.float64,
              dt: wp.float64, alpha: wp.float64, npl: wp.int32):
    i, j, k = wp.tid()
    i += npl
    j += npl
    k += npl
    diff = vis[i, j + 1, k] * _lap4(v, i, j, k, dx, dy, dz)
    vstar[i, j, k] = vstar[i, j, k] + dt * alpha * diff


@wp.kernel
def diff4th_w(w: wp.array3d(dtype=wp.float64), vis: wp.array3d(dtype=wp.float64),
              wstar: wp.array3d(dtype=wp.float64),
              dx: wp.float64, dy: wp.float64, dz: wp.float64,
              dt: wp.float64, alpha: wp.float64, npl: wp.int32):
    i, j, k = wp.tid()
    i += npl
    j += npl
    k += npl
    diff = vis[i, j, k + 1] * _lap4(w, i, j, k, dx, dy, dz)
    wstar[i, j, k] = wstar[i, j, k] + dt * alpha * diff


# ---------------------------------------------------------------------------
# calvel (press.for): u = ustar - dt*alpha*grad(p)
# ---------------------------------------------------------------------------

@wp.kernel
def calvel_u(ustar: wp.array3d(dtype=wp.float64), p: wp.array3d(dtype=wp.float64),
             u: wp.array3d(dtype=wp.float64), dx: wp.float64,
             dt: wp.float64, alpha: wp.float64, npl: wp.int32):
    """u = ustar - dt*alpha*dp/dx.  u(full i) sits on the east face of
    p-cell i; the wall faces are ghost planes handled by the BC kernels."""
    i, j, k = wp.tid()
    i += npl
    j += npl
    k += npl
    u[i, j, k] = ustar[i, j, k] - dt * alpha * (p[i + 1, j, k] - p[i, j, k]) / dx


@wp.kernel
def calvel_v(vstar: wp.array3d(dtype=wp.float64), p: wp.array3d(dtype=wp.float64),
             v: wp.array3d(dtype=wp.float64), dy: wp.float64,
             dt: wp.float64, alpha: wp.float64, npl: wp.int32):
    i, j, k = wp.tid()
    i += npl
    j += npl
    k += npl
    v[i, j, k] = vstar[i, j, k] - dt * alpha * (p[i, j + 1, k] - p[i, j, k]) / dy


@wp.kernel
def calvel_w(wstar: wp.array3d(dtype=wp.float64), p: wp.array3d(dtype=wp.float64),
             w: wp.array3d(dtype=wp.float64), dz: wp.float64,
             dt: wp.float64, alpha: wp.float64, npl: wp.int32):
    i, j, k = wp.tid()
    i += npl
    j += npl
    k += npl
    w[i, j, k] = wstar[i, j, k] - dt * alpha * (p[i, j, k + 1] - p[i, j, k]) / dz


# ---------------------------------------------------------------------------
# calmas (press.for): 4th-order divergence, su = div/dt, rmax = max|div*dV|
# ---------------------------------------------------------------------------

@wp.kernel
def calmas(u: wp.array3d(dtype=wp.float64), v: wp.array3d(dtype=wp.float64),
           w: wp.array3d(dtype=wp.float64), su: wp.array3d(dtype=wp.float64),
           rmax: wp.array(dtype=wp.float64),
           dx: wp.float64, dy: wp.float64, dz: wp.float64,
           dt: wp.float64, npl: wp.int32):
    i, j, k = wp.tid()
    i += npl
    j += npl
    k += npl
    dudx = (-u[i + 1, j, k] + F(27.0) * u[i, j, k]
            - F(27.0) * u[i - 1, j, k] + u[i - 2, j, k]) / (F(24.0) * dx)
    dvdy = (-v[i, j + 1, k] + F(27.0) * v[i, j, k]
            - F(27.0) * v[i, j - 1, k] + v[i, j - 2, k]) / (F(24.0) * dy)
    dwdz = (-w[i, j, k + 1] + F(27.0) * w[i, j, k]
            - F(27.0) * w[i, j, k - 1] + w[i, j, k - 2]) / (F(24.0) * dz)
    fact = dudx + dvdy + dwdz
    su[i, j, k] = fact / dt
    wp.atomic_max(rmax, 0, wp.abs(fact * dx * dy * dz))


# ---------------------------------------------------------------------------
# WALE sub-grid viscosity (eddyvis_wale.for, cw = 0.46)
# vis = 1/Re + nut at cell centres
# ---------------------------------------------------------------------------

@wp.kernel
def wale_viscosity(u: wp.array3d(dtype=wp.float64), v: wp.array3d(dtype=wp.float64),
                   w: wp.array3d(dtype=wp.float64), vis: wp.array3d(dtype=wp.float64),
                   dx: wp.float64, dy: wp.float64, dz: wp.float64,
                   rrey: wp.float64, cw: wp.float64, npl: wp.int32):
    i, j, k = wp.tid()
    i += npl
    j += npl
    k += npl

    # velocity gradient g[a][b] = d c_a / d x_b at cell centre
    dudx = (u[i, j, k] - u[i - 1, j, k]) / dx
    dudy = (F(0.5) * (u[i, j + 1, k] + u[i, j + 1, k + 1])
            - F(0.5) * (u[i, j, k] + u[i, j, k + 1])) / dy
    dudz = (F(0.5) * (u[i, j, k + 1] + u[i, j + 1, k + 1])
            - F(0.5) * (u[i, j, k] + u[i, j + 1, k])) / dz

    dvdx = (F(0.5) * (v[i + 1, j, k] + v[i + 1, j, k + 1])
            - F(0.5) * (v[i, j, k] + v[i, j, k + 1])) / dx
    dvdy = (v[i, j, k] - v[i, j - 1, k]) / dy
    dvdz = (F(0.5) * (v[i, j, k + 1] + v[i + 1, j, k + 1])
            - F(0.5) * (v[i, j, k] + v[i + 1, j, k])) / dz

    dwdx = (F(0.5) * (w[i + 1, j, k] + w[i + 1, j + 1, k])
            - F(0.5) * (w[i, j, k] + w[i, j + 1, k])) / dx
    dwdy = (F(0.5) * (w[i, j + 1, k] + w[i + 1, j + 1, k])
            - F(0.5) * (w[i, j, k] + w[i + 1, j, k])) / dy
    dwdz = (w[i, j, k] - w[i, j, k - 1]) / dz

    # g2_ab = g_ak g_kb  (traceless-symmetric part of gg^T)
    g2xx = dudx * dudx + dudy * dvdx + dudz * dwdx
    g2yy = dvdx * dudy + dvdy * dvdy + dvdz * dwdy
    g2zz = dwdx * dudz + dwdy * dvdz + dwdz * dwdz
    g2xy = dudx * dudy + dudy * dvdy + dudz * dwdy
    g2yx = dvdx * dudx + dvdy * dvdx + dvdz * dwdx
    g2xz = dudx * dudz + dudy * dvdz + dudz * dwdz
    g2zx = dwdx * dudx + dwdy * dvdx + dwdz * dudz
    g2yz = dvdx * dudz + dvdy * dvdz + dvdz * dwdz
    g2zy = dwdx * dudy + dwdy * dvdy + dwdz * dudy

    tr = (g2xx + g2yy + g2zz) / F(3.0)
    sdxx = g2xx - tr
    sdyy = g2yy - tr
    sdzz = g2zz - tr
    sdxy = F(0.5) * (g2xy + g2yx)
    sdxz = F(0.5) * (g2xz + g2zx)
    sdyz = F(0.5) * (g2yz + g2zy)

    sdsd = (sdxx * sdxx + sdyy * sdyy + sdzz * sdzz
            + F(2.0) * (sdxy * sdxy + sdxz * sdxz + sdyz * sdyz))

    sxx = dudx
    syy = dvdy
    szz = dwdz
    sxy = F(0.5) * (dudy + dvdx)
    sxz = F(0.5) * (dudz + dwdx)
    syz = F(0.5) * (dvdz + dwdy)
    ss = (sxx * sxx + syy * syy + szz * szz
          + F(2.0) * (sxy * sxy + sxz * sxz + syz * syz))

    delta = wp.pow(dx * dy * dz, F(1.0) / F(3.0))
    ls = cw * delta
    sdsd15 = wp.pow(sdsd, F(1.5))
    denom = wp.pow(wp.abs(ss), F(2.5)) + wp.pow(sdsd, F(1.25)) + F(1.0e-20)
    nut = ls * ls * sdsd15 / denom

    vis[i, j, k] = rrey + nut


# ---------------------------------------------------------------------------
# reductions
# ---------------------------------------------------------------------------

@wp.kernel
def maxabs_reduce(f: wp.array3d(dtype=wp.float64), out: wp.array(dtype=wp.float64),
                  npl: wp.int32):
    i, j, k = wp.tid()
    i += npl
    j += npl
    k += npl
    wp.atomic_max(out, 0, wp.abs(f[i, j, k]))


@wp.kernel
def dtvisc_reduce(u: wp.array3d(dtype=wp.float64), v: wp.array3d(dtype=wp.float64),
                  w: wp.array3d(dtype=wp.float64), vis: wp.array3d(dtype=wp.float64),
                  out: wp.array(dtype=wp.float64),
                  dx: wp.float64, dy: wp.float64, dz: wp.float64, npl: wp.int32):
    """Cell-centred advective+viscous dt limit (checkdt.for, SGS branch)."""
    i, j, k = wp.tid()
    i += npl
    j += npl
    k += npl
    uc = F(0.5) * (u[i, j, k] + u[i - 1, j, k])
    vc = F(0.5) * (v[i, j, k] + v[i, j - 1, k])
    wc = F(0.5) * (w[i, j, k] + w[i, j, k - 1])
    val = F(1.0) / (wp.abs(uc) / dx + wp.abs(vc) / dy + wp.abs(wc) / dz
                    + F(2.0) * vis[i, j, k]
                    * (F(1.0) / (dx * dx) + F(1.0) / (dy * dy)
                       + F(1.0) / (dz * dz))
                    + F(1.0e-30))
    wp.atomic_min(out, 0, val)


@wp.kernel
def add_inplace(dst: wp.array3d(dtype=wp.float64), src: wp.array3d(dtype=wp.float64),                npl: wp.int32):
    """dst += src over real cells plus one ghost ring
    (newsolv_mg: p(isp-1:iep+1,...) += pp; the pp ring carries the
    multigrid boundary values mgbound/exchangepp left on cof)."""
    i, j, k = wp.tid()
    i += npl - 1
    j += npl - 1
    k += npl - 1
    dst[i, j, k] = dst[i, j, k] + src[i, j, k]


# ---------------------------------------------------------------------------
# correctoutflux (initial.for): scale one outlet face plane by fct
# ---------------------------------------------------------------------------

@wp.kernel
def scale_x_plane(f: wp.array3d(dtype=wp.float64), plane: wp.int32,
                  b0: wp.int32, fct: wp.float64):
    j, k = wp.tid()
    f[plane, b0 + j, b0 + k] = f[plane, b0 + j, b0 + k] * fct


@wp.kernel
def scale_y_plane(f: wp.array3d(dtype=wp.float64), plane: wp.int32,
                  b0: wp.int32, fct: wp.float64):
    i, k = wp.tid()
    f[b0 + i, plane, b0 + k] = f[b0 + i, plane, b0 + k] * fct


@wp.kernel
def scale_z_plane(f: wp.array3d(dtype=wp.float64), plane: wp.int32,
                  b0: wp.int32, fct: wp.float64):
    i, j = wp.tid()
    f[b0 + i, b0 + j, plane] = f[b0 + i, b0 + j, plane] * fct


@wp.kernel
def fill_const(f: wp.array3d(dtype=wp.float64), val: wp.float64, npl: wp.int32):
    """Real-cell range of f set to a constant."""
    i, j, k = wp.tid()
    i += npl
    j += npl
    k += npl
    f[i, j, k] = val


@wp.kernel
def init_noise(u: wp.array3d(dtype=wp.float64), v: wp.array3d(dtype=wp.float64),
               w: wp.array3d(dtype=wp.float64),
               amp: wp.float64, seed: wp.int32, npl: wp.int32):
    i, j, k = wp.tid()
    i += npl
    j += npl
    k += npl
    # xorshift-style deterministic hash noise, uniform in [-amp, amp]
    ui = wp.uint32(i)
    uj = wp.uint32(j)
    uk = wp.uint32(k)
    h = (ui * wp.uint32(73856093)) ^ (uj * wp.uint32(19349663)) ^ (uk * wp.uint32(83492791)) ^ wp.uint32(seed)
    h = h ^ (h >> wp.uint32(13))
    h = h * wp.uint32(1274126177)
    h = h ^ (h >> wp.uint32(16))
    r = (wp.float64(h) / wp.float64(4294967295.0)) * F(2.0) - F(1.0)
    u[i, j, k] = amp * r
    v[i, j, k] = amp * r
    w[i, j, k] = -amp * r