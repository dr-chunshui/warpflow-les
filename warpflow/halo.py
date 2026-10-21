"""MPI halo exchange for device arrays (numpy staging).

Ghost layers toward neighbouring sub-domains are exchanged sequentially along
x, then y, then z: the y (z) exchange sends full rows/planes including the
x (x,y) ghost layers already received, so edge and corner ghosts are filled
without diagonal communication - the same scheme as Hydro3D's exchange().

Physical-boundary ghosts are NOT touched here; they are filled by the
boundary-condition kernels in bcs.py.
"""
from mpi4py import MPI

import numpy as np
import warp as wp


@wp.kernel
def _gather_x(arr: wp.array3d(dtype=wp.float64), out: wp.array(dtype=wp.float64),
              ofs: wp.int32, B: wp.int32, C: wp.int32):
    m, j, k = wp.tid()
    out[(m * B + j) * C + k] = arr[ofs + m, j, k]


@wp.kernel
def _scatter_x(arr: wp.array3d(dtype=wp.float64), inp: wp.array(dtype=wp.float64),
               ofs: wp.int32, B: wp.int32, C: wp.int32):
    m, j, k = wp.tid()
    arr[ofs + m, j, k] = inp[(m * B + j) * C + k]


@wp.kernel
def _gather_y(arr: wp.array3d(dtype=wp.float64), out: wp.array(dtype=wp.float64),
              ofs: wp.int32, nl: wp.int32, C: wp.int32):
    i, m, k = wp.tid()
    out[(i * nl + m) * C + k] = arr[i, ofs + m, k]


@wp.kernel
def _scatter_y(arr: wp.array3d(dtype=wp.float64), inp: wp.array(dtype=wp.float64),
               ofs: wp.int32, nl: wp.int32, C: wp.int32):
    i, m, k = wp.tid()
    arr[i, ofs + m, k] = inp[(i * nl + m) * C + k]


@wp.kernel
def _gather_z(arr: wp.array3d(dtype=wp.float64), out: wp.array(dtype=wp.float64),
              ofs: wp.int32, nl: wp.int32, B: wp.int32):
    i, j, m = wp.tid()
    out[(i * B + j) * nl + m] = arr[i, j, ofs + m]


@wp.kernel
def _scatter_z(arr: wp.array3d(dtype=wp.float64), inp: wp.array(dtype=wp.float64),
               ofs: wp.int32, nl: wp.int32, B: wp.int32):
    i, j, m = wp.tid()
    arr[i, j, ofs + m] = inp[(i * B + j) * nl + m]


class HaloExchanger:
    """Exchanges `npl` ghost layers of device array3d fields over MPI."""

    def __init__(self, topo, device, npl):
        self.topo = topo
        self.device = device
        self.npl = npl
        self._stage = {}  # packed size -> (np_send, np_recv, dev_send, dev_recv)

    def _buffers(self, size):
        if size not in self._stage:
            self._stage[size] = (
                np.empty(size, dtype=np.float64),
                np.empty(size, dtype=np.float64),
                wp.empty(size, dtype=wp.float64, device=self.device),
                wp.empty(size, dtype=wp.float64, device=self.device),
            )
        return self._stage[size]

    def exchange(self, arr, nl=None):
        """Fill neighbour-side ghost layers of `arr` (all three axes)."""
        if nl is None:
            nl = self.npl
        npl = self.npl
        A, B, C = arr.shape
        for axis in range(3):
            lo, hi = self.topo.neighbors["xyz"[axis]]
            if lo < 0 and hi < 0:
                continue
            if axis == 0:
                n = A - 2 * npl
                cnt = nl * B * C
                gdim, sdim = (nl, B, C), (nl, B, C)
                gker, sker = _gather_x, _scatter_x
                kargs = (B, C)
            elif axis == 1:
                n = B - 2 * npl
                cnt = A * nl * C
                gdim, sdim = (A, nl, C), (A, nl, C)
                gker, sker = _gather_y, _scatter_y
                kargs = (nl, C)
            else:
                n = C - 2 * npl
                cnt = A * B * nl
                gdim, sdim = (A, B, nl), (A, B, nl)
                gker, sker = _gather_z, _scatter_z
                kargs = (nl, B)
            send_lo, send_hi = npl, npl + n - nl
            recv_lo, recv_hi = npl - nl, npl + n

            # with the low neighbour: send my low block, receive its high
            # block into my low ghosts (and symmetrically for high).
            # Tags are paired by DATA FLOW: "block sent to the low
            # neighbour" (21) is received by that neighbour as "block
            # received from its high neighbour", and vice versa (22).
            pairs = []
            if lo >= 0:
                pairs.append((lo, send_lo, recv_lo, 21, 22))
            if hi >= 0:
                pairs.append((hi, send_hi, recv_hi, 22, 21))
            for peer, send_ofs, recv_ofs, stag, rtag in pairs:
                snd, rcv, dev_s, dev_r = self._buffers(cnt)
                wp.launch(gker, dim=gdim,
                          inputs=[arr, dev_s, send_ofs, *kargs],
                          device=self.device)
                snd[:] = dev_s.numpy()
                self.topo.comm.Sendrecv(snd, dest=peer, sendtag=stag,
                                        recvbuf=rcv, source=peer, recvtag=rtag)
                dev_r.assign(rcv)
                wp.launch(sker, dim=sdim,
                          inputs=[arr, dev_r, recv_ofs, *kargs],
                          device=self.device)
