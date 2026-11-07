"""1D pulsatile blood-flow solver for arterial networks (replaces the Gate 0 legacy solver).

Model (per vessel, state A = area, Q = flow):
    dA/dt + dQ/dx = 0
    dQ/dt + d/dx( Q^2/A + b/(3 rho A0) A^{3/2} ) = -K_R Q/A,     K_R = 8 pi mu / rho
    tube law  P = (b/A0) (sqrt(A) - sqrt(A0)),   b = sqrt(pi) h E / (1 - sigma^2)
    wave speed c = c0 (A/A0)^{1/4},  c0^2 = b / (2 rho sqrt(A0))   (= tube-law consistent)

Numerics:
  * cell-centred finite volume, Richtmyer two-step Lax-Wendroff interface fluxes (2nd order
    interior), explicit friction;
  * boundary / junction states from Riemann invariants W1,2 = u +- 4c (outgoing invariant traced
    back along its characteristic, linear interpolation), solved with Newton:
      - inlet:    prescribed Q(t)
      - junction: conservation of mass + continuity of total pressure P + rho u^2 / 2
      - outlet:   three-element Windkessel (R1 = Z0 by default, R2, C), backward-Euler in time
  * boundary flux for a step = average of boundary-state fluxes at t_n and t_{n+1}, so the
    discrete volume balance holds to round-off:
        sum(cell volume change) = integral(inflow) - integral(outflow)

Units: SI (m, s, kg, Pa, m^3/s).
Verification lives in tests/test_solver_1d.py.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
from numba import njit

PI = math.pi


# ----------------------------------------------------------------------------------------
# numba kernels
# ----------------------------------------------------------------------------------------
@njit(cache=True)
def _wave(t, wtab, period):
    """Periodic linear interpolation of the inflow waveform table."""
    m = wtab.shape[0]
    x = ((t / period) % 1.0) * m
    i = int(x)
    f = x - i
    return wtab[i % m] * (1.0 - f) + wtab[(i + 1) % m] * f


@njit(cache=True)
def _solve_inlet(qin, w2, a0, c0, a_guess):
    a = a_guess
    for _ in range(60):
        c = c0 * (a / a0) ** 0.25
        f = qin / a - 4.0 * c - w2
        df = -qin / (a * a) - c / a
        an = a - f / df
        if an < 0.2 * a0:
            an = 0.5 * (a + 0.2 * a0)
        if abs(an - a) < 1e-13 * a0:
            return an
        a = an
    return a


@njit(cache=True)
def _solve_outlet(w1, a0, c0, b, rho, reff, pconst, a_guess):
    a = a_guess
    for _ in range(60):
        c = c0 * (a / a0) ** 0.25
        u = w1 - 4.0 * c
        p = b * (math.sqrt(a) - math.sqrt(a0)) / a0
        h = p - reff * a * u - pconst
        dh = rho * c * c / a - reff * (u - c)  # d(A u)/dA = u + A du/dA = u - c
        an = a - h / dh
        if an < 0.2 * a0:
            an = 0.5 * (a + 0.2 * a0)
        if abs(an - a) < 1e-13 * a0:
            return an
        a = an
    return a


@njit(cache=True)
def _solve_junction(n, ve, wv, a0v, c0v, bv, rho, aguess, aout):
    """Unknowns: area at each of the n vessel ends. ve[k]=1 right end (flow into node), 0 left.
    wv[k] = outgoing Riemann invariant. Writes areas into aout."""
    x = np.empty(n)
    for k in range(n):
        x[k] = aguess[k]
    J = np.empty((n, n))
    r = np.empty(n)
    u = np.empty(n)
    dudA = np.empty(n)
    ptot = np.empty(n)
    dpt = np.empty(n)
    for _ in range(60):
        for k in range(n):
            c = c0v[k] * (x[k] / a0v[k]) ** 0.25
            if ve[k] == 1:
                u[k] = wv[k] - 4.0 * c
                dudA[k] = -c / x[k]
            else:
                u[k] = wv[k] + 4.0 * c
                dudA[k] = c / x[k]
            p = bv[k] * (math.sqrt(x[k]) - math.sqrt(a0v[k])) / a0v[k]
            ptot[k] = p + 0.5 * rho * u[k] * u[k]
            dpt[k] = rho * c * c / x[k] + rho * u[k] * dudA[k]
        for i in range(n):
            for j in range(n):
                J[i, j] = 0.0
        s0 = 0.0
        for k in range(n):
            sg = 1.0 if ve[k] == 1 else -1.0
            s0 += sg * x[k] * u[k]
            J[0, k] = sg * (u[k] + x[k] * dudA[k])
        r[0] = s0
        for k in range(1, n):
            r[k] = ptot[k] - ptot[0]
            J[k, k] = dpt[k]
            J[k, 0] = -dpt[0]
        # solve J d = -r (Gaussian elimination, partial pivoting)
        for i in range(n):
            r[i] = -r[i]
        for col in range(n):
            piv = col
            for i in range(col + 1, n):
                if abs(J[i, col]) > abs(J[piv, col]):
                    piv = i
            if piv != col:
                for j in range(n):
                    tmp = J[col, j]
                    J[col, j] = J[piv, j]
                    J[piv, j] = tmp
                tmp = r[col]
                r[col] = r[piv]
                r[piv] = tmp
            for i in range(col + 1, n):
                f = J[i, col] / J[col, col]
                for j in range(col, n):
                    J[i, j] -= f * J[col, j]
                r[i] -= f * r[col]
        d = np.empty(n)
        for i in range(n - 1, -1, -1):
            s = r[i]
            for j in range(i + 1, n):
                s -= J[i, j] * d[j]
            d[i] = s / J[i, i]
        scale = 1.0
        for k in range(n):
            if x[k] + d[k] < 0.2 * a0v[k]:
                sc = 0.5 * (x[k] - 0.2 * a0v[k]) / (-d[k])
                if sc < scale:
                    scale = sc
        dmax = 0.0
        for k in range(n):
            x[k] += scale * d[k]
            v = abs(scale * d[k]) / a0v[k]
            if v > dmax:
                dmax = v
        if dmax < 1e-13:
            break
    for k in range(n):
        aout[k] = x[k]


@njit(cache=True)
def _step(t, dt, rho, kfr, pv,
          A, Q, off, ncell, dx, A0, bco, c0,
          Ab, Qb,
          jn_cnt, jn_v, jn_e,
          in_v, in_scale, wtab, period,
          out_v, R1, R2, Cw, Pc,
          Fa, Fq, Wout, Abn, Qbn, An, Qn, acc):
    """Advance all cells by dt. acc = [vol_in, vol_out, max_courant, bad_flag]."""
    nv = ncell.shape[0]
    # 1) interior interface fluxes (Richtmyer)
    for v in range(nv):
        n = ncell[v]
        o = off[v]
        fo = o + v
        a0 = A0[v]
        b = bco[v]
        k3 = b / (3.0 * rho * a0)
        h = dt / (2.0 * dx[v])
        for j in range(n - 1):
            a1 = A[o + j]
            q1 = Q[o + j]
            a2 = A[o + j + 1]
            q2 = Q[o + j + 1]
            f1q = q1 * q1 / a1 + k3 * a1 * math.sqrt(a1)
            f2q = q2 * q2 / a2 + k3 * a2 * math.sqrt(a2)
            s1 = -kfr * q1 / a1
            s2 = -kfr * q2 / a2
            am = 0.5 * (a1 + a2) - h * (q2 - q1)
            qm = 0.5 * (q1 + q2) - h * (f2q - f1q) + 0.5 * dt * 0.5 * (s1 + s2)
            Fa[fo + j + 1] = qm
            Fq[fo + j + 1] = qm * qm / am + k3 * am * math.sqrt(am)
    # 2) outgoing invariants traced to the boundaries
    for v in range(nv):
        n = ncell[v]
        o = off[v]
        a0 = A0[v]
        c00 = c0[v]
        # left end: W2 = u - 4c, travels with u - c
        u0 = Q[o] / A[o]
        cc0 = c00 * (A[o] / a0) ** 0.25
        u1 = Q[o + 1] / A[o + 1]
        cc1 = c00 * (A[o + 1] / a0) ** 0.25
        s = (cc0 - u0) * dt
        wf = (u0 - 4.0 * cc0) + ((u1 - 4.0 * cc1) - (u0 - 4.0 * cc0)) * (s - 0.5 * dx[v]) / dx[v]
        Wout[v, 0] = wf - dt * kfr * u0 / A[o]
        # right end: W1 = u + 4c, travels with u + c
        ua = Q[o + n - 1] / A[o + n - 1]
        ca = c00 * (A[o + n - 1] / a0) ** 0.25
        ub = Q[o + n - 2] / A[o + n - 2]
        cb = c00 * (A[o + n - 2] / a0) ** 0.25
        s = (ua + ca) * dt
        wf = (ua + 4.0 * ca) + ((ub + 4.0 * cb) - (ua + 4.0 * ca)) * (s - 0.5 * dx[v]) / dx[v]
        Wout[v, 1] = wf - dt * kfr * ua / A[o + n - 1]
    # 3) boundary states at t+dt
    for k in range(in_v.shape[0]):
        v = in_v[k]
        qin = in_scale[k] * _wave(t + dt, wtab, period)
        a = _solve_inlet(qin, Wout[v, 0], A0[v], c0[v], Ab[v, 0])
        Abn[v, 0] = a
        Qbn[v, 0] = qin
    for k in range(out_v.shape[0]):
        v = out_v[k]
        theta = dt / (Cw[k] * R2[k])
        a1c = 1.0 / (1.0 + theta)
        a2c = (dt / Cw[k]) / (1.0 + theta)
        a3c = theta / (1.0 + theta)
        reff = R1[k] + a2c
        pconst = a1c * Pc[k] + a3c * pv
        a = _solve_outlet(Wout[v, 1], A0[v], c0[v], bco[v], rho, reff, pconst, Ab[v, 1])
        c = c0[v] * (a / A0[v]) ** 0.25
        q = a * (Wout[v, 1] - 4.0 * c)
        Abn[v, 1] = a
        Qbn[v, 1] = q
        Pc[k] = a1c * Pc[k] + a2c * q + a3c * pv
    for j in range(jn_cnt.shape[0]):
        m = jn_cnt[j]
        ve = np.empty(m, dtype=np.int64)
        wv = np.empty(m)
        a0v = np.empty(m)
        c0v = np.empty(m)
        bv = np.empty(m)
        ag = np.empty(m)
        ao = np.empty(m)
        for k in range(m):
            v = jn_v[j, k]
            e = jn_e[j, k]
            ve[k] = e
            wv[k] = Wout[v, e]
            a0v[k] = A0[v]
            c0v[k] = c0[v]
            bv[k] = bco[v]
            ag[k] = Ab[v, e]
        _solve_junction(m, ve, wv, a0v, c0v, bv, rho, ag, ao)
        for k in range(m):
            v = jn_v[j, k]
            e = jn_e[j, k]
            c = c0[v] * (ao[k] / A0[v]) ** 0.25
            if e == 1:
                u = wv[k] - 4.0 * c
            else:
                u = wv[k] + 4.0 * c
            Abn[v, e] = ao[k]
            Qbn[v, e] = ao[k] * u
    # 4) boundary flux averages and cell update
    cmax = acc[2]
    for v in range(nv):
        n = ncell[v]
        o = off[v]
        fo = o + v
        a0 = A0[v]
        k3 = bco[v] / (3.0 * rho * a0)
        for e in range(2):
            a_old = Ab[v, e]
            q_old = Qb[v, e]
            a_new = Abn[v, e]
            q_new = Qbn[v, e]
            fqo = q_old * q_old / a_old + k3 * a_old * math.sqrt(a_old)
            fqn = q_new * q_new / a_new + k3 * a_new * math.sqrt(a_new)
            idx = fo if e == 0 else fo + n
            Fa[idx] = 0.5 * (q_old + q_new)
            Fq[idx] = 0.5 * (fqo + fqn)
        r = dt / dx[v]
        for j in range(n):
            a = A[o + j]
            q = Q[o + j]
            An[o + j] = a - r * (Fa[fo + j + 1] - Fa[fo + j])
            Qn[o + j] = q - r * (Fq[fo + j + 1] - Fq[fo + j]) - dt * kfr * q / a
            lam = abs(q / a) + c0[v] * (a / a0) ** 0.25
            cr = lam * dt / dx[v]
            if cr > cmax:
                cmax = cr
            if An[o + j] < 0.05 * a0 or not np.isfinite(An[o + j]) or not np.isfinite(Qn[o + j]):
                acc[3] = 1.0
    acc[2] = cmax
    for k in range(in_v.shape[0]):
        v = in_v[k]
        acc[0] += dt * 0.5 * (Qb[v, 0] + Qbn[v, 0])
    for k in range(out_v.shape[0]):
        v = out_v[k]
        acc[1] += dt * 0.5 * (Qb[v, 1] + Qbn[v, 1])
    for i in range(A.shape[0]):
        A[i] = An[i]
        Q[i] = Qn[i]
    for v in range(nv):
        for e in range(2):
            Ab[v, e] = Abn[v, e]
            Qb[v, e] = Qbn[v, e]


@njit(cache=True)
def _integrate(nsteps, t0, dt, rho, kfr, pv,
               A, Q, off, ncell, dx, A0, bco, c0,
               Ab, Qb,
               jn_cnt, jn_v, jn_e,
               in_v, in_scale, wtab, period,
               out_v, R1, R2, Cw, Pc,
               Fa, Fq, Wout, Abn, Qbn, An, Qn, acc,
               rec_every, probe_cells, recA, recQ):
    nrec = 0
    for s in range(nsteps):
        t = t0 + s * dt
        _step(t, dt, rho, kfr, pv, A, Q, off, ncell, dx, A0, bco, c0, Ab, Qb,
              jn_cnt, jn_v, jn_e, in_v, in_scale, wtab, period,
              out_v, R1, R2, Cw, Pc, Fa, Fq, Wout, Abn, Qbn, An, Qn, acc)
        if acc[3] > 0.0:
            return s + 1
        if rec_every > 0 and (s + 1) % rec_every == 0:
            for p in range(probe_cells.shape[0]):
                recA[nrec, p] = A[probe_cells[p]]
                recQ[nrec, p] = Q[probe_cells[p]]
            nrec += 1
    return nsteps


# ----------------------------------------------------------------------------------------
# Python-side network description and driver
# ----------------------------------------------------------------------------------------
@dataclass
class Network:
    names: list
    rho: float
    mu: float
    pv: float
    off: np.ndarray
    ncell: np.ndarray
    dx: np.ndarray
    length: np.ndarray
    r0: np.ndarray
    A0: np.ndarray
    bco: np.ndarray
    c0: np.ndarray
    jn_cnt: np.ndarray
    jn_v: np.ndarray
    jn_e: np.ndarray
    in_v: np.ndarray
    in_scale: np.ndarray
    wtab: np.ndarray
    period: float
    out_v: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    Cw: np.ndarray
    A: np.ndarray = field(default=None)
    Q: np.ndarray = field(default=None)
    Ab: np.ndarray = field(default=None)
    Qb: np.ndarray = field(default=None)
    Pc: np.ndarray = field(default=None)
    t: float = 0.0
    acc: np.ndarray = field(default=None)
    friction_scale: float = 1.0  # 1 = parabolic 8 pi mu/rho; 2.53 = Boileau zeta = 9 closure

    @property
    def kfr(self):
        return self.friction_scale * 8.0 * PI * self.mu / self.rho

    def vessel_index(self, name):
        return self.names.index(name)

    def cells(self, name):
        v = self.vessel_index(name)
        return slice(int(self.off[v]), int(self.off[v] + self.ncell[v]))

    def x_centres(self, name):
        v = self.vessel_index(name)
        return (np.arange(self.ncell[v]) + 0.5) * self.dx[v]

    def pressure(self, A, name):
        v = self.vessel_index(name)
        return self.bco[v] * (np.sqrt(A) - np.sqrt(self.A0[v])) / self.A0[v]

    def total_volume(self):
        return float(np.sum(self.A * np.repeat(self.dx, self.ncell)))

    def init_uniform_pressure(self, p_init, q_init=0.0):
        a0 = np.repeat(self.A0, self.ncell)
        b = np.repeat(self.bco, self.ncell)
        self.A = (np.sqrt(a0) + p_init * a0 / b) ** 2
        self.Q = np.full_like(self.A, q_init)
        nv = len(self.names)
        self.Ab = np.empty((nv, 2))
        self.Qb = np.zeros((nv, 2))
        for v in range(nv):
            self.Ab[v, 0] = self.A[self.off[v]]
            self.Ab[v, 1] = self.A[self.off[v] + self.ncell[v] - 1]
            self.Qb[v, :] = q_init
        self.Pc = np.full(len(self.out_v), float(p_init))
        self.t = 0.0
        self.acc = np.zeros(4)

    def stable_dt(self, cfl=0.8, u_max=3.0, a_ratio=1.15):
        lam = a_ratio ** 0.25 * self.c0 + u_max
        return cfl * float(np.min(self.dx / lam))

    def advance(self, nsteps, dt, rec_every=0, probes=None):
        """Advance nsteps of size dt. probes=[(vessel_name, x_fraction)] recorded every rec_every
        steps; returns (steps_done, recA, recQ)."""
        nv = len(self.names)
        nprobe = 0 if probes is None else len(probes)
        if rec_every > 0 and nprobe > 0:
            cells = []
            for name, xf in probes:
                v = self.vessel_index(name)
                j = min(int(xf * self.ncell[v]), int(self.ncell[v]) - 1)
                cells.append(int(self.off[v]) + j)
            probe_cells = np.array(cells, dtype=np.int64)
            nr = nsteps // rec_every
            recA = np.zeros((nr, nprobe))
            recQ = np.zeros((nr, nprobe))
        else:
            probe_cells = np.zeros(0, dtype=np.int64)
            recA = np.zeros((0, 0))
            recQ = np.zeros((0, 0))
        n_if = int(self.A.shape[0]) + nv
        Fa = np.zeros(n_if)
        Fq = np.zeros(n_if)
        Wout = np.zeros((nv, 2))
        Abn = np.zeros((nv, 2))
        Qbn = np.zeros((nv, 2))
        An = np.zeros_like(self.A)
        Qn = np.zeros_like(self.Q)
        done = _integrate(
            nsteps, self.t, dt, self.rho, self.kfr, self.pv,
            self.A, self.Q, self.off, self.ncell, self.dx, self.A0, self.bco, self.c0,
            self.Ab, self.Qb, self.jn_cnt, self.jn_v, self.jn_e,
            self.in_v, self.in_scale, self.wtab, self.period,
            self.out_v, self.R1, self.R2, self.Cw, self.Pc,
            Fa, Fq, Wout, Abn, Qbn, An, Qn, self.acc,
            rec_every, probe_cells, recA, recQ)
        self.t += done * dt
        if done < nsteps:
            raise FloatingPointError(f"solver blew up after {done} steps (area <5% of A0 or non-finite)")
        return done, recA, recQ


def build_network(vessels, nodes, inlets, outlets, rho=1050.0, mu=4.5e-3, sigma=0.5,
                  pv=666.612, dx_target=1.0e-3, min_cells=3, wtab=None, period=1.0):
    """vessels: list of dict(name, L, r0, h, E[, ncell]); nodes: list of lists of (vessel, end)
    with end 0 = left (x=0), 1 = right (x=L); inlets: {vessel: flow_scale} on the left end;
    outlets: {vessel: dict(R_total=, C=) or dict(R1=, R2=, C=)} on the right end."""
    names = [v["name"] for v in vessels]
    nv = len(names)
    ncell = np.array([v.get("ncell") or max(min_cells, int(round(v["L"] / dx_target))) for v in vessels],
                     dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(ncell)[:-1]]).astype(np.int64)
    length = np.array([v["L"] for v in vessels])
    dx = length / ncell
    r0 = np.array([v["r0"] for v in vessels])
    A0 = PI * r0 ** 2
    h = np.array([v["h"] for v in vessels])
    E = np.array([v["E"] for v in vessels])
    bco = math.sqrt(PI) * h * E / (1.0 - sigma ** 2)
    c0 = np.sqrt(bco / (2.0 * rho * np.sqrt(A0)))

    ends_used = {}
    for j, node in enumerate(nodes):
        for (vn, e) in node:
            key = (vn, e)
            if key in ends_used:
                raise ValueError(f"vessel end {key} used twice")
            ends_used[key] = ("node", j)
    for vn in inlets:
        if (vn, 0) in ends_used:
            raise ValueError(f"inlet end {(vn, 0)} also used elsewhere")
        ends_used[(vn, 0)] = ("inlet", vn)
    for vn in outlets:
        if (vn, 1) in ends_used:
            raise ValueError(f"outlet end {(vn, 1)} also used elsewhere")
        ends_used[(vn, 1)] = ("outlet", vn)
    for vn in names:
        for e in (0, 1):
            if (vn, e) not in ends_used:
                raise ValueError(f"vessel end {(vn, e)} has no boundary condition or junction")

    nj = len(nodes)
    maxm = max([len(n) for n in nodes] + [2])
    jn_cnt = np.array([len(n) for n in nodes], dtype=np.int64)
    jn_v = np.zeros((max(nj, 1), maxm), dtype=np.int64)
    jn_e = np.zeros((max(nj, 1), maxm), dtype=np.int64)
    for j, node in enumerate(nodes):
        if len(node) < 2:
            raise ValueError("junction needs at least two vessel ends")
        for k, (vn, e) in enumerate(node):
            jn_v[j, k] = names.index(vn)
            jn_e[j, k] = e
    in_v = np.array([names.index(vn) for vn in inlets], dtype=np.int64)
    in_scale = np.array([inlets[vn] for vn in inlets], dtype=float)
    out_v = np.array([names.index(vn) for vn in outlets], dtype=np.int64)
    R1 = np.zeros(len(outlets))
    R2 = np.zeros(len(outlets))
    Cw = np.zeros(len(outlets))
    for k, vn in enumerate(outlets):
        v = names.index(vn)
        spec = outlets[vn]
        z0 = rho * c0[v] / A0[v]
        if "R_total" in spec:
            R1[k] = z0
            # R_total below Z0 would make R2 <= 0 (perturbed small stiff vessels): floor R2 at 0.2 Z0
            R2[k] = max(spec["R_total"] - z0, 0.2 * z0)
        else:
            R1[k] = spec["R1"]
            R2[k] = spec["R2"]
        Cw[k] = spec["C"]
    if wtab is None:
        wtab = np.ones(1)
    return Network(names=names, rho=rho, mu=mu, pv=pv, off=off, ncell=ncell, dx=dx, length=length,
                   r0=r0, A0=A0, bco=bco, c0=c0, jn_cnt=jn_cnt, jn_v=jn_v, jn_e=jn_e,
                   in_v=in_v, in_scale=in_scale, wtab=np.asarray(wtab, dtype=float), period=float(period),
                   out_v=out_v, R1=R1, R2=R2, Cw=Cw)


def simulate_periodic(net, p_init, cfl=0.8, n_out=100, min_cycles=3, max_cycles=30, tol=1e-4,
                      progress=None, pc_init=None):
    """Run to a periodic state, then record one cycle of cell fields at n_out times.

    Returns dict with t, per-vessel (x, A, Q, P) arrays (n_out x ncell) and diagnostics
    (cycles, cycle_change, volume balance, max Courant, wall time)."""
    t_wall = time.perf_counter()
    net.init_uniform_pressure(p_init)
    if pc_init is not None:  # pure-resistor terminals (huge R2*C) never relax: start the capacitor at the outlet pressure
        net.Pc[:] = pc_init
    dt0 = net.stable_dt(cfl)
    steps_per_cycle = int(math.ceil(net.period / dt0 / n_out)) * n_out
    dt = net.period / steps_per_cycle
    v_start = net.total_volume()
    prev = None
    cycles = 0
    change = float("inf")
    while cycles < max_cycles:
        net.advance(steps_per_cycle, dt)
        cycles += 1
        cur = np.concatenate([net.A / np.repeat(net.A0, net.ncell), net.Q / np.repeat(net.A0, net.ncell) / 1.0])
        if prev is not None:
            change = float(np.linalg.norm(cur - prev) / np.linalg.norm(cur))
            if cycles >= min_cycles and change < tol:
                break
        prev = cur
        if progress:
            progress(cycles, change)
    # record one more cycle
    per = steps_per_cycle // n_out
    ts = np.zeros(n_out)
    snaps_A = np.zeros((n_out, net.A.shape[0]))
    snaps_Q = np.zeros((n_out, net.A.shape[0]))
    for i in range(n_out):
        net.advance(per, dt)
        ts[i] = net.t
        snaps_A[i] = net.A
        snaps_Q[i] = net.Q
    out = {"t": ts - ts[0] + 0.0, "dt": dt, "names": list(net.names)}
    for v, name in enumerate(net.names):
        sl = slice(int(net.off[v]), int(net.off[v] + net.ncell[v]))
        out[name] = {
            "x": net.x_centres(name),
            "A": snaps_A[:, sl].copy(),
            "Q": snaps_Q[:, sl].copy(),
            "P": net.bco[v] * (np.sqrt(snaps_A[:, sl]) - np.sqrt(net.A0[v])) / net.A0[v],
        }
    out["diagnostics"] = {
        "cycles_to_converge": cycles,
        "cycle_change": change,
        "converged": bool(change < tol),
        "max_courant": float(net.acc[2]),
        "vol_in": float(net.acc[0]),
        "vol_out": float(net.acc[1]),
        "vol_stored_change": net.total_volume() - v_start,
        "wall_s": time.perf_counter() - t_wall,
    }
    return out
