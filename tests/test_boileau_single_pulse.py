"""Boileau et al. 2015 (Int J Numer Meth Biomed Eng 31:e02732) test case 1: single pulse in a 10 m reflection-free
tube, inviscid. Parameters from the paper: A0 = pi cm^2, h = 1.5 mm, E = 400 kPa, rho = 1050, Qin = 1e-6 exp(-1e4 (t-0.05)^2),
reference c0 = 6.17 m/s, absorbing outlet. Linear theory: Q(x,t) = Qin(t - x/c0), P = Z0 Q with Z0 = rho c0 / A0.
Tube law constants match ours with sigma = 0.5: beta = (4/3) sqrt(pi) E h. Viscous variant not tested (friction closure differs).
"""
import math

import numpy as np
import pytest

from physics.solver_1d import build_network

RHO, A0, H, E = 1050.0, math.pi * 1e-4, 1.5e-3, 4.0e5
R0 = math.sqrt(A0 / math.pi)
C0_PAPER = 6.17


def test_single_pulse_inviscid():
    n_tab, period = 30000, 3.0
    tt = np.arange(n_tab) / n_tab * period
    wtab = 1e-6 * np.exp(-1e4 * (tt - 0.05) ** 2)
    v = dict(name="a", L=10.0, r0=R0, h=H, E=E)
    beta = math.sqrt(math.pi) * H * E / 0.75
    c0 = math.sqrt(beta / (2 * RHO * math.sqrt(A0)))
    assert c0 == pytest.approx(C0_PAPER, abs=0.01)
    z0 = RHO * c0 / A0
    net = build_network([v], [], {"a": 1.0}, {"a": dict(R1=z0, R2=1e12, C=1e-6)}, rho=RHO, mu=0.0, sigma=0.5,
                        pv=0.0, wtab=wtab, period=period, dx_target=2.0e-3)
    net.init_uniform_pressure(0.0)
    dt = net.stable_dt(0.8, u_max=0.01, a_ratio=1.001)
    n = int(1.9 / dt)
    _, rA, rQ = net.advance(n, dt, rec_every=1, probes=[("a", 0.5), ("a", 0.97)])
    t = (np.arange(n) + 1) * dt
    for k, x in enumerate((5.0, 9.7)):
        q_ref = 1e-6 * np.exp(-1e4 * (t - x / c0 - 0.05) ** 2)
        err = np.linalg.norm(rQ[:, k] - q_ref) / np.linalg.norm(q_ref)
        assert err < 0.03, (x, err)  # shape, timing and amplitude after 5 m and 9.7 m
        p = beta * (np.sqrt(rA[:, k]) - math.sqrt(A0)) / A0
        assert p.max() == pytest.approx(z0 * 1e-6, rel=0.03)
