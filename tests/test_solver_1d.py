"""Verification of physics/solver_1d.py against analytic limits."""
import math

import numpy as np
import pytest

from physics.solver_1d import build_network

RHO = 1050.0


def _vessel(name="a", L=0.2, r0=2.0e-3, h=3.0e-4, E=1.6e6, **kw):
    return dict(name=name, L=L, r0=r0, h=h, E=E, **kw)


def _gauss(n=2000, centre=0.1, width=0.01, amp=1.0):
    t = np.arange(n) / n
    return amp * np.exp(-0.5 * ((t - centre) / width) ** 2)


def _absorbing(net_vessel_c0_a0, c=1e-6):
    c0, a0 = net_vessel_c0_a0
    return dict(R1=RHO * c0 / a0, R2=1e12, C=c)


def _c0_a0(v, sigma=0.5):
    b = math.sqrt(math.pi) * v["h"] * v["E"] / (1 - sigma ** 2)
    a0 = math.pi * v["r0"] ** 2
    return math.sqrt(b / (2 * RHO * math.sqrt(a0))), a0


def test_wave_speed_matches_tube_law():
    """Small pulse arrival time between two probes gives c0 = sqrt(2/3 E h/(rho r0))."""
    v = _vessel(L=1.0)
    c0, a0 = _c0_a0(v)
    assert c0 == pytest.approx(math.sqrt(2.0 / 3.0 * v["E"] * v["h"] / (RHO * v["r0"])), rel=1e-12)
    net = build_network([v], [], {"a": 1.0}, {"a": _absorbing((c0, a0))},
                        wtab=_gauss(amp=2e-7), period=1.0, dx_target=2e-3, mu=0.0)
    net.init_uniform_pressure(0.0)
    dt = net.stable_dt(0.5, u_max=0.1, a_ratio=1.01)
    n = int(0.25 / dt)
    _, rA, _ = net.advance(n, dt, rec_every=1, probes=[("a", 0.25), ("a", 0.75)])
    t = (np.arange(n) + 1) * dt
    p1 = rA[:, 0] - a0
    p2 = rA[:, 1] - a0
    t1 = np.sum(t * p1 ** 2) / np.sum(p1 ** 2)
    t2 = np.sum(t * p2 ** 2) / np.sum(p2 ** 2)
    c_meas = 0.5 * v["L"] / (t2 - t1)
    assert c_meas == pytest.approx(c0, rel=0.02)


def test_poiseuille_pressure_drop():
    """Steady flow: dP/dx = 8 pi mu Q / A^2."""
    v = _vessel(L=0.3, r0=1.5e-3)
    c0, a0 = _c0_a0(v)
    q = 2e-6
    net = build_network([v], [], {"a": 1.0}, {"a": dict(R1=RHO * c0 / a0, R2=2e9, C=1e-12)},
                        wtab=np.array([q]), period=1.0, dx_target=3e-3)
    net.init_uniform_pressure(1.0e3)
    dt = net.stable_dt(0.8)
    net.advance(int(3.0 / dt), dt)
    x = net.x_centres("a")
    P = net.pressure(net.A, "a")
    sl = slice(10, -10)
    slope = np.polyfit(x[sl], P[sl], 1)[0]
    a_mid = float(np.mean(net.A[sl]))
    expected = -8 * math.pi * net.mu * q / a_mid ** 2
    assert slope == pytest.approx(expected, rel=0.03)
    assert np.allclose(net.Q, q, rtol=2e-3)


def _bifurcation(wtab, period=1.0, mu=0.0, dx=2e-3):
    p = _vessel("p", L=1.0, r0=2.0e-3)
    d1 = _vessel("d1", L=0.5, r0=1.4e-3)
    d2 = _vessel("d2", L=0.5, r0=1.2e-3)
    outs = {n: _absorbing(_c0_a0(v)) for n, v in (("d1", d1), ("d2", d2))}
    net = build_network([p, d1, d2], [[("p", 1), ("d1", 0), ("d2", 0)]], {"p": 1.0}, outs,
                        wtab=wtab, period=period, dx_target=dx, mu=mu)
    return net, (p, d1, d2)


def test_bifurcation_reflection_coefficient():
    net, (p, d1, d2) = _bifurcation(_gauss(amp=2e-7))
    ys = []
    for v in (p, d1, d2):
        c0, a0 = _c0_a0(v)
        ys.append(a0 / (RHO * c0))
    r_theory = (ys[0] - ys[1] - ys[2]) / (ys[0] + ys[1] + ys[2])
    t_theory = 2 * ys[0] / (ys[0] + ys[1] + ys[2])
    net.init_uniform_pressure(0.0)
    dt = net.stable_dt(0.5, u_max=0.1, a_ratio=1.01)
    n = int(0.9 / dt)
    _, rA, _ = net.advance(n, dt, rec_every=1, probes=[("p", 0.3), ("d1", 0.5)])
    a0p = p["r0"] ** 2 * math.pi
    a0d = d1["r0"] ** 2 * math.pi
    bp = math.sqrt(math.pi) * p["h"] * p["E"] / 0.75
    bd = math.sqrt(math.pi) * d1["h"] * d1["E"] / 0.75
    Pp = bp * (np.sqrt(rA[:, 0]) - math.sqrt(a0p)) / a0p
    Pd = bd * (np.sqrt(rA[:, 1]) - math.sqrt(a0d)) / a0d
    tt = (np.arange(n) + 1) * dt
    c0p, _ = _c0_a0(p)
    t_arrive_in = 0.1 + 0.3 * p["L"] / c0p
    t_arrive_ref = 0.1 + (p["L"] + 0.7 * p["L"]) / c0p
    inc = Pp[(tt > t_arrive_in - 0.05) & (tt < t_arrive_in + 0.05)].max()
    ref = Pp[(tt > t_arrive_ref - 0.05) & (tt < t_arrive_ref + 0.05)].max()
    assert ref / inc == pytest.approx(r_theory, abs=0.03)
    trans = Pd.max()
    assert trans / inc == pytest.approx(t_theory, rel=0.05)


def test_global_volume_balance_bifurcation():
    wt = np.maximum(0.0, np.sin(np.linspace(0, 2 * np.pi, 400, endpoint=False))) * 3e-6
    net, _ = _bifurcation(wt, period=0.8, mu=4.5e-3)
    net.init_uniform_pressure(1.0e3)
    v0 = net.total_volume()
    dt = net.stable_dt(0.8)
    net.advance(int(0.8 / dt) * 2, dt)
    vin, vout = net.acc[0], net.acc[1]
    dv = net.total_volume() - v0
    assert vin > 1e-7
    assert abs(vin - vout - dv) / vin < 1e-6


def test_periodic_state_reached():
    from physics.solver_1d import simulate_periodic
    wt = np.maximum(0.0, np.sin(np.linspace(0, 2 * np.pi, 400, endpoint=False))) * 3e-6
    p = _vessel("p", L=0.1)
    c0, a0 = _c0_a0(p)
    net = build_network([p], [], {"p": 1.0}, {"p": dict(R_total=4e9, C=2e-10)}, wtab=wt, period=0.8,
                        dx_target=2e-3)
    out = simulate_periodic(net, 1.0e3, n_out=50, max_cycles=40, tol=1e-5)
    assert out["diagnostics"]["converged"]
    # one-cycle volume in = out at periodic state
    # periodic state: mean inflow over the recorded cycle equals mean outflow
    qin = out["p"]["Q"][:, 0].mean()
    qout = out["p"]["Q"][:, -1].mean()
    assert qin == pytest.approx(qout, rel=2e-2)


@pytest.mark.parametrize("fine,coarse", [(160, 40)])
def test_convergence_order(fine, coarse):
    """Smooth pulse, error vs a fine reference decays at observed order >= 1.5."""
    wt = 1e-6 * (1 + 0.5 * np.sin(2 * np.pi * np.arange(400) / 400))

    def run(ncell):
        v = _vessel("a", L=0.4, ncell=ncell)
        c0, a0 = _c0_a0(v)
        net = build_network([v], [], {"a": 1.0}, {"a": dict(R1=RHO * c0 / a0, R2=3e9, C=1e-10)},
                            wtab=wt, period=0.4, mu=0.0)
        net.init_uniform_pressure(1.0e3)
        dt = 0.4 / 32000
        net.advance(int(0.8 / dt), dt)
        x = net.x_centres("a")
        return x, net.pressure(net.A, "a")

    xr, pr = run(640)

    def err(n):
        x, p = run(n)
        pi = np.interp(x, xr, pr)
        return float(np.sqrt(np.mean((p - pi) ** 2)))

    e1, e2 = err(coarse), err(coarse * 2)
    order = math.log(e1 / e2) / math.log(2.0)
    assert order > 1.5, (e1, e2, order)
