"""Two-level Lorenz-96 system adapted to the shared closure framework."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from closure_inference import ClosureModelSpec


SYSTEM_NAME = "lorenz96"
DISPLAY_NAME = "Two-level Lorenz-96"


@dataclass(frozen=True)
class L96Config:
    n_slow: int = 36
    n_fast: int = 10
    forcing: float = 10 #10 
    h: float = 1.0
    b: float = 10.0
    c: float = 10.0
    dt: float = 5.0e-4
    t_start: float = 0.0
    t_end: float = 40.0
    seed: int = 10

    @property
    def coupling(self):
        return self.h * self.c / self.b

    @property
    def time(self):
        return np.arange(self.t_start, self.t_end, self.dt, dtype=float)


def default_config():
    return L96Config()


def default_model_specs():
    return [
        ClosureModelSpec("local-linear", "local", 1),
        ClosureModelSpec("local-quadratic", "local", 2),
        ClosureModelSpec("r1-linear", "r1", 1),
        ClosureModelSpec("r1-quadratic", "r1", 2),
        ClosureModelSpec("r2-linear", "r2", 1),
        ClosureModelSpec("r2-quadratic", "r2", 2),
    ]


def state_labels(cfg):
    return [rf"$X_{{{k+1}}}$" for k in range(cfg.n_slow)]


def closure_label(cfg):
    return r"$G_i=\sum_jY_{j,i}$"


# def initial_conditions(cfg):
#     rng = np.random.default_rng(cfg.seed)

#     x0 = cfg.forcing + 0.5 * rng.standard_normal(cfg.n_slow)

#     y0 = (
#         x0[np.newaxis, :] / cfg.c
#         + 0.1 * rng.standard_normal((cfg.n_fast, cfg.n_slow))
#     )

#     return x0, y0

# def initial_conditions(cfg):
#     rng = np.random.default_rng(cfg.seed)

#     # Slow variables: smaller perturbation around forcing equilibrium
#     x0 = cfg.forcing + 0.05 * rng.standard_normal(cfg.n_slow)

#     # Fast variables: retain enough variation to avoid synchronization
#     y0 = (
#         x0[np.newaxis, :] / cfg.c
#         + 0.02 * rng.standard_normal((cfg.n_fast, cfg.n_slow))
#     )

#     return x0, y0
# def initial_conditions(cfg): 
#     rng = np.random.default_rng(cfg.seed) 
#     x0 = rng.standard_normal(cfg.n_slow) 
#     y0 = np.tile(x0, (cfg.n_fast, 1)) / cfg.c 
#     return x0, y0

def initial_conditions(cfg):
    rng = np.random.default_rng(cfg.seed)
    # Controlled slow-scale initial condition
    x0 = rng.standard_normal(cfg.n_slow)

    # Fast-scale variables with small random perturbations
    y0 = (
        np.tile(x0, (cfg.n_fast, 1)) / cfg.c
        + 0.002 * rng.standard_normal((cfg.n_fast, cfg.n_slow))
    )
    return x0, y0

def slow_tendency(x, y, cfg):
    g = np.sum(y, axis=0)
    nonlinear = np.roll(x, 1) * (np.roll(x, -1) - np.roll(x, 2))
    return nonlinear - x - cfg.coupling * g + cfg.forcing


def fast_tendency(x, y, cfg):
    fast_nonlinear = (
        cfg.c * cfg.b * np.roll(y, -1, axis=0)
        * (np.roll(y, 1, axis=0) - np.roll(y, -2, axis=0))
    )
    return fast_nonlinear - cfg.c * y + cfg.coupling * x[None, :]


def rk4_step(x, y, cfg):
    dt = cfg.dt
    k1x = slow_tendency(x, y, cfg)
    k1y = fast_tendency(x, y, cfg)

    k2x = slow_tendency(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, cfg)
    k2y = fast_tendency(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y, cfg)

    k3x = slow_tendency(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, cfg)
    k3y = fast_tendency(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y, cfg)

    k4x = slow_tendency(x + dt * k3x, y + dt * k3y, cfg)
    k4y = fast_tendency(x + dt * k3x, y + dt * k3y, cfg)

    x_new = x + dt / 6.0 * (k1x + 2*k2x + 2*k3x + k4x)
    y_new = y + dt / 6.0 * (k1y + 2*k2y + 2*k3y + k4y)
    return x_new, y_new


def generate_truth(cfg):
    t = cfg.time
    x0, y0 = initial_conditions(cfg)
    x = np.empty((len(t), cfg.n_slow))
    y = np.empty((len(t), cfg.n_fast, cfg.n_slow))
    x[0], y[0] = x0, y0

    for n in range(1, len(t)):
        x[n], y[n] = rk4_step(x[n-1], y[n-1], cfg)

    g_true = np.sum(y, axis=1)
    return t, x, y, g_true


def add_observation_noise(x_true, noise_std, seed):
    rng = np.random.default_rng(seed)
    return np.asarray(x_true) + noise_std * rng.standard_normal(np.asarray(x_true).shape)


def resolved_only_tendency_without_closure(x, cfg):
    x = np.asarray(x, dtype=float)
    nonlinear = np.roll(x, 1, axis=-1) * (
        np.roll(x, -1, axis=-1) - np.roll(x, 2, axis=-1)
    )
    return nonlinear - x + cfg.forcing


def recover_closure_from_x(
    x_obs,
    cfg,
    derivative: Literal["gradient", "savgol"] = "gradient",
    sg_window: int = 101,
    sg_polyorder: int = 3,
):
    if derivative == "gradient":
        dxdt = np.gradient(x_obs, cfg.dt, axis=0, edge_order=2)
    elif derivative == "savgol":
        window = int(sg_window)
        if window % 2 == 0:
            window += 1
        window = min(window, x_obs.shape[0] - (1 - x_obs.shape[0] % 2))
        dxdt = savgol_filter(
            x_obs, window_length=window, polyorder=sg_polyorder,
            deriv=1, delta=cfg.dt, axis=0, mode="interp"
        )
    else:
        raise ValueError(f"Unknown derivative method: {derivative}")

    known = resolved_only_tendency_without_closure(x_obs, cfg)
    return (known - dxdt) / cfg.coupling


def resolved_tendency_with_closure(x, g, cfg):
    return resolved_only_tendency_without_closure(x, cfg) - cfg.coupling * np.asarray(g)


def integrate_with_prescribed_closure(t, x0, g_time_series, cfg, **kwargs):
    g = np.asarray(g_time_series, dtype=float)
    expected = (len(t), cfg.n_slow)
    if g.shape != expected:
        raise ValueError(f"L96 closure must have shape {expected}; got {g.shape}.")
    g_fun = interp1d(
        t, g, axis=0, kind="linear",
        bounds_error=False,
        fill_value=(g[0], g[-1]),
        assume_sorted=True,
    )

    sol = solve_ivp(
        lambda time, x: resolved_tendency_with_closure(x, g_fun(time), cfg),
        (float(t[0]), float(t[-1])),
        np.asarray(x0, dtype=float),
        t_eval=t,
        rtol=kwargs.get("rtol", 1e-7),
        atol=kwargs.get("atol", 1e-9),
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def _radius(spec):
    mapping = {"local": 0, "r1": 1, "r2": 2}
    if spec.variant not in mapping:
        raise ValueError(f"Unknown L96 variant: {spec.variant}")
    return mapping[spec.variant]


def _poly_features(values, degree):
    values = np.asarray(values, dtype=float)
    feats = [1.0]
    feats.extend(values.tolist())
    if degree == 2:
        for i, j in combinations_with_replacement(range(values.size), 2):
            feats.append(values[i] * values[j])
    return np.asarray(feats)


def build_design_matrix(x, spec):
    x = np.asarray(x, dtype=float)
    radius = _radius(spec)
    rows = []
    for n in range(x.shape[0]):
        for k in range(x.shape[1]):
            values = [
                x[n, (k + off) % x.shape[1]]
                for off in range(-radius, radius + 1)
            ]
            rows.append(_poly_features(values, spec.degree))
    return np.vstack(rows)


def closure_feature_matrix(x, spec):
    x = np.asarray(x, dtype=float)
    radius = _radius(spec)
    rows = []
    for k in range(x.size):
        values = [
            x[(k + off) % x.size]
            for off in range(-radius, radius + 1)
        ]
        rows.append(_poly_features(values, spec.degree))
    return np.vstack(rows)


def closure_output_shape(x):
    x = np.asarray(x)
    return x.shape


def instantaneous_closure_shape(x):
    return np.asarray(x).shape
