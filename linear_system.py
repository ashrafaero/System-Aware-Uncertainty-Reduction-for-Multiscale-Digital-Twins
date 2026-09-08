"""Canonical two-state linear slow-fast system with one hidden variable.

Full system
-----------
dx/dt = a x + b y
dy/dt = (c x + d y) / epsilon

Resolved variable: x
Hidden variable:   y
Unresolved closure in the x equation: G = b y

The model-discovery stage receives only x(t), reconstructs G from dx/dt-a*x,
and fits G_hat(x).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Literal

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from closure_inference import ClosureModelSpec


SYSTEM_NAME = "linear"
DISPLAY_NAME = "Linear slow-fast system"


@dataclass(frozen=True)
class LinearConfig:
    a: float = -0.5
    b: float = 1.0
    c: float = 0.5
    d: float = -2.0
    epsilon: float = 0.25
    dt: float = 2.0e-3
    t_start: float = 0.0
    t_end: float = 20.0
    seed: int = 10

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.t_start, self.t_end, self.dt, dtype=float)


def default_config() -> LinearConfig:
    return LinearConfig()


def default_model_specs() -> list[ClosureModelSpec]:
    return [
        ClosureModelSpec("linear", variant="x", degree=1),
        # ClosureModelSpec("quadratic", variant="x", degree=2),
    ]


def state_labels(cfg: LinearConfig) -> list[str]:
    return [r"$x$"]


def closure_label(cfg: LinearConfig) -> str:
    return r"$G=b\,y$"


def _full_rhs(t, state, cfg):
    x, y = state
    return np.array([
        cfg.a * x + cfg.b * y,
        (cfg.c * x + cfg.d * y) / cfg.epsilon,
    ])


def initial_conditions(cfg: LinearConfig) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    # Seed-dependence allows a meaningful independent-trajectory test.
    return np.array([
        1.0 + 0.15 * rng.standard_normal(),
        -0.5 + 0.15 * rng.standard_normal(),
    ])


def generate_truth(cfg: LinearConfig):
    t = cfg.time
    state0 = initial_conditions(cfg)
    sol = solve_ivp(
        lambda time, state: _full_rhs(time, state, cfg),
        (float(t[0]), float(t[-1])),
        state0,
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    full = sol.y.T
    x = full[:, [0]]
    hidden = full[:, [1]]
    g_true = cfg.b * hidden
    return t, x, hidden, g_true


def add_observation_noise(x_true, noise_std, seed):
    rng = np.random.default_rng(seed)
    return np.asarray(x_true) + noise_std * rng.standard_normal(np.asarray(x_true).shape)


def _derivative(x_obs, cfg, derivative, sg_window, sg_polyorder):
    if derivative == "gradient":
        return np.gradient(x_obs, cfg.dt, axis=0, edge_order=2)
    if derivative == "savgol":
        window = int(sg_window)
        if window % 2 == 0:
            window += 1
        window = min(window, x_obs.shape[0] - (1 - x_obs.shape[0] % 2))
        return savgol_filter(
            x_obs,
            window_length=window,
            polyorder=sg_polyorder,
            deriv=1,
            delta=cfg.dt,
            axis=0,
            mode="interp",
        )
    raise ValueError(f"Unknown derivative method: {derivative}")


def recover_closure_from_x(
    x_obs,
    cfg,
    derivative: Literal["gradient", "savgol"] = "gradient",
    sg_window: int = 101,
    sg_polyorder: int = 3,
):
    dxdt = _derivative(x_obs, cfg, derivative, sg_window, sg_polyorder)
    return dxdt - cfg.a * x_obs


def resolved_tendency_with_closure(x, g, cfg):
    x = np.asarray(x, dtype=float)
    g_scalar = float(np.asarray(g).reshape(-1)[0])
    return np.array([cfg.a * x[0] + g_scalar])


def integrate_with_prescribed_closure(t, x0, g_time_series, cfg, **kwargs):
    g = np.asarray(g_time_series, dtype=float)
    if g.shape != (len(t), 1):
        raise ValueError("Linear-system closure must have shape (len(t), 1).")
    g_fun = interp1d(
        t, g[:, 0], kind="linear",
        bounds_error=False,
        fill_value=(g[0, 0], g[-1, 0]),
    )
    sol = solve_ivp(
        lambda time, x: np.array([cfg.a * x[0] + float(g_fun(time))]),
        (float(t[0]), float(t[-1])),
        np.asarray(x0, dtype=float),
        t_eval=t,
        rtol=kwargs.get("rtol", 1e-8),
        atol=kwargs.get("atol", 1e-10),
    )
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


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
    rows = [_poly_features([value], spec.degree) for value in x[:, 0]]
    return np.vstack(rows)


def closure_feature_matrix(x, spec):
    x = np.asarray(x, dtype=float)
    return _poly_features([x[0]], spec.degree)[None, :]


def closure_output_shape(x):
    return (np.asarray(x).shape[0], 1)


def instantaneous_closure_shape(x):
    return (1,)
