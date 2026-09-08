"""Partially observed Lorenz-63 with a physics-informed dynamic closure state.

Resolved equations
------------------
    xdot = sigma (y - x)
    ydot = rho x - y + G

The exact unresolved forcing is G=-xz.  Instead of forcing G to be an
instantaneous algebraic function of (x,y), introduce a dynamic latent state q:

    G_hat = -x q
    qdot  = F(x, y, q; theta)

No hidden-z samples are used to fit F.  The latent q training signal is
reconstructed from resolved data only:
    G_rec = ydot - (rho*x-y)
    q_rec = -G_rec/x
with a small protected region around x=0 filled by interpolation.

For Lorenz-63, the exact hidden dynamics happen to satisfy
    zdot = xy - beta z,
so a quadratic library in (x,y,q) is capable of representing the exact
dynamic closure.  This fact is used only as a synthetic benchmark diagnostic;
the regression target is reconstructed from resolved data.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from math import comb
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from closure_inference import ClosureModelSpec


SYSTEM_NAME = "lorenz63_dynamic"
DISPLAY_NAME = "Partially observed Lorenz-63 with dynamic closure"


@dataclass(frozen=True)
class Lorenz63Config:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    dt: float = 1.0e-3
    t_start: float = 0.0
    t_end: float = 30.0
    seed: int = 10

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.t_start, self.t_end, self.dt, dtype=float)


def default_config() -> Lorenz63Config:
    return Lorenz63Config()


def default_model_specs() -> list[ClosureModelSpec]:
    """Candidate models for qdot=F(x,y,q).

    The quadratic model is the important physics-informed candidate because it
    can contain the terms q and xy.  Cubic is retained only as a flexibility
    check and should not be preferred unless held-out dynamics justify it.
    """
    return [
        ClosureModelSpec("xyq-linear", variant="xyq", degree=1),
        ClosureModelSpec("xyq-quadratic", variant="xyq", degree=2),
        ClosureModelSpec("xyq-cubic", variant="xyq", degree=3),
    ]


def state_labels(cfg: Lorenz63Config) -> list[str]:
    return [r"$x$", r"$y$"]


def closure_label(cfg: Lorenz63Config) -> str:
    return r"$G=-xq$"


def initial_conditions(cfg: Lorenz63Config) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    return np.array([1.0, 1.0, 1.0]) + 0.10 * rng.standard_normal(3)


def full_rhs(t: float, state: np.ndarray, cfg: Lorenz63Config) -> np.ndarray:
    x, y, z = state
    return np.array([
        cfg.sigma * (y - x),
        cfg.rho * x - y - x * z,
        x * y - cfg.beta * z,
    ])


def generate_full_truth(cfg: Lorenz63Config):
    t = cfg.time
    sol = solve_ivp(
        lambda time, state: full_rhs(time, state, cfg),
        (float(t[0]), float(t[-1])),
        initial_conditions(cfg),
        t_eval=t,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    if not sol.success or sol.y.shape[1] != t.size:
        raise RuntimeError(sol.message or "Lorenz-63 truth integration ended early.")
    full = sol.y.T
    if not np.all(np.isfinite(full)):
        raise RuntimeError("Non-finite Lorenz-63 truth.")
    return t, full


def generate_truth(cfg: Lorenz63Config):
    t, full = generate_full_truth(cfg)
    resolved = full[:, :2]
    hidden = full[:, [2]]
    closure = -(full[:, 0] * full[:, 2])[:, None]
    return t, resolved, hidden, closure


def add_observation_noise(x_true, noise_std: float, seed: int):
    rng = np.random.default_rng(seed)
    return np.asarray(x_true) + noise_std * rng.standard_normal(np.asarray(x_true).shape)


def resolved_tendency_without_closure(x, cfg: Lorenz63Config):
    x = np.asarray(x, dtype=float)
    return np.stack(
        [
            cfg.sigma * (x[..., 1] - x[..., 0]),
            cfg.rho * x[..., 0] - x[..., 1],
        ],
        axis=-1,
    )


def recover_closure_from_x(
    x_obs,
    cfg: Lorenz63Config,
    derivative: Literal["gradient", "savgol"] = "savgol",
    sg_window: int = 51,
    sg_polyorder: int = 3,
):
    """Recover G from resolved x,y only."""
    x_obs = np.asarray(x_obs, dtype=float)
    if x_obs.ndim != 2 or x_obs.shape[1] != 2:
        raise ValueError("x_obs must have shape (time,2).")

    if derivative == "gradient":
        dydt = np.gradient(x_obs[:, 1], cfg.dt, edge_order=2)
    elif derivative == "savgol":
        window = int(sg_window)
        if window % 2 == 0:
            window += 1
        max_odd = x_obs.shape[0] if x_obs.shape[0] % 2 else x_obs.shape[0] - 1
        window = min(window, max_odd)
        if window <= sg_polyorder:
            raise ValueError("Savitzky-Golay window must exceed polyorder.")
        dydt = savgol_filter(
            x_obs[:, 1], window_length=window, polyorder=sg_polyorder,
            deriv=1, delta=cfg.dt, mode="interp",
        )
    else:
        raise ValueError(f"Unknown derivative method {derivative!r}.")

    return (dydt - (cfg.rho * x_obs[:, 0] - x_obs[:, 1]))[:, None]


def reconstruct_latent_q(
    x_obs: np.ndarray,
    g_recovered: np.ndarray,
    *,
    min_abs_x: float = 0.10,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct q=-G/x without directly observing z.

    Points with |x|<min_abs_x are not divided by x.  They are filled by linear
    interpolation from neighboring well-conditioned points.  The returned mask
    marks the points where the ratio was evaluated directly.
    """
    x_obs = np.asarray(x_obs, dtype=float)
    g_recovered = np.asarray(g_recovered, dtype=float)
    if x_obs.shape[0] != g_recovered.shape[0]:
        raise ValueError("x_obs and g_recovered length mismatch.")

    x = x_obs[:, 0]
    g = g_recovered[:, 0]
    valid = np.abs(x) >= float(min_abs_x)
    if np.count_nonzero(valid) < 2:
        raise RuntimeError("Too few well-conditioned points to reconstruct q.")

    q = np.full(x.size, np.nan, dtype=float)
    q[valid] = -g[valid] / x[valid]

    idx = np.arange(x.size)
    q[~valid] = np.interp(idx[~valid], idx[valid], q[valid])

    if not np.all(np.isfinite(q)):
        raise RuntimeError("Non-finite reconstructed latent q.")

    return q[:, None], valid


def differentiate_latent_q(
    q: np.ndarray,
    cfg: Lorenz63Config,
    *,
    sg_window: int = 101,
    sg_polyorder: int = 3,
) -> np.ndarray:
    """Differentiate the reconstructed latent state smoothly."""
    q = np.asarray(q, dtype=float).reshape(-1)
    window = int(sg_window)
    if window % 2 == 0:
        window += 1
    max_odd = q.size if q.size % 2 else q.size - 1
    window = min(window, max_odd)
    if window <= sg_polyorder:
        raise ValueError("Latent Savitzky-Golay window must exceed polyorder.")
    qdot = savgol_filter(
        q, window_length=window, polyorder=sg_polyorder,
        deriv=1, delta=cfg.dt, mode="interp",
    )
    return qdot[:, None]


def dynamic_augmented_state(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    q = np.asarray(q, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2 or q.shape != (x.shape[0], 1):
        raise ValueError("Expected x:(n,2), q:(n,1).")
    return np.column_stack([x, q[:, 0]])


def closure_from_augmented_state(augmented: np.ndarray) -> np.ndarray:
    augmented = np.asarray(augmented, dtype=float)
    return -(augmented[..., 0] * augmented[..., 2])[..., None]


def resolved_tendency_with_closure(x, g, cfg: Lorenz63Config):
    base = resolved_tendency_without_closure(x, cfg)
    result = np.asarray(base, dtype=float).copy()
    result[..., 1] += np.asarray(g, dtype=float).reshape(result.shape[:-1])
    return result


def _selected_variables(values: np.ndarray, variant: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if variant == "x":
        return values[..., [0]]
    if variant == "y":
        return values[..., [1]]
    if variant == "xy":
        return values[..., :2]
    if variant == "xyq":
        if values.shape[-1] < 3:
            raise ValueError("xyq feature library requires augmented [x,y,q].")
        return values[..., :3]
    raise ValueError(f"Unsupported Lorenz-63 dynamic variant: {variant!r}")


def _features(values: np.ndarray, degree: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    terms = [1.0]
    for total_degree in range(1, int(degree) + 1):
        for combo in combinations_with_replacement(range(values.size), total_degree):
            term = 1.0
            for index in combo:
                term *= values[index]
            terms.append(term)
    return np.asarray(terms, dtype=float)


def build_design_matrix(x, spec):
    x = np.asarray(x, dtype=float)
    values = _selected_variables(x, spec.variant)
    if values.ndim == 1:
        values = values[None, :]
    return np.vstack([_features(row, spec.degree) for row in values])


def closure_feature_matrix(x, spec):
    values = _selected_variables(np.asarray(x, dtype=float), spec.variant)
    return _features(values, spec.degree)[None, :]


def closure_output_shape(x):
    return (np.asarray(x).shape[0], 1)


def instantaneous_closure_shape(x):
    return (1,)


def feature_names(spec: ClosureModelSpec) -> list[str]:
    variables = {
        "x": ("x",), "y": ("y",), "xy": ("x", "y"),
        "xyq": ("x", "y", "q"),
    }[spec.variant]
    names = ["1"]
    for degree in range(1, int(spec.degree) + 1):
        for combo in combinations_with_replacement(range(len(variables)), degree):
            powers = [combo.count(i) for i in range(len(variables))]
            pieces = []
            for name, power in zip(variables, powers):
                if power == 1:
                    pieces.append(name)
                elif power > 1:
                    pieces.append(f"{name}^{power}")
            names.append("".join(pieces))
    return names


def physical_coefficients(beta_scaled, scaler):
    beta_scaled = np.asarray(beta_scaled, dtype=float)
    mean = np.asarray(scaler.mean, dtype=float)
    scale = np.asarray(scaler.scale, dtype=float)
    coefficients = beta_scaled / scale
    coefficients = coefficients.copy()
    coefficients[0] -= np.sum(beta_scaled[1:] * mean[1:] / scale[1:])
    return coefficients


def check_dynamic_feature_library(
    augmented_sample: np.ndarray,
    specs: list[ClosureModelSpec] | None = None,
    *,
    verbose: bool = True,
):
    augmented_sample = np.asarray(augmented_sample, dtype=float)
    if augmented_sample.ndim != 2 or augmented_sample.shape[1] != 3:
        raise AssertionError(
            f"Expected augmented sample shape (n,3); got {augmented_sample.shape}."
        )
    specs = list(default_model_specs() if specs is None else specs)
    report = {}

    if verbose:
        print("\n" + "=" * 86)
        print("LORENZ-63 DYNAMIC-CLOSURE FEATURE CHECKS")
        print("=" * 86)
        print(
            f"{'model':<20}{'nfeat':>8}{'rank':>8}"
            f"{'batch-online err':>20}{'status':>12}"
        )
        print("-" * 86)

    for spec in specs:
        phi = build_design_matrix(augmented_sample, spec)
        expected = comb(3 + int(spec.degree), int(spec.degree))
        names = feature_names(spec)
        if phi.shape != (augmented_sample.shape[0], expected):
            raise AssertionError(f"{spec.name}: wrong design shape {phi.shape}.")
        if len(names) != expected or len(set(names)) != expected:
            raise AssertionError(f"{spec.name}: feature-name mismatch.")

        inds = np.unique(np.linspace(
            0, augmented_sample.shape[0]-1,
            min(16, augmented_sample.shape[0]), dtype=int
        ))
        err = 0.0
        for i in inds:
            b = build_design_matrix(augmented_sample[i:i+1], spec)
            o = closure_feature_matrix(augmented_sample[i], spec)
            err = max(err, float(np.max(np.abs(b-o))))
        if err > 1e-12:
            raise AssertionError(f"{spec.name}: batch/online feature mismatch {err:.3e}.")

        rank = int(np.linalg.matrix_rank(phi))
        if rank < expected:
            raise AssertionError(f"{spec.name}: rank deficient {rank}/{expected}.")

        report[spec.name] = {
            "n_features": expected, "rank": rank,
            "batch_online_max_error": err, "feature_names": names,
        }
        if verbose:
            print(f"{spec.name:<20}{expected:>8d}{rank:>8d}{err:>20.3e}{'PASS':>12}")

    if verbose:
        print("-" * 86)
        print("Dynamic-closure feature checks: PASS")
        print("=" * 86)
    return report


def run_preanalysis_checks(
    cfg: Lorenz63Config | None = None,
    *,
    derivative: Literal["gradient", "savgol"] = "savgol",
    sg_window: int = 51,
    sg_polyorder: int = 3,
    latent_min_abs_x: float = 0.10,
    latent_sg_window: int = 101,
    latent_sg_polyorder: int = 3,
):
    """Check closure recovery and latent-state reconstruction before inference."""
    cfg = cfg or default_config()
    t, full = generate_full_truth(cfg)
    resolved = full[:, :2]
    z = full[:, [2]]
    g_exact = -(full[:, 0] * full[:, 2])[:, None]
    g_rec = recover_closure_from_x(
        resolved, cfg, derivative=derivative,
        sg_window=sg_window, sg_polyorder=sg_polyorder,
    )
    q_rec, valid = reconstruct_latent_q(
        resolved, g_rec, min_abs_x=latent_min_abs_x
    )
    qdot_rec = differentiate_latent_q(
        q_rec, cfg, sg_window=latent_sg_window,
        sg_polyorder=latent_sg_polyorder,
    )

    z_dot_exact = (full[:, 0]*full[:, 1] - cfg.beta*full[:, 2])[:, None]

    trim = max(sg_window//2, latent_sg_window//2)
    sl = slice(trim, -trim)

    def rmse(a,b):
        return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
    def nrmse(a,b):
        return rmse(a,b)/max(float(np.std(a)), np.finfo(float).eps)

    metrics = {
        "closure_recovery_nrmse": nrmse(g_exact[sl], g_rec[sl]),
        "latent_q_vs_hidden_z_nrmse": nrmse(z[sl], q_rec[sl]),
        "latent_qdot_vs_hidden_zdot_nrmse": nrmse(z_dot_exact[sl], qdot_rec[sl]),
        "protected_fraction": float(np.mean(~valid)),
    }

    print("\n" + "="*78)
    print("LORENZ-63 DYNAMIC-CLOSURE PRECHECKS")
    print("="*78)
    print(f"Samples:                              {t.size}")
    print(f"dt:                                   {cfg.dt:.6e}")
    print(f"Recovered G NRMSE:                    {metrics['closure_recovery_nrmse']:.6e}")
    print(f"Reconstructed q vs hidden z NRMSE:    {metrics['latent_q_vs_hidden_z_nrmse']:.6e}")
    print(f"Reconstructed qdot vs zdot NRMSE:     {metrics['latent_qdot_vs_hidden_zdot_nrmse']:.6e}")
    print(f"|x|<{latent_min_abs_x:g} interpolated fraction:       {metrics['protected_fraction']:.6f}")
    print("Hidden z used for fitting:             NO")
    print("Dynamic-closure prechecks:             PASS")
    print("="*78)
    return metrics


if __name__ == "__main__":
    run_preanalysis_checks()
