"""Ensemble Kalman filtering for closure-augmented digital twins.

The notation follows the digital-twin draft:

    pi_k(x_k, xi) = p(x_k, xi | D_k),
    x_{k+1} = F_k(x_k, xi, u_k) + w_k,
    y_{k+1} = H_{k+1}(x_{k+1}, xi) + v_{k+1}.

For the two-level Lorenz-96 application, ``x_k`` is the resolved state ``X``
and ``xi`` is the learned closure-coefficient vector ``beta``.  The posterior
``pi_k`` is represented by an augmented ensemble ``{(X, beta)^(n)}``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations_with_replacement
from typing import Any, Callable

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class EnKFConfig:
    """Sequential digital-twin/EnKF configuration."""

    ensemble_size: int = 40
    assimilation_stride: int = 100
    observed_state_stride: int = 2
    observation_noise_std: float = 0.50
    initial_state_std: float = 0.75
    process_noise_std: float = 0.02
    parameter_process_noise_std: float = 1.0e-4
    state_inflation: float = 1.02
    parameter_inflation: float = 1.005
    localization_radius: float | None = 4.0
    estimate_closure_parameters: bool = True
    seed: int = 2026

    def __post_init__(self) -> None:
        if self.ensemble_size < 3:
            raise ValueError("ensemble_size must be at least 3.")
        if self.assimilation_stride < 1 or self.observed_state_stride < 1:
            raise ValueError("Assimilation and observation strides must be positive.")
        if self.observation_noise_std <= 0.0:
            raise ValueError("observation_noise_std must be positive.")
        if self.initial_state_std < 0.0 or self.process_noise_std < 0.0:
            raise ValueError("State uncertainty scales must be nonnegative.")
        if self.parameter_process_noise_std < 0.0:
            raise ValueError("parameter_process_noise_std must be nonnegative.")
        if self.state_inflation < 1.0 or self.parameter_inflation < 1.0:
            raise ValueError("Inflation factors must be at least one.")


def default_qoi(x: Array) -> Array:
    """System QoI: resolved-state energy Q(X)=K^{-1} sum_k X_k^2/2."""
    x = np.asarray(x, dtype=float)
    return 0.5 * np.mean(x**2, axis=-1)


def gaspari_cohn(distance: Array, radius: float) -> Array:
    """Compactly supported Gaspari-Cohn correlation taper.

    ``radius`` is the half-width at which the first polynomial branch ends;
    correlations vanish at distance ``2*radius``.
    """
    if radius <= 0.0:
        raise ValueError("localization radius must be positive.")
    r = np.abs(np.asarray(distance, dtype=float)) / radius
    out = np.zeros_like(r)
    inner = r <= 1.0
    outer = (r > 1.0) & (r <= 2.0)
    ri = r[inner]
    ro = r[outer]
    out[inner] = 1.0 - 5.0 / 3.0 * ri**2 + 5.0 / 8.0 * ri**3 + 0.5 * ri**4 - 0.25 * ri**5
    out[outer] = (
        4.0 - 5.0 * ro + 5.0 / 3.0 * ro**2 + 5.0 / 8.0 * ro**3
        - 0.5 * ro**4 + 1.0 / 12.0 * ro**5 - 2.0 / (3.0 * ro)
    )
    return np.clip(out, 0.0, 1.0)


def periodic_localization_matrix(
    n_state: int,
    observation_indices: Array,
    radius: float,
) -> Array:
    """State-observation taper for the cyclic Lorenz-96 lattice."""
    state = np.arange(n_state)[:, None]
    obs = np.asarray(observation_indices, dtype=int)[None, :]
    direct = np.abs(state - obs)
    distance = np.minimum(direct, n_state - direct)
    return gaspari_cohn(distance, radius)


def evaluate_closure(
    x: Array,
    beta: Array,
    spec: Any,
    scaler: Any,
    problem: Any,
) -> Array:
    """Evaluate the inferred instantaneous closure G-hat(X; beta)."""
    phi = np.asarray(problem.closure_feature_matrix(x, spec), dtype=float)
    phi_scaled = scaler.transform(phi)
    return np.asarray(phi_scaled @ beta, dtype=float).reshape(
        problem.instantaneous_closure_shape(x)
    )


def evaluate_closure_ensemble(
    x_ensemble: Array,
    beta_ensemble: Array,
    spec: Any,
    scaler: Any,
    problem: Any,
) -> Array:
    """Vectorized closure evaluation, with a Lorenz-96 fast path."""
    x_ensemble = np.asarray(x_ensemble, dtype=float)
    beta_ensemble = np.asarray(beta_ensemble, dtype=float)
    if getattr(problem, "SYSTEM_NAME", None) != "lorenz96":
        return np.stack(
            [
                evaluate_closure(x, beta, spec, scaler, problem)
                for x, beta in zip(x_ensemble, beta_ensemble)
            ]
        )

    radius = {"local": 0, "r1": 1, "r2": 2}.get(spec.variant)
    if radius is None:
        raise ValueError(f"Unknown Lorenz-96 closure variant: {spec.variant}")
    values = np.stack(
        [np.roll(x_ensemble, -offset, axis=1) for offset in range(-radius, radius + 1)],
        axis=-1,
    )
    features = [np.ones(values.shape[:2], dtype=float)]
    features.extend(values[..., j] for j in range(values.shape[-1]))
    if spec.degree == 2:
        features.extend(
            values[..., i] * values[..., j]
            for i, j in combinations_with_replacement(range(values.shape[-1]), 2)
        )
    phi = np.stack(features, axis=-1)
    phi_scaled = (phi - np.asarray(scaler.mean)) / np.asarray(scaler.scale)
    return np.einsum("nkp,np->nk", phi_scaled, beta_ensemble)


def resolved_rhs(
    x: Array,
    beta: Array,
    spec: Any,
    scaler: Any,
    physical: Any,
    problem: Any,
) -> Array:
    g_hat = evaluate_closure(x, beta, spec, scaler, problem)
    return np.asarray(
        problem.resolved_tendency_with_closure(x, g_hat, physical),
        dtype=float,
    )


def rk4_member(
    x: Array,
    beta: Array,
    dt: float,
    spec: Any,
    scaler: Any,
    physical: Any,
    problem: Any,
) -> Array:
    """Advance one closure-augmented resolved member by one model step."""
    args = (beta, spec, scaler, physical, problem)
    k1 = resolved_rhs(x, *args)
    k2 = resolved_rhs(x + 0.5 * dt * k1, *args)
    k3 = resolved_rhs(x + 0.5 * dt * k2, *args)
    k4 = resolved_rhs(x + dt * k3, *args)
    return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def resolved_rhs_ensemble(
    x_ensemble: Array,
    beta_ensemble: Array,
    spec: Any,
    scaler: Any,
    physical: Any,
    problem: Any,
) -> Array:
    if getattr(problem, "SYSTEM_NAME", None) != "lorenz96":
        return np.stack(
            [
                resolved_rhs(x, beta, spec, scaler, physical, problem)
                for x, beta in zip(x_ensemble, beta_ensemble)
            ]
        )
    g_hat = evaluate_closure_ensemble(
        x_ensemble, beta_ensemble, spec, scaler, problem
    )
    return np.asarray(
        problem.resolved_tendency_with_closure(x_ensemble, g_hat, physical),
        dtype=float,
    )


def rk4_ensemble(
    x: Array,
    beta: Array,
    dt: float,
    spec: Any,
    scaler: Any,
    physical: Any,
    problem: Any,
) -> Array:
    """Advance all members simultaneously when the problem is vectorized."""
    args = (beta, spec, scaler, physical, problem)
    k1 = resolved_rhs_ensemble(x, *args)
    k2 = resolved_rhs_ensemble(x + 0.5 * dt * k1, *args)
    k3 = resolved_rhs_ensemble(x + 0.5 * dt * k2, *args)
    k4 = resolved_rhs_ensemble(x + dt * k3, *args)
    return x + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def forecast_ensemble(
    x_ensemble: Array,
    beta_ensemble: Array,
    *,
    n_steps: int,
    dt: float,
    spec: Any,
    scaler: Any,
    physical: Any,
    problem: Any,
) -> Array:
    """Apply the digital-twin forecast map F_k to every ensemble member."""
    forecast = np.asarray(x_ensemble, dtype=float).copy()
    for _ in range(int(n_steps)):
        forecast = rk4_ensemble(
            forecast, beta_ensemble, dt, spec, scaler, physical, problem
        )
    if not np.all(np.isfinite(forecast)):
        raise RuntimeError("Non-finite state encountered during EnKF forecast.")
    return forecast


def inflate_ensemble(ensemble: Array, factor: float) -> Array:
    mean = np.mean(ensemble, axis=0, keepdims=True)
    return mean + factor * (ensemble - mean)


def stochastic_enkf_update(
    x_forecast: Array,
    beta_forecast: Array,
    observation: Array,
    observation_indices: Array,
    observation_noise_std: float,
    rng: np.random.Generator,
    *,
    localization_radius: float | None,
    estimate_closure_parameters: bool,
) -> tuple[Array, Array, Array]:
    """Joint state-parameter EnKF analysis for one assimilation time."""
    x_forecast = np.asarray(x_forecast, dtype=float)
    beta_forecast = np.asarray(beta_forecast, dtype=float)
    observation_indices = np.asarray(observation_indices, dtype=int)
    n_ensemble, n_state = x_forecast.shape

    augmented = np.concatenate([x_forecast, beta_forecast], axis=1)
    predicted_observation = x_forecast[:, observation_indices]
    a_z = augmented - augmented.mean(axis=0, keepdims=True)
    a_y = predicted_observation - predicted_observation.mean(axis=0, keepdims=True)
    p_zy = a_z.T @ a_y / (n_ensemble - 1)
    p_yy = a_y.T @ a_y / (n_ensemble - 1)

    if localization_radius is not None:
        taper_x = periodic_localization_matrix(
            n_state, observation_indices, localization_radius
        )
        p_zy[:n_state] *= taper_x
        # beta is a global, translation-invariant closure parameter vector;
        # its cross-covariance is intentionally not spatially localized.

    if not estimate_closure_parameters:
        p_zy[n_state:] = 0.0

    r_matrix = observation_noise_std**2 * np.eye(observation_indices.size)
    innovation_covariance = p_yy + r_matrix
    kalman_gain = np.linalg.solve(innovation_covariance, p_zy.T).T

    perturbed_observations = observation + rng.normal(
        0.0, observation_noise_std,
        size=(n_ensemble, observation_indices.size),
    )
    innovations = perturbed_observations - predicted_observation
    analysis = augmented + innovations @ kalman_gain.T
    return analysis[:, :n_state], analysis[:, n_state:], kalman_gain


def _ensemble_summary(ensemble: Array) -> tuple[Array, Array, Array, Array]:
    return (
        np.mean(ensemble, axis=0),
        np.std(ensemble, axis=0, ddof=1),
        np.percentile(ensemble, 2.5, axis=0),
        np.percentile(ensemble, 97.5, axis=0),
    )


def run_sequential_digital_twin(
    *,
    t: Array,
    x_truth: Array,
    x0_mean: Array,
    beta_samples: Array,
    spec: Any,
    scaler: Any,
    physical: Any,
    problem: Any,
    config: EnKFConfig | None = None,
    observation_indices: Array | None = None,
    qoi: Callable[[Array], Array] = default_qoi,
) -> dict[str, Any]:
    """Run forecast-analysis cycles on a synthetic physical asset.

    The hidden fast variables and true closure are never supplied to the
    filter.  ``x_truth`` is used only to generate measurements and evaluate
    the digital twin.
    """
    cfg = config or EnKFConfig()
    t = np.asarray(t, dtype=float)
    x_truth = np.asarray(x_truth, dtype=float)
    x0_mean = np.asarray(x0_mean, dtype=float)
    beta_samples = np.asarray(beta_samples, dtype=float)
    if x_truth.shape != (t.size, x0_mean.size):
        raise ValueError("x_truth must have shape (len(t), len(x0_mean)).")
    if beta_samples.ndim != 2:
        raise ValueError("beta_samples must have shape (n_draws, n_parameters).")
    if t.size < 2 or not np.all(np.diff(t) > 0.0):
        raise ValueError("t must be strictly increasing with at least two entries.")

    n_state = x0_mean.size
    if observation_indices is None:
        observation_indices = np.arange(
            0, n_state, cfg.observed_state_stride, dtype=int
        )
    observation_indices = np.asarray(observation_indices, dtype=int)
    if observation_indices.size == 0 or np.any(observation_indices < 0) or np.any(observation_indices >= n_state):
        raise ValueError("observation_indices contains no valid observed states.")
    observation_indices = np.unique(observation_indices)

    assimilation_indices = np.arange(
        0, t.size, cfg.assimilation_stride, dtype=int
    )
    if assimilation_indices[-1] != t.size - 1:
        assimilation_indices = np.append(assimilation_indices, t.size - 1)
    dt = float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t), dt, rtol=1.0e-7, atol=1.0e-12):
        raise ValueError("The current RK4 forecast requires a uniform time grid.")

    rng = np.random.default_rng(cfg.seed)
    draw_ids = rng.choice(
        beta_samples.shape[0], size=cfg.ensemble_size,
        replace=beta_samples.shape[0] < cfg.ensemble_size,
    )
    beta_ensemble = beta_samples[draw_ids].copy()
    x_ensemble = x0_mean + rng.normal(
        0.0, cfg.initial_state_std,
        size=(cfg.ensemble_size, n_state),
    )

    truth_at_obs = x_truth[assimilation_indices]
    observations = truth_at_obs[:, observation_indices] + rng.normal(
        0.0, cfg.observation_noise_std,
        size=(assimilation_indices.size, observation_indices.size),
    )

    shape = (assimilation_indices.size, n_state)
    closure_shape = (
        assimilation_indices.size,
        *problem.instantaneous_closure_shape(x0_mean),
    )
    parameter_shape = (assimilation_indices.size, beta_ensemble.shape[1])
    saved = {
        "forecast_mean": np.empty(shape), "forecast_std": np.empty(shape),
        "forecast_lower": np.empty(shape), "forecast_upper": np.empty(shape),
        "analysis_mean": np.empty(shape), "analysis_std": np.empty(shape),
        "analysis_lower": np.empty(shape), "analysis_upper": np.empty(shape),
        "beta_mean": np.empty(parameter_shape), "beta_std": np.empty(parameter_shape),
        "closure_forecast_mean": np.empty(closure_shape), "closure_forecast_std": np.empty(closure_shape),
        "closure_analysis_mean": np.empty(closure_shape), "closure_analysis_std": np.empty(closure_shape),
        "qoi_forecast_mean": np.empty(assimilation_indices.size),
        "qoi_forecast_std": np.empty(assimilation_indices.size),
        "qoi_forecast_lower": np.empty(assimilation_indices.size),
        "qoi_forecast_upper": np.empty(assimilation_indices.size),
        "qoi_analysis_mean": np.empty(assimilation_indices.size),
        "qoi_analysis_std": np.empty(assimilation_indices.size),
        "qoi_analysis_lower": np.empty(assimilation_indices.size),
        "qoi_analysis_upper": np.empty(assimilation_indices.size),
    }
    gain_norm = np.empty(assimilation_indices.size)

    previous_index = assimilation_indices[0]
    for j, current_index in enumerate(assimilation_indices):
        n_steps = int(current_index - previous_index)
        if n_steps:
            x_ensemble = forecast_ensemble(
                x_ensemble, beta_ensemble, n_steps=n_steps, dt=dt,
                spec=spec, scaler=scaler, physical=physical, problem=problem,
            )
            if cfg.process_noise_std:
                x_ensemble += rng.normal(
                    0.0, cfg.process_noise_std, size=x_ensemble.shape
                )
            if cfg.estimate_closure_parameters and cfg.parameter_process_noise_std:
                beta_ensemble += rng.normal(
                    0.0, cfg.parameter_process_noise_std,
                    size=beta_ensemble.shape,
                )
        x_ensemble = inflate_ensemble(x_ensemble, cfg.state_inflation)
        if cfg.estimate_closure_parameters:
            beta_ensemble = inflate_ensemble(
                beta_ensemble, cfg.parameter_inflation
            )

        forecast_summary = _ensemble_summary(x_ensemble)
        for key, value in zip(
            ("forecast_mean", "forecast_std", "forecast_lower", "forecast_upper"),
            forecast_summary,
        ):
            saved[key][j] = value
        q_forecast = np.asarray(qoi(x_ensemble), dtype=float)
        saved["qoi_forecast_mean"][j] = np.mean(q_forecast)
        saved["qoi_forecast_std"][j] = np.std(q_forecast, ddof=1)
        saved["qoi_forecast_lower"][j], saved["qoi_forecast_upper"][j] = np.percentile(q_forecast, [2.5, 97.5])
        g_forecast = evaluate_closure_ensemble(
            x_ensemble, beta_ensemble, spec, scaler, problem
        )
        saved["closure_forecast_mean"][j] = np.mean(g_forecast, axis=0)
        saved["closure_forecast_std"][j] = np.std(g_forecast, axis=0, ddof=1)

        x_ensemble, beta_ensemble, gain = stochastic_enkf_update(
            x_ensemble, beta_ensemble, observations[j], observation_indices,
            cfg.observation_noise_std, rng,
            localization_radius=cfg.localization_radius,
            estimate_closure_parameters=cfg.estimate_closure_parameters,
        )
        gain_norm[j] = np.linalg.norm(gain, ord="fro")
        analysis_summary = _ensemble_summary(x_ensemble)
        for key, value in zip(
            ("analysis_mean", "analysis_std", "analysis_lower", "analysis_upper"),
            analysis_summary,
        ):
            saved[key][j] = value
        saved["beta_mean"][j] = np.mean(beta_ensemble, axis=0)
        saved["beta_std"][j] = np.std(beta_ensemble, axis=0, ddof=1)
        q_analysis = np.asarray(qoi(x_ensemble), dtype=float)
        saved["qoi_analysis_mean"][j] = np.mean(q_analysis)
        saved["qoi_analysis_std"][j] = np.std(q_analysis, ddof=1)
        saved["qoi_analysis_lower"][j], saved["qoi_analysis_upper"][j] = np.percentile(q_analysis, [2.5, 97.5])
        g_analysis = evaluate_closure_ensemble(
            x_ensemble, beta_ensemble, spec, scaler, problem
        )
        saved["closure_analysis_mean"][j] = np.mean(g_analysis, axis=0)
        saved["closure_analysis_std"][j] = np.std(g_analysis, axis=0, ddof=1)
        previous_index = current_index

    forecast_error = saved["forecast_mean"] - truth_at_obs
    analysis_error = saved["analysis_mean"] - truth_at_obs
    saved.update(
        {
            "config": asdict(cfg),
            "assimilation_indices": assimilation_indices,
            "assimilation_times": t[assimilation_indices],
            "observation_indices": observation_indices,
            "observations": observations,
            "truth_at_assimilation": truth_at_obs,
            "forecast_rmse": np.sqrt(np.mean(forecast_error**2, axis=1)),
            "analysis_rmse": np.sqrt(np.mean(analysis_error**2, axis=1)),
            "forecast_spread": np.sqrt(np.mean(saved["forecast_std"]**2, axis=1)),
            "analysis_spread": np.sqrt(np.mean(saved["analysis_std"]**2, axis=1)),
            "qoi_truth": np.asarray(qoi(truth_at_obs), dtype=float),
            "kalman_gain_frobenius": gain_norm,
            "forecast_coverage": np.mean(
                (truth_at_obs >= saved["forecast_lower"])
                & (truth_at_obs <= saved["forecast_upper"]), axis=1,
            ),
            "analysis_coverage": np.mean(
                (truth_at_obs >= saved["analysis_lower"])
                & (truth_at_obs <= saved["analysis_upper"]), axis=1,
            ),
        }
    )
    return saved
