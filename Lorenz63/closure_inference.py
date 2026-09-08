"""System-independent Bayesian closure inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class ClosureModelSpec:
    """Generic model descriptor.

    `variant` is interpreted by the selected physical-system module.
    Examples:
      linear system: "x"
      Lorenz-63:     "x", "xy"
      Lorenz-96:     "local", "r1", "r2"
    """
    name: str
    variant: str
    degree: int

    def __post_init__(self) -> None:
        if self.degree not in (1, 2, 3):
            raise ValueError(
                "Only polynomial degrees 1, 2, and 3 are currently supported."
            )
    


@dataclass
class FeatureScaler:
    mean: np.ndarray
    scale: np.ndarray

    def transform(self, phi: np.ndarray) -> np.ndarray:
        phi = np.asarray(phi, dtype=float)
        return (phi - self.mean) / self.scale


def fit_feature_scaler(phi: np.ndarray) -> FeatureScaler:
    phi = np.asarray(phi, dtype=float)
    if phi.ndim != 2:
        raise ValueError("phi must be a 2-D design matrix.")
    mean = phi.mean(axis=0)
    scale = phi.std(axis=0)

    # The first feature is always the intercept.
    mean[0] = 0.0
    scale[0] = 1.0
    # scale[scale < 1.0e-12] = 1.0
    scale[scale < 1.0e-8] = 1.0
    return FeatureScaler(mean=mean, scale=scale)


def build_design_matrix(
    x: np.ndarray,
    spec: ClosureModelSpec,
    problem: Any,
) -> np.ndarray:
    """Dispatch feature construction to the selected system."""
    phi = np.asarray(problem.build_design_matrix(x, spec), dtype=float)
    if phi.ndim != 2:
        raise ValueError("Problem build_design_matrix must return a 2-D array.")
    return phi


def least_squares_initialization(
    phi_scaled: np.ndarray,
    g_target: np.ndarray,
) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(
        np.asarray(phi_scaled, dtype=float),
        np.asarray(g_target, dtype=float).reshape(-1),
        rcond=None,
    )
    return beta


def run_mcmc_regression(
    phi_scaled: np.ndarray,
    g_target: np.ndarray,
    *,
    warmup_steps: int = 300,
    num_samples: int = 500,
    prior_scale: float | None = None,
    seed: int = 10,
    max_tree_depth: int = 4,
    target_accept_prob: float = 0.85,
) -> dict[str, np.ndarray]:
    """Bayesian polynomial regression for direct closure recovery."""
    import torch
    import pyro
    import pyro.distributions as dist
    from pyro.infer import MCMC, NUTS

    pyro.clear_param_store()
    pyro.set_rng_seed(seed)

    phi_t = torch.as_tensor(phi_scaled, dtype=torch.float64)
    y_t = torch.as_tensor(np.asarray(g_target).reshape(-1), dtype=torch.float64)

    if phi_t.shape[0] != y_t.shape[0]:
        raise ValueError(
            f"Design-matrix rows ({phi_t.shape[0]}) and target size "
            f"({y_t.shape[0]}) do not match."
        )

    y_scale = float(np.std(g_target))
    if prior_scale is None:
        prior_scale = max(2.0 * y_scale, 1.0)
    sigma_prior_scale = max(y_scale, 1.0e-3)
    p = phi_t.shape[1]

    def bayes_model(phi, y):
        beta = pyro.sample(
            "beta",
            dist.Normal(
                torch.zeros(p, dtype=torch.float64),
                prior_scale * torch.ones(p, dtype=torch.float64),
            ).to_event(1),
        )
        # sigma = pyro.sample(
        #     "sigma",
        #     dist.HalfNormal(
        #         torch.tensor(sigma_prior_scale, dtype=torch.float64)
        #     ),
        # )
        
        sigma_raw = pyro.sample(
            "sigma_raw",
            dist.HalfNormal(
                torch.tensor(sigma_prior_scale, dtype=torch.float64)
            ),
        )

        sigma_floor = torch.tensor(
            max(1.0e-4, 1.0e-3 * y_scale),
            dtype=torch.float64,
        )

        sigma = sigma_floor + sigma_raw
    
        mu = phi @ beta
        with pyro.plate("observations", y.shape[0]):
            pyro.sample("obs", dist.Normal(mu, sigma), obs=y)

    kernel = NUTS(
        bayes_model,
        max_tree_depth=max_tree_depth,
        target_accept_prob=target_accept_prob,
    )
    mcmc = MCMC(
        kernel,
        warmup_steps=warmup_steps,
        num_samples=num_samples,
        disable_progbar=False,
    )
    mcmc.run(phi_t, y_t)
    samples = mcmc.get_samples()

    return {
        "beta": samples["beta"].detach().cpu().numpy(),
        # "sigma": samples["sigma"].detach().cpu().numpy(),
        "sigma": (
            samples["sigma_raw"].detach().cpu().numpy()
            + max(1.0e-4, 1.0e-3 * y_scale)
        ),
    
    }


def posterior_closure_mean(
    x: np.ndarray,
    spec: ClosureModelSpec,
    scaler: FeatureScaler,
    beta_samples: np.ndarray,
    problem: Any,
) -> np.ndarray:
    phi = scaler.transform(build_design_matrix(x, spec, problem))
    beta_mean = np.asarray(beta_samples).mean(axis=0)
    pred = phi @ beta_mean
    return pred.reshape(problem.closure_output_shape(x))


def posterior_closure_interval(
    x: np.ndarray,
    spec: ClosureModelSpec,
    scaler: FeatureScaler,
    beta_samples: np.ndarray,
    problem: Any,
    *,
    lower: float = 2.5,
    upper: float = 97.5,
    max_draws: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    phi = scaler.transform(build_design_matrix(x, spec, problem))
    draws = np.asarray(beta_samples)
    if draws.shape[0] > max_draws:
        ids = np.linspace(0, draws.shape[0] - 1, max_draws, dtype=int)
        draws = draws[ids]
    pred = phi @ draws.T
    shape = problem.closure_output_shape(x)
    lo = np.percentile(pred, lower, axis=1).reshape(shape)
    hi = np.percentile(pred, upper, axis=1).reshape(shape)
    return lo, hi


def learned_closure_vector(
    x: np.ndarray,
    beta: np.ndarray,
    spec: ClosureModelSpec,
    scaler: FeatureScaler,
    problem: Any,
) -> np.ndarray:
    """Evaluate learned closure from the current resolved state."""
    phi = np.asarray(problem.closure_feature_matrix(x, spec), dtype=float)
    phi_scaled = scaler.transform(phi)
    return np.asarray(phi_scaled @ beta, dtype=float).reshape(
        problem.instantaneous_closure_shape(x)
    )


def integrate_learned_closure(
    t: np.ndarray,
    x0: np.ndarray,
    beta: np.ndarray,
    spec: ClosureModelSpec,
    scaler: FeatureScaler,
    cfg: Any,
    problem: Any,
    *,
    rtol: float = 1.0e-7,
    atol: float = 1.0e-9,
) -> np.ndarray:
    """A-posteriori integration with no access to hidden/reference variables."""

    # def rhs(time: float, x: np.ndarray) -> np.ndarray:
    #     g = learned_closure_vector(x, beta, spec, scaler, problem)
    #     return problem.resolved_tendency_with_closure(x, g, cfg)
    if problem.SYSTEM_NAME == "lorenz63":

        def rhs(time: float, state: np.ndarray) -> np.ndarray:
            x, y, g = state

            dgdt = learned_closure_vector(
                state,
                beta,
                spec,
                scaler,
                problem,
            )[0]

            return np.array([
                cfg.sigma * (y - x),
                cfg.rho * x - y + g,
                dgdt,
            ])

    else:

        def rhs(time: float, state: np.ndarray) -> np.ndarray:
            g = learned_closure_vector(
                state,
                beta,
                spec,
                scaler,
                problem,
            )

            return problem.resolved_tendency_with_closure(
                state,
                g,
                cfg,
            )    
    # Stop integration if the learned model becomes unstable
    blowup_limit = 1.0e3

    def blowup_event(time, x):
        return blowup_limit - np.linalg.norm(x)

    blowup_event.terminal = True
    blowup_event.direction = -1
    
    if problem.SYSTEM_NAME == "lorenz63":
        x0 = np.asarray(x0, dtype=float)

        if x0.size == 2:
            raise ValueError(
                "Lorenz-63 dynamic closure requires "
                "initial state [x0, y0, G0]."
            )

    sol = solve_ivp(
        rhs,
        (float(t[0]), float(t[-1])),
        np.asarray(x0, dtype=float),
        t_eval=np.asarray(t, dtype=float),
        method="RK45",
        rtol=rtol,
        atol=atol,
        events=blowup_event,
    )
    
    if sol.status == 1:
        t_fail = sol.t_events[0][0]

        raise RuntimeError(
            f"Learned closure became unstable at t={t_fail:.4f}; "
            f"state norm exceeded {blowup_limit:g}."
        )
    
    
    if not sol.success:
        raise RuntimeError(sol.message)
        
    trajectory = sol.y.T
    
    if problem.SYSTEM_NAME == "lorenz63":
        return trajectory[:, :2]
    
    
    return trajectory


def posterior_trajectory_ensemble(
    t: np.ndarray,
    x0: np.ndarray,
    beta_samples: np.ndarray,
    spec: ClosureModelSpec,
    scaler: FeatureScaler,
    cfg: Any,
    problem: Any,
    *,
    n_draws: int = 20,
) -> np.ndarray:
    draws = np.asarray(beta_samples)
    n_draws = min(n_draws, draws.shape[0])
    ids = np.linspace(0, draws.shape[0] - 1, n_draws, dtype=int)
    trajectories = [
        integrate_learned_closure(
            t, x0, draws[i], spec, scaler, cfg, problem
        )
        for i in ids
    ]
    return np.stack(trajectories, axis=0)


def state_metrics(
    x_ref: np.ndarray,
    x_pred: np.ndarray,
) -> dict[str, float]:
    if x_ref.shape != x_pred.shape:
        raise ValueError("x_ref and x_pred must have the same shape.")
    err = x_pred - x_ref
    return {
        "trajectory_rmse": float(np.sqrt(np.mean(err**2))),
        "mean_stat_error": float(
            np.mean(
                np.abs(
                    np.mean(x_pred, axis=0)
                    - np.mean(x_ref, axis=0)
                )
            )
        ),
        "std_stat_error": float(
            np.mean(
                np.abs(
                    np.std(x_pred, axis=0)
                    - np.std(x_ref, axis=0)
                )
            )
        ),
    }
