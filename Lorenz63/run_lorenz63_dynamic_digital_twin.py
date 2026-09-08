"""Run an EnKF digital twin for the Lorenz-63 dynamic closure.

Dynamic closure model
---------------------
    xdot = sigma (y-x)
    ydot = rho x - y - x q
    qdot = F(x,y,q; beta)

The EnKF state is augmented as
    [x, y, q, beta_1, ..., beta_p].

Only x and/or y are observed.  The latent q and closure coefficients are
updated through ensemble cross-covariances; hidden z and exact G=-xz are used
only for synthetic verification after the analysis.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

import lorenz63_dynamic_system as problem


ObservationMode = Literal["x", "y", "xy"]


@dataclass(frozen=True)
class DynamicEnKFConfig:
    ensemble_size: int = 150
    assimilation_stride: int = 5
    observation_noise_std: float = 0.50

    initial_x_std: float = 1.0
    initial_y_std: float = 1.0
    initial_q_std: float = 0.50

    process_noise_x: float = 0.10
    process_noise_y: float = 0.10
    process_noise_q: float = 0.05
    parameter_process_noise_std: float = 2.5e-6

    state_inflation: float = 1.02
    parameter_inflation: float = 1.0
    estimate_closure_parameters: bool = True
    seed: int = 2026

    def __post_init__(self):
        if self.ensemble_size < 3:
            raise ValueError("ensemble_size must be at least 3.")
        if self.assimilation_stride < 1:
            raise ValueError("assimilation_stride must be positive.")
        if self.observation_noise_std <= 0:
            raise ValueError("observation_noise_std must be positive.")
        if min(
            self.initial_x_std, self.initial_y_std, self.initial_q_std,
            self.process_noise_x, self.process_noise_y, self.process_noise_q,
            self.parameter_process_noise_std,
        ) < 0:
            raise ValueError("Uncertainty scales must be non-negative.")
        if self.state_inflation < 1.0 or self.parameter_inflation < 1.0:
            raise ValueError("Inflation factors must be at least one.")


def observation_indices(mode: ObservationMode) -> np.ndarray:
    mapping = {
        "x": np.array([0], dtype=int),
        "y": np.array([1], dtype=int),
        "xy": np.array([0, 1], dtype=int),
    }
    return mapping[mode]


def load_results(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def select_dynamic_model(data, model_name: str | None = None):
    if model_name is None:
        model_name = data["selected_model_name"]
    item = next(
        (r for r in data["results"] if r["spec"].name == model_name),
        None,
    )
    if item is None:
        available = ", ".join(r["spec"].name for r in data["results"])
        raise ValueError(f"Unknown model {model_name!r}. Available: {available}")
    test_item = next(
        (r for r in data["test_results"] if r["spec"].name == model_name),
        None,
    )
    if test_item is None:
        raise ValueError(f"No independent-test result for {model_name}.")
    return item, test_item


def validate_dynamic_closure_payload(data):
    required = (
        "system_name", "physical_test", "selected_model_name",
        "results", "test_results", "test",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Dynamic-closure pickle missing keys: {missing}")
    if data["system_name"] != "lorenz63_dynamic":
        raise ValueError(
            f"Expected system_name='lorenz63_dynamic'; got {data['system_name']!r}."
        )

    test = data["test"]
    t = np.asarray(test["t"], dtype=float)
    x = np.asarray(test["x_truth"], dtype=float)
    hidden = np.asarray(test["hidden"], dtype=float)
    g = np.asarray(test["g_truth"], dtype=float)
    qrec = np.asarray(test["q_recovered"], dtype=float)

    if x.shape != (len(t), 2):
        raise AssertionError("Independent x_truth must have shape (n,2).")
    for name, arr in (("hidden", hidden), ("g_truth", g), ("q_recovered", qrec)):
        if arr.shape != (len(t), 1):
            raise AssertionError(f"{name} must have shape (n,1).")
    if not all(np.all(np.isfinite(a)) for a in (t, x, hidden, g, qrec)):
        raise AssertionError("Non-finite independent-test data.")

    dt = float(np.median(np.diff(t)))
    if not np.isclose(dt, data["physical_test"].dt, rtol=1e-10, atol=1e-14):
        raise AssertionError("Time spacing does not match physical_test.dt.")

    print("\n" + "="*78)
    print("LORENZ-63 DYNAMIC DT INPUT CHECKS")
    print("="*78)
    print(f"Independent-test samples:            {len(t)}")
    print(f"dt:                                  {dt:.6e}")
    print(f"Selected closure model:              {data['selected_model_name']}")
    print("Hidden z used by filter:              NO")
    print("Exact G used by filter:               NO")
    print("Dynamic-DT input checks:              PASS")
    print("="*78)


def _scaled_features(augmented, spec, scaler):
    phi = problem.build_design_matrix(augmented, spec)
    return scaler.transform(phi)


def qdot_ensemble(state_ensemble, beta_ensemble, spec, scaler):
    phi_s = _scaled_features(state_ensemble, spec, scaler)
    return np.sum(phi_s * beta_ensemble, axis=1)


def rhs_ensemble(state_ensemble, beta_ensemble, spec, scaler, physical):
    """Vectorized RHS for all [x,y,q] ensemble members."""
    s = np.asarray(state_ensemble, dtype=float)
    x, y, q = s[:, 0], s[:, 1], s[:, 2]
    qdot = qdot_ensemble(s, beta_ensemble, spec, scaler)

    rhs = np.empty_like(s)
    rhs[:, 0] = physical.sigma * (y - x)
    rhs[:, 1] = physical.rho * x - y - x*q
    rhs[:, 2] = qdot
    return rhs


def rk4_ensemble(state, beta, dt, spec, scaler, physical):
    args = (beta, spec, scaler, physical)
    k1 = rhs_ensemble(state, *args)
    k2 = rhs_ensemble(state + 0.5*dt*k1, *args)
    k3 = rhs_ensemble(state + 0.5*dt*k2, *args)
    k4 = rhs_ensemble(state + dt*k3, *args)
    out = state + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
    if not np.all(np.isfinite(out)):
        raise RuntimeError("Non-finite dynamic-closure ensemble forecast.")
    return out


def forecast_ensemble(state, beta, n_steps, dt, spec, scaler, physical):
    out = np.asarray(state, dtype=float).copy()
    for _ in range(int(n_steps)):
        out = rk4_ensemble(out, beta, dt, spec, scaler, physical)
    return out


def inflate(arr, factor):
    mean = np.mean(arr, axis=0, keepdims=True)
    return mean + factor*(arr-mean)


def enkf_update(
    state_forecast,
    beta_forecast,
    observation,
    obs_indices,
    obs_noise_std,
    rng,
    estimate_parameters,
):
    """Joint stochastic EnKF update for [x,y,q,beta]."""
    state_forecast = np.asarray(state_forecast, dtype=float)
    beta_forecast = np.asarray(beta_forecast, dtype=float)

    ne, ns = state_forecast.shape
    aug = np.concatenate([state_forecast, beta_forecast], axis=1)
    ypred = state_forecast[:, obs_indices]

    a_aug = aug - aug.mean(axis=0, keepdims=True)
    a_y = ypred - ypred.mean(axis=0, keepdims=True)

    p_zy = a_aug.T @ a_y / (ne-1)
    p_yy = a_y.T @ a_y / (ne-1)

    if not estimate_parameters:
        p_zy[ns:] = 0.0

    r = obs_noise_std**2 * np.eye(len(obs_indices))
    gain = np.linalg.solve(p_yy + r, p_zy.T).T

    pert = observation + rng.normal(
        0.0, obs_noise_std, size=(ne, len(obs_indices))
    )
    updated = aug + (pert-ypred) @ gain.T
    return updated[:, :ns], updated[:, ns:], gain


def summary(arr):
    arr = np.asarray(arr, dtype=float)
    return (
        arr.mean(axis=0),
        arr.std(axis=0, ddof=1),
        np.percentile(arr, 2.5, axis=0),
        np.percentile(arr, 97.5, axis=0),
    )


def qoi_xy(state):
    state = np.asarray(state, dtype=float)
    return 0.25*(state[..., 0]**2 + state[..., 1]**2)


def run_dynamic_twin(data, trained, config, observe):
    test = data["test"]
    physical = data["physical_test"]
    spec = trained["spec"]
    scaler = trained["scaler"]
    beta_samples = np.asarray(trained["posterior"]["beta"], dtype=float)

    t = np.asarray(test["t"], dtype=float)
    truth_xy = np.asarray(test["x_truth"], dtype=float)
    truth_q = np.asarray(test["hidden"], dtype=float)[:, 0]  # verification only
    truth_aug = np.column_stack([truth_xy, truth_q])
    truth_g = np.asarray(test["g_truth"], dtype=float)

    dt = float(np.median(np.diff(t)))
    obs_idx = observation_indices(observe)
    assimilation_indices = np.arange(
        0, len(t), config.assimilation_stride, dtype=int
    )
    if assimilation_indices[-1] != len(t)-1:
        assimilation_indices = np.append(assimilation_indices, len(t)-1)

    rng = np.random.default_rng(config.seed)

    draw_ids = rng.choice(
        beta_samples.shape[0],
        size=config.ensemble_size,
        replace=beta_samples.shape[0] < config.ensemble_size,
    )
    beta = beta_samples[draw_ids].copy()

    # q0 comes from resolved-data reconstruction, not hidden z.
    q0 = float(np.asarray(test["q_recovered"])[0, 0])
    xy0 = truth_xy[0]
    state = np.column_stack([
        xy0[0] + rng.normal(0, config.initial_x_std, config.ensemble_size),
        xy0[1] + rng.normal(0, config.initial_y_std, config.ensemble_size),
        q0 + rng.normal(0, config.initial_q_std, config.ensemble_size),
    ])

    truth_obs = truth_xy[assimilation_indices][:, obs_idx]
    observations = truth_obs + rng.normal(
        0.0, config.observation_noise_std, size=truth_obs.shape
    )

    ncy = len(assimilation_indices)
    nstate = 3
    npar = beta.shape[1]
    saved = {
        "forecast_mean": np.empty((ncy,nstate)),
        "forecast_std": np.empty((ncy,nstate)),
        "forecast_lower": np.empty((ncy,nstate)),
        "forecast_upper": np.empty((ncy,nstate)),
        "analysis_mean": np.empty((ncy,nstate)),
        "analysis_std": np.empty((ncy,nstate)),
        "analysis_lower": np.empty((ncy,nstate)),
        "analysis_upper": np.empty((ncy,nstate)),
        "beta_mean": np.empty((ncy,npar)),
        "beta_std": np.empty((ncy,npar)),
        "closure_forecast_mean": np.empty((ncy,1)),
        "closure_forecast_std": np.empty((ncy,1)),
        "closure_analysis_mean": np.empty((ncy,1)),
        "closure_analysis_std": np.empty((ncy,1)),
        "qoi_analysis_mean": np.empty(ncy),
        "qoi_analysis_lower": np.empty(ncy),
        "qoi_analysis_upper": np.empty(ncy),
    }
    gain_norm = np.empty(ncy)

    previous = assimilation_indices[0]
    process_std = np.array([
        config.process_noise_x,
        config.process_noise_y,
        config.process_noise_q,
    ])

    for j,current in enumerate(assimilation_indices):
        n_steps = int(current-previous)
        if n_steps:
            state = forecast_ensemble(
                state,beta,n_steps,dt,spec,scaler,physical
            )
            if np.any(process_std):
                state += rng.normal(0.0,process_std,size=state.shape)
            if config.estimate_closure_parameters and config.parameter_process_noise_std:
                beta += rng.normal(
                    0.0,config.parameter_process_noise_std,size=beta.shape
                )

        state = inflate(state,config.state_inflation)
        if config.estimate_closure_parameters:
            beta = inflate(beta,config.parameter_inflation)

        fs = summary(state)
        for key,val in zip(
            ("forecast_mean","forecast_std","forecast_lower","forecast_upper"),fs
        ):
            saved[key][j] = val

        gf = -(state[:,0]*state[:,2])
        saved["closure_forecast_mean"][j,0] = np.mean(gf)
        saved["closure_forecast_std"][j,0] = np.std(gf,ddof=1)

        state,beta,gain = enkf_update(
            state,beta,observations[j],obs_idx,
            config.observation_noise_std,rng,
            config.estimate_closure_parameters,
        )
        gain_norm[j] = np.linalg.norm(gain,ord="fro")

        ans = summary(state)
        for key,val in zip(
            ("analysis_mean","analysis_std","analysis_lower","analysis_upper"),ans
        ):
            saved[key][j] = val
        saved["beta_mean"][j] = beta.mean(axis=0)
        saved["beta_std"][j] = beta.std(axis=0,ddof=1)

        ga = -(state[:,0]*state[:,2])
        saved["closure_analysis_mean"][j,0] = np.mean(ga)
        saved["closure_analysis_std"][j,0] = np.std(ga,ddof=1)

        qa = qoi_xy(state)
        saved["qoi_analysis_mean"][j] = np.mean(qa)
        saved["qoi_analysis_lower"][j],saved["qoi_analysis_upper"][j] = np.percentile(qa,[2.5,97.5])

        previous=current

    ids = assimilation_indices
    truth_at = truth_aug[ids]
    forecast_err_xy = saved["forecast_mean"][:,:2]-truth_xy[ids]
    analysis_err_xy = saved["analysis_mean"][:,:2]-truth_xy[ids]

    saved.update({
        "config": asdict(config),
        "assimilation_indices":ids,
        "assimilation_times":t[ids],
        "observation_indices":obs_idx,
        "observations":observations,
        "truth_at_assimilation":truth_at,
        "truth_xy_at_assimilation":truth_xy[ids],
        "truth_q_at_assimilation":truth_q[ids],
        "closure_truth_at_assimilation":truth_g[ids],
        "forecast_rmse":np.sqrt(np.mean(forecast_err_xy**2,axis=1)),
        "analysis_rmse":np.sqrt(np.mean(analysis_err_xy**2,axis=1)),
        "forecast_spread":np.sqrt(np.mean(saved["forecast_std"][:,:2]**2,axis=1)),
        "analysis_spread":np.sqrt(np.mean(saved["analysis_std"][:,:2]**2,axis=1)),
        "latent_q_forecast_rmse":np.abs(saved["forecast_mean"][:,2]-truth_q[ids]),
        "latent_q_analysis_rmse":np.abs(saved["analysis_mean"][:,2]-truth_q[ids]),
        "closure_forecast_rmse":np.abs(saved["closure_forecast_mean"][:,0]-truth_g[ids,0]),
        "closure_analysis_rmse":np.abs(saved["closure_analysis_mean"][:,0]-truth_g[ids,0]),
        "forecast_coverage":np.mean(
            (truth_xy[ids]>=saved["forecast_lower"][:,:2]) &
            (truth_xy[ids]<=saved["forecast_upper"][:,:2]),axis=1
        ),
        "analysis_coverage":np.mean(
            (truth_xy[ids]>=saved["analysis_lower"][:,:2]) &
            (truth_xy[ids]<=saved["analysis_upper"][:,:2]),axis=1
        ),
        "latent_q_analysis_coverage":(
            (truth_q[ids]>=saved["analysis_lower"][:,2]) &
            (truth_q[ids]<=saved["analysis_upper"][:,2])
        ).astype(float),
        "qoi_truth":qoi_xy(truth_aug[ids]),
        "kalman_gain_frobenius":gain_norm,
    })
    return saved


def validate_result(result):
    n=len(result["assimilation_times"])
    for key in (
        "forecast_mean","analysis_mean","forecast_lower","forecast_upper",
        "analysis_lower","analysis_upper"
    ):
        arr=np.asarray(result[key])
        if arr.shape!=(n,3) or not np.all(np.isfinite(arr)):
            raise AssertionError(f"{key}: invalid shape/finiteness {arr.shape}.")
    if np.any(result["analysis_lower"]>result["analysis_mean"]) or np.any(
        result["analysis_mean"]>result["analysis_upper"]
    ):
        raise AssertionError("Analysis interval does not bracket mean.")
    print("\n"+"="*78)
    print("LORENZ-63 DYNAMIC DT OUTPUT CHECKS")
    print("="*78)
    print(f"Assimilation cycles:                  {n}")
    print("Augmented state [x,y,q] dimensions:   PASS")
    print("Finite forecast/analysis values:      PASS")
    print("Analysis interval ordering:           PASS")
    print("Dynamic-DT output checks:             PASS")
    print("="*78)


def run_from_pickle(
    input_filename="closure_analysis_results_lorenz63_dynamic.pkl",
    output_filename="digital_twin_results_lorenz63_dynamic.pkl",
    *,
    model_name=None,
    observe: ObservationMode="xy",
    config=None,
):
    data=load_results(input_filename)
    validate_dynamic_closure_payload(data)
    trained,test_item=select_dynamic_model(data,model_name)
    cfg=config or DynamicEnKFConfig()

    print("\n"+"="*78)
    print("LORENZ-63 DYNAMIC DIGITAL-TWIN SETUP")
    print("="*78)
    print(f"Closure model:                       {trained['spec'].name}")
    print(f"Observed variables:                  {observe}")
    print(f"Ensemble size:                       {cfg.ensemble_size}")
    print(f"Assimilation stride:                 {cfg.assimilation_stride}")
    print(f"Assimilation interval:               {cfg.assimilation_stride*data['physical_test'].dt:.6e}")
    print(f"Observation-noise std:                {cfg.observation_noise_std:.6e}")
    print(f"Estimate closure parameters:          {cfg.estimate_closure_parameters}")
    print("Latent q observed directly:           NO")
    print("="*78)

    result=run_dynamic_twin(data,trained,cfg,observe)

    ids=result["assimilation_indices"]
    # Use the already-computed autonomous dynamic-closure test rollout as baseline.
    free=np.asarray(test_item["x_online"],dtype=float)
    if free.shape==(len(data["test"]["t"]),2) and np.all(np.isfinite(free)):
        free_xy=free[ids]
        truth=np.asarray(data["test"]["x_truth"])[ids]
        result["free_run_mean"]=free_xy
        result["free_run_rmse"]=np.sqrt(np.mean((free_xy-truth)**2,axis=1))
    else:
        result["free_run_mean"]=np.full((len(ids),2),np.nan)
        result["free_run_rmse"]=np.full(len(ids),np.nan)

    result.update({
        "source_filename":str(input_filename),
        "system_name":"lorenz63_dynamic_enkf",
        "display_name":"Lorenz-63 dynamic-closure digital twin",
        "model_name":trained["spec"].name,
        "observation_mode":observe,
        "state_labels":[r"$x$",r"$y$",r"$q$"],
        "feature_names":problem.feature_names(trained["spec"]),
    })

    validate_result(result)

    output=Path(output_filename)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(result,f)

    free_mean=float(np.nanmean(result["free_run_rmse"]))
    forecast_mean=float(np.mean(result["forecast_rmse"]))
    analysis_mean=float(np.mean(result["analysis_rmse"]))
    q_analysis=float(np.mean(result["latent_q_analysis_rmse"]))
    g_analysis=float(np.mean(result["closure_analysis_rmse"]))

    print("\n"+"="*78)
    print("LORENZ-63 DYNAMIC DT METRICS")
    print("="*78)
    print(f"Mean free-run state RMSE:             {free_mean:.6f}")
    print(f"Mean forecast state RMSE:             {forecast_mean:.6f}")
    print(f"Mean analysis state RMSE:             {analysis_mean:.6f}")
    print(f"Analysis improvement vs free run:     {100*(1-analysis_mean/max(free_mean,np.finfo(float).eps)):.2f}%")
    print(f"Analysis improvement vs forecast:     {100*(1-analysis_mean/max(forecast_mean,np.finfo(float).eps)):.2f}%")
    print(f"Mean latent-q analysis abs. error:     {q_analysis:.6f}")
    print(f"Mean closure analysis abs. error:      {g_analysis:.6f}")
    print(f"Mean forecast coverage (x,y):          {np.mean(result['forecast_coverage']):.3f}")
    print(f"Mean analysis coverage (x,y):          {np.mean(result['analysis_coverage']):.3f}")
    print(f"Mean latent-q analysis coverage:       {np.mean(result['latent_q_analysis_coverage']):.3f}")
    print(f"Saved to:                              {output}")
    print("="*78)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",nargs="?",default="closure_analysis_results_lorenz63_dynamic.pkl"
    )
    parser.add_argument(
        "--output",default="digital_twin_results_lorenz63_dynamic.pkl"
    )
    parser.add_argument("--model",default=None)
    parser.add_argument("--observe",choices=("x","y","xy"),default="xy")
    parser.add_argument("--ensemble-size",type=int,default=150)
    parser.add_argument("--assimilation-stride",type=int,default=5)
    parser.add_argument("--observation-noise",type=float,default=0.50)
    parser.add_argument("--fixed-parameters",action="store_true") #, default=True
    parser.add_argument("--seed",type=int,default=2026)
    args=parser.parse_args()

    base=DynamicEnKFConfig()
    cfg=DynamicEnKFConfig(
        ensemble_size=args.ensemble_size,
        assimilation_stride=args.assimilation_stride,
        observation_noise_std=args.observation_noise,
        initial_x_std=base.initial_x_std,
        initial_y_std=base.initial_y_std,
        initial_q_std=base.initial_q_std,
        process_noise_x=base.process_noise_x,
        process_noise_y=base.process_noise_y,
        process_noise_q=base.process_noise_q,
        parameter_process_noise_std=base.parameter_process_noise_std,
        state_inflation=base.state_inflation,
        parameter_inflation=base.parameter_inflation,
        estimate_closure_parameters=not args.fixed_parameters,
        seed=args.seed,
    )
    run_from_pickle(
        args.input,args.output,model_name=args.model,observe=args.observe,config=cfg
    )


if __name__=="__main__":
    main()
