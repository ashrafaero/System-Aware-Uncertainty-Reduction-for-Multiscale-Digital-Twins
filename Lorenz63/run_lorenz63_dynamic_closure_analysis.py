"""Bayesian dynamic-closure analysis for partially observed Lorenz-63.

The learned closure is dynamic:
    G_hat = -x q,
    qdot  = F(x,y,q;theta).

q is reconstructed from resolved data only through the recovered closure
G_rec = ydot-(rho*x-y), not from the hidden z trajectory.  Hidden z is retained
only for synthetic benchmark diagnostics.
"""

from __future__ import annotations

import argparse
import pickle
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import lorenz63_dynamic_system as problem
from closure_inference import (
    build_design_matrix,
    fit_feature_scaler,
    run_mcmc_regression,
)


@dataclass(frozen=True)
class Lorenz63DynamicAnalysisConfig:
    observation_noise_std: float = 0.0
    train_start_fraction: float = 1.0/6.0
    train_end_fraction: float = 3.0/4.0

    derivative_method: str = "savgol"
    savgol_window: int = 51
    savgol_polyorder: int = 3

    latent_min_abs_x: float = 0.10
    latent_savgol_window: int = 101
    latent_savgol_polyorder: int = 3

    mcmc_time_stride: int = 5
    warmup_steps: int = 300
    num_samples: int = 600
    test_seed_offset: int = 100

    validation_fraction: float = 0.20
    selection_relative_tolerance: float = 0.05
    forecast_error_threshold: float = 0.30
    forecast_persistence_time: float = 0.10
    run_prechecks: bool = True


def _training_slice(n, cfg):
    i0 = int(round(cfg.train_start_fraction*n))
    i1 = int(round(cfg.train_end_fraction*n))
    if not 0 <= i0 < i1 <= n:
        raise ValueError("Invalid training interval.")
    return slice(i0, i1)


def _rmse(a,b):
    a,b=np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    return float(np.sqrt(np.mean((a-b)**2)))


def _nrmse(a,b):
    return _rmse(a,b)/max(float(np.std(a)),np.finfo(float).eps)


def _corr(a,b):
    a=np.asarray(a).reshape(-1); b=np.asarray(b).reshape(-1)
    return float(np.corrcoef(a,b)[0,1])


def _r2(a,b):
    a=np.asarray(a).reshape(-1); b=np.asarray(b).reshape(-1)
    den=np.sum((a-a.mean())**2)
    return float(1.0-np.sum((a-b)**2)/max(den,np.finfo(float).eps))


def fidelity(reference, prediction):
    return {
        "rmse": _rmse(reference,prediction),
        "nrmse_std": _nrmse(reference,prediction),
        "r2": _r2(reference,prediction),
        "corr": _corr(reference,prediction),
        "bias": float(np.mean(np.asarray(prediction)-np.asarray(reference))),
    }


def posterior_prediction(augmented, spec, scaler, beta_samples, credible_mass=0.95):
    phi = build_design_matrix(augmented, spec, problem)
    phi_s = scaler.transform(phi)
    beta = np.asarray(beta_samples,dtype=float)
    mean = phi_s @ beta.mean(axis=0)

    alpha=(1.0-credible_mass)/2.0
    lo=np.empty(len(augmented)); hi=np.empty(len(augmented))
    chunk=500
    for i in range(0,len(augmented),chunk):
        j=min(i+chunk,len(augmented))
        draws=phi_s[i:j] @ beta.T
        lo[i:j]=np.quantile(draws,alpha,axis=1)
        hi[i:j]=np.quantile(draws,1-alpha,axis=1)
    return mean[:,None],lo[:,None],hi[:,None]


def integrate_dynamic_closure(t, x0, q0, beta_mean, spec, scaler, physical):
    t=np.asarray(t,dtype=float)
    beta_mean=np.asarray(beta_mean,dtype=float)

    def rhs(time,state):
        x,y,q=state
        aug=np.array([[x,y,q]],dtype=float)
        phi=build_design_matrix(aug,spec,problem)
        qdot=float(scaler.transform(phi)[0] @ beta_mean)
        g=-x*q
        return np.array([
            physical.sigma*(y-x),
            physical.rho*x-y+g,
            qdot,
        ])

    sol=solve_ivp(
        rhs,(float(t[0]),float(t[-1])),
        np.array([x0[0],x0[1],float(q0)],dtype=float),
        t_eval=t,rtol=1e-9,atol=1e-11,
    )
    if not sol.success or sol.y.shape[1] != t.size:
        raise RuntimeError(sol.message or "Dynamic closure integration ended early.")
    out=sol.y.T
    if not np.all(np.isfinite(out)):
        raise RuntimeError("Non-finite dynamic reduced trajectory.")
    return out


def normalized_state_error(reference,prediction):
    reference=np.asarray(reference,dtype=float)
    prediction=np.asarray(prediction,dtype=float)
    scale=np.std(reference,axis=0)
    scale=np.where(scale>np.finfo(float).eps,scale,1.0)
    return np.sqrt(np.mean(((prediction-reference)/scale[None,:])**2,axis=1))


def forecast_horizon(t,error,threshold,persistence_time):
    t=np.asarray(t,dtype=float); error=np.asarray(error,dtype=float)
    dt=float(np.median(np.diff(t)))
    n=max(1,int(round(persistence_time/dt)))
    bad=error>threshold
    if n==1:
        inds=np.flatnonzero(bad)
    else:
        inds=np.flatnonzero(np.convolve(bad.astype(int),np.ones(n,dtype=int),mode="valid")>=n)
    if inds.size==0:
        return float(t[-1]-t[0])
    return float(t[inds[0]]-t[0])


def screen_candidates(aug,qdot,candidates,cfg):
    n=len(aug)
    split=int(round((1.0-cfg.validation_fraction)*n))
    split=min(max(split,20),n-20)
    rows=[]
    print("\n"+"="*100)
    print("DYNAMIC-CLOSURE CANDIDATE SCREEN: qdot = F(x,y,q)")
    print("="*100)
    print(f"{'model':<22}{'nfeat':>8}{'fit NRMSE':>14}{'val NRMSE':>14}{'val R2':>12}{'val corr':>12}")
    print("-"*100)
    for spec in candidates:
        pf=build_design_matrix(aug[:split],spec,problem)
        pv=build_design_matrix(aug[split:],spec,problem)
        beta,*_=np.linalg.lstsq(pf,qdot[:split,0],rcond=None)
        fit=(pf@beta)[:,None]; val=(pv@beta)[:,None]
        fm=fidelity(qdot[:split],fit); vm=fidelity(qdot[split:],val)
        rows.append({"spec":spec,"fit":fm,"validation":vm,"n_features":pf.shape[1]})
        print(f"{spec.name:<22}{pf.shape[1]:>8d}{fm['nrmse_std']:>14.6f}{vm['nrmse_std']:>14.6f}{vm['r2']:>12.6f}{vm['corr']:>12.6f}")
    print("-"*100)

    # For this dynamic Lorenz-63 closure, use the quadratic model as the
    # primary candidate once it passes a stringent held-out accuracy test.
    # A quadratic library is the minimum generic polynomial library that can
    # represent the latent interaction xy and a linear relaxation in q.  The
    # cubic model remains a flexibility diagnostic, but is not promoted merely
    # for shaving an already tiny validation residual.
    quadratic = next(
        (r for r in rows if r["spec"].name == "xyq-quadratic"),
        None,
    )
    if (
        quadratic is not None
        and quadratic["validation"]["nrmse_std"] <= 5.0e-3
        and quadratic["validation"]["r2"] >= 0.999
    ):
        selected = quadratic
        selection_reason = (
            "quadratic passed the stringent held-out dynamic-closure threshold "
            "(validation NRMSE <= 5e-3 and R2 >= 0.999)"
        )
    else:
        best_val=min(r["validation"]["nrmse_std"] for r in rows)
        admissible=[
            r for r in rows
            if r["validation"]["nrmse_std"] <= (1.0+cfg.selection_relative_tolerance)*best_val
        ]
        selected=min(admissible,key=lambda r:r["n_features"])
        selection_reason = (
            f"simplest model within {100*cfg.selection_relative_tolerance:.1f}% "
            "of best validation NRMSE"
        )

    print(f"Selected candidate: {selected['spec'].name}")
    print(f"Selection reason: {selection_reason}.")
    print("="*100)
    return rows,selected["spec"].name


def prepare_resolved_closure_data(t,x_truth,x_obs,g_truth,hidden,physical,cfg,sl):
    g_rec=problem.recover_closure_from_x(
        x_obs,physical,derivative=cfg.derivative_method,
        sg_window=cfg.savgol_window,sg_polyorder=cfg.savgol_polyorder,
    )
    q_rec,ratio_valid=problem.reconstruct_latent_q(
        x_obs,g_rec,min_abs_x=cfg.latent_min_abs_x,
    )
    qdot_rec=problem.differentiate_latent_q(
        q_rec,physical,sg_window=cfg.latent_savgol_window,
        sg_polyorder=cfg.latent_savgol_polyorder,
    )
    return {
        "t":np.asarray(t)[sl],
        "x_truth":np.asarray(x_truth)[sl],
        "x_obs":np.asarray(x_obs)[sl],
        "g_truth":np.asarray(g_truth)[sl],
        "hidden":np.asarray(hidden)[sl],
        "g_recovered":np.asarray(g_rec)[sl],
        "q_recovered":np.asarray(q_rec)[sl],
        "qdot_recovered":np.asarray(qdot_rec)[sl],
        "ratio_valid":np.asarray(ratio_valid)[sl],
    }


def fit_models(train,candidates,physical,cfg):
    aug=problem.dynamic_augmented_state(train["x_obs"],train["q_recovered"])
    problem.check_dynamic_feature_library(
        aug[::max(1,cfg.mcmc_time_stride)],candidates,verbose=True
    )
    screen,selected_name=screen_candidates(
        aug,train["qdot_recovered"],candidates,cfg
    )

    results=[]
    print("\n"+"="*100)
    print("BAYESIAN DYNAMIC-CLOSURE FITS")
    print("="*100)
    for j,spec in enumerate(candidates):
        tic=time.perf_counter()
        aug_fit=aug[::cfg.mcmc_time_stride]
        qdot_fit=train["qdot_recovered"][::cfg.mcmc_time_stride]
        phi=build_design_matrix(aug_fit,spec,problem)
        scaler=fit_feature_scaler(phi)
        posterior=run_mcmc_regression(
            scaler.transform(phi),qdot_fit,
            warmup_steps=cfg.warmup_steps,
            num_samples=cfg.num_samples,
            seed=physical.seed+j,
        )
        qdot_mean,qdot_lo,qdot_hi=posterior_prediction(
            aug,spec,scaler,posterior["beta"]
        )
        beta_mean=np.mean(posterior["beta"],axis=0)

        online=integrate_dynamic_closure(
            train["t"],train["x_obs"][0],train["q_recovered"][0,0],
            beta_mean,spec,scaler,physical,
        )
        x_online=online[:,:2]
        q_online=online[:,[2]]
        g_online=problem.closure_from_augmented_state(online)

        err=normalized_state_error(train["x_truth"],x_online)
        evaluation={
            "qdot":fidelity(train["qdot_recovered"],qdot_mean),
            "closure":fidelity(train["g_truth"],g_online),
            "latent_q":fidelity(train["hidden"],q_online), # diagnostic only
            "state_rmse":_rmse(train["x_truth"],x_online),
            "state_nrmse":_nrmse(train["x_truth"],x_online),
            "forecast_horizon":forecast_horizon(
                train["t"],err,cfg.forecast_error_threshold,
                cfg.forecast_persistence_time,
            ),
            "normalized_error_curve":err,
        }

        results.append({
            "spec":spec,"scaler":scaler,"posterior":posterior,
            "qdot_mean":qdot_mean,"qdot_lower":qdot_lo,"qdot_upper":qdot_hi,
            "online_augmented":online,"x_online":x_online,
            "q_online":q_online,"g_online":g_online,
            "evaluation":evaluation,
        })
        print(
            f"{spec.name:<22} qdot NRMSE={evaluation['qdot']['nrmse_std']:.5f} | "
            f"G NRMSE={evaluation['closure']['nrmse_std']:.5f} | "
            f"horizon={evaluation['forecast_horizon']:.3f} | "
            f"{time.perf_counter()-tic:.1f}s"
        )
    print("="*100)
    return results,screen,selected_name


def _restart_window_metrics(data, source, physical, cfg, window_time=2.0, n_windows=12):
    """Evaluate autonomous skill on repeated finite windows.

    For a chaotic system, a single 30-time-unit free run can fail because tiny
    learned-vector-field errors eventually amplify.  Repeated restart windows
    are a better diagnostic of local autonomous model quality and also prevent
    one numerical blow-up from destroying the entire independent-test report.
    """
    spec=source["spec"]; scaler=source["scaler"]; posterior=source["posterior"]
    beta_mean=np.mean(posterior["beta"],axis=0)

    t=np.asarray(data["t"],dtype=float)
    x=np.asarray(data["x_truth"],dtype=float)
    q=np.asarray(data["q_recovered"],dtype=float)
    dt=float(np.median(np.diff(t)))
    nstep=max(2,int(round(window_time/dt))+1)

    max_start=max(0,len(t)-nstep)
    starts=np.unique(np.linspace(0,max_start,min(n_windows,max_start+1),dtype=int))

    state_nrmse=[]
    closure_nrmse=[]
    horizons=[]
    failed=0
    for i0 in starts:
        i1=min(i0+nstep,len(t))
        tw=t[i0:i1]
        try:
            online=integrate_dynamic_closure(
                tw,x[i0],q[i0,0],beta_mean,spec,scaler,physical
            )
        except RuntimeError:
            failed += 1
            continue

        xp=online[:,:2]
        gp=problem.closure_from_augmented_state(online)
        gt=np.asarray(data["g_truth"])[i0:i1]
        err=normalized_state_error(x[i0:i1],xp)

        state_nrmse.append(_nrmse(x[i0:i1],xp))
        closure_nrmse.append(_nrmse(gt,gp))
        horizons.append(
            forecast_horizon(
                tw,err,cfg.forecast_error_threshold,cfg.forecast_persistence_time
            )
        )

    return {
        "window_time": float(window_time),
        "n_windows": int(len(starts)),
        "n_successful_windows": int(len(starts)-failed),
        "n_failed_windows": int(failed),
        "mean_state_nrmse": float(np.mean(state_nrmse)) if state_nrmse else np.inf,
        "mean_closure_nrmse": float(np.mean(closure_nrmse)) if closure_nrmse else np.inf,
        "mean_forecast_horizon": float(np.mean(horizons)) if horizons else 0.0,
    }


def evaluate_independent(test,trained,physical,cfg):
    aug=problem.dynamic_augmented_state(test["x_obs"],test["q_recovered"])
    results=[]
    for source in trained:
        spec=source["spec"]; scaler=source["scaler"]; posterior=source["posterior"]
        qdot_mean,qdot_lo,qdot_hi=posterior_prediction(
            aug,spec,scaler,posterior["beta"]
        )
        beta_mean=np.mean(posterior["beta"],axis=0)
        restart_metrics = _restart_window_metrics(
            test, source, physical, cfg, window_time=2.0, n_windows=12
        )

        long_run_success = True
        try:
            online=integrate_dynamic_closure(
                test["t"],test["x_obs"][0],test["q_recovered"][0,0],
                beta_mean,spec,scaler,physical,
            )
            x_online=online[:,:2]
            q_online=online[:,[2]]
            g_online=problem.closure_from_augmented_state(online)
            err=normalized_state_error(test["x_truth"],x_online)
            closure_metrics=fidelity(test["g_truth"],g_online)
            latent_metrics=fidelity(test["hidden"],q_online)
            state_rmse=_rmse(test["x_truth"],x_online)
            state_nrmse=_nrmse(test["x_truth"],x_online)
            horizon=forecast_horizon(
                test["t"],err,cfg.forecast_error_threshold,
                cfg.forecast_persistence_time,
            )
        except RuntimeError as exc:
            long_run_success = False
            online=np.full((len(test["t"]),3),np.nan,dtype=float)
            x_online=online[:,:2]
            q_online=online[:,[2]]
            g_online=np.full((len(test["t"]),1),np.nan,dtype=float)
            err=np.full(len(test["t"]),np.nan,dtype=float)
            closure_metrics={"rmse":np.inf,"nrmse_std":np.inf,"r2":-np.inf,"corr":np.nan,"bias":np.nan}
            latent_metrics={"rmse":np.inf,"nrmse_std":np.inf,"r2":-np.inf,"corr":np.nan,"bias":np.nan}
            state_rmse=np.inf
            state_nrmse=np.inf
            horizon=0.0
            print(
                f"WARNING: {spec.name} full independent rollout failed: {exc}"
            )

        evaluation={
            "qdot":fidelity(test["qdot_recovered"],qdot_mean),
            "closure":closure_metrics,
            "latent_q":latent_metrics,
            "state_rmse":state_rmse,
            "state_nrmse":state_nrmse,
            "forecast_horizon":horizon,
            "normalized_error_curve":err,
            "long_run_success":long_run_success,
            "restart_window":restart_metrics,
        }
        results.append({
            "spec":spec,"scaler":scaler,"posterior":posterior,
            "qdot_mean":qdot_mean,"qdot_lower":qdot_lo,"qdot_upper":qdot_hi,
            "online_augmented":online,"x_online":x_online,
            "q_online":q_online,"g_online":g_online,
            "evaluation":evaluation,
        })
    return results


def print_summary(results,test_results,selected_name):
    tr=next(r for r in results if r["spec"].name==selected_name)
    te=next(r for r in test_results if r["spec"].name==selected_name)
    print("\n"+"="*84)
    print(f"REPORTABLE DYNAMIC-CLOSURE SUMMARY: {selected_name}")
    print("="*84)
    for label,item in (("TRAINING",tr),("INDEPENDENT TEST",te)):
        ev=item["evaluation"]
        print(label)
        print(f"  qdot NRMSE / R2 / corr : {ev['qdot']['nrmse_std']:.6f} / {ev['qdot']['r2']:.6f} / {ev['qdot']['corr']:.6f}")
        print(f"  G NRMSE / R2 / corr    : {ev['closure']['nrmse_std']:.6f} / {ev['closure']['r2']:.6f} / {ev['closure']['corr']:.6f}")
        print(f"  state RMSE / NRMSE     : {ev['state_rmse']:.6f} / {ev['state_nrmse']:.6f}")
        print(f"  forecast horizon        : {ev['forecast_horizon']:.6f}")
        print(f"  q-vs-hidden-z NRMSE*    : {ev['latent_q']['nrmse_std']:.6f}")
        if "restart_window" in ev:
            rw=ev["restart_window"]
            print(
                f"  restart-window state NRMSE: {rw['mean_state_nrmse']:.6f} "
                f"({rw['n_successful_windows']}/{rw['n_windows']} successful)"
            )
            print(f"  restart-window mean horizon: {rw['mean_forecast_horizon']:.6f}")
        if "long_run_success" in ev:
            print(f"  full 30-unit rollout success: {ev['long_run_success']}")
    print("* hidden z is verification only; it is not used to fit the dynamic closure.")
    print("="*84)


def run_analysis(
    output_filename="closure_analysis_results_lorenz63_dynamic.pkl",
    analysis=None,
):
    cfg=analysis or Lorenz63DynamicAnalysisConfig()
    physical=problem.default_config()

    if cfg.run_prechecks:
        precheck=problem.run_preanalysis_checks(
            physical,
            derivative=cfg.derivative_method,
            sg_window=cfg.savgol_window,
            sg_polyorder=cfg.savgol_polyorder,
            latent_min_abs_x=cfg.latent_min_abs_x,
            latent_sg_window=cfg.latent_savgol_window,
            latent_sg_polyorder=cfg.latent_savgol_polyorder,
        )
    else:
        precheck={}

    t,x_true,hidden,g_true=problem.generate_truth(physical)
    x_obs=problem.add_observation_noise(
        x_true,cfg.observation_noise_std,physical.seed+1
    )
    sl=_training_slice(len(t),cfg)
    train=prepare_resolved_closure_data(
        t,x_true,x_obs,g_true,hidden,physical,cfg,sl
    )

    candidates=problem.default_model_specs()
    results,screen,selected_name=fit_models(
        train,candidates,physical,cfg
    )

    physical_test=replace(physical,seed=physical.seed+cfg.test_seed_offset)
    tt,xt,ht,gt=problem.generate_truth(physical_test)
    xo=problem.add_observation_noise(
        xt,cfg.observation_noise_std,physical_test.seed+1
    )
    test=prepare_resolved_closure_data(
        tt,xt,xo,gt,ht,physical_test,cfg,slice(0,len(tt))
    )
    test_results=evaluate_independent(
        test,results,physical_test,cfg
    )

    print_summary(results,test_results,selected_name)

    payload={
        "system_name":problem.SYSTEM_NAME,
        "display_name":problem.DISPLAY_NAME,
        "physical":physical,
        "physical_test":physical_test,
        "analysis":asdict(cfg),
        "precheck":precheck,
        "candidate_screen":screen,
        "selected_model_name":selected_name,
        "train":train,
        "test":test,
        "results":results,
        "test_results":test_results,
    }
    output_path=Path(output_filename)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(payload,f)
    print(f"Saved dynamic closure results to: {output_path}")
    return output_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",default="closure_analysis_results_lorenz63_dynamic.pkl"
    )
    args=parser.parse_args()
    run_analysis(args.output)


if __name__=="__main__":
    main()
