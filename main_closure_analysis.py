"""Run unresolved-closure recovery for one selected canonical problem."""

from __future__ import annotations

from dataclasses import replace
# import argparse
import os

import numpy as np
import time
from scipy.signal import savgol_filter

from problem_registry import get_problem #, available_problems

from closure_inference import (
    build_design_matrix,
    fit_feature_scaler,
    integrate_learned_closure,
    posterior_closure_mean,
    run_mcmc_regression,
    state_metrics,
)

from closure_evaluation import (
    AnalysisConfig,
    EvaluationConfig,
    closure_fidelity,
    evaluate_closure_model,
    print_metric_table,
    save_results,
)

def training_slice(nt, cfg):
    i0 = int(round(cfg.train_start_fraction * nt))
    i1 = int(round(cfg.train_end_fraction * nt))
    if not (0 <= i0 < i1 <= nt):
        raise ValueError("Invalid training interval.")
    return slice(i0, i1)


def run_analysis(system_name: str, analysis: AnalysisConfig | None = None):
    analysis = analysis or AnalysisConfig()
    problem = get_problem(system_name)
    physical = problem.default_config()
    model_specs = problem.default_model_specs()

    evaluation_cfg = EvaluationConfig(
        credible_mass=0.95,
        forecast_error_threshold=analysis.forecast_error_threshold,
        forecast_persistence=analysis.forecast_persistence,
        acf_max_lag=analysis.acf_max_lag,
        psd_nperseg=analysis.psd_nperseg,
        spatial_max_lag=analysis.spatial_max_lag,
        posterior_chunk_time=analysis.posterior_chunk_time,
    )

    print(f"\nSYSTEM: {problem.DISPLAY_NAME}")

    # 1) Complete synthetic truth; hidden state is NOT used in fitting.
    t, x_true, hidden_true, g_true = problem.generate_truth(physical)
    x_obs = problem.add_observation_noise(
        x_true, analysis.observation_noise_std, seed=physical.seed + 1
    )

    # 2) Recover G from resolved observations only.
    g_recovered = problem.recover_closure_from_x(
        x_obs,
        physical,
        derivative=analysis.derivative_method,
        sg_window=analysis.savgol_window,
        sg_polyorder=analysis.savgol_polyorder,
    )
    print("\nClosure extraction: recovered G vs hidden true G")
    print(closure_fidelity(g_true, g_recovered))

    x_from_recovered_g = problem.integrate_with_prescribed_closure(
        t, x_obs[0], g_recovered, physical
    )
    print("Resolved-state reconstruction from prescribed recovered G")
    print(state_metrics(x_true, x_from_recovered_g))

    

    # 3) Training interval.
    sl = training_slice(len(t), analysis)
    t_train = t[sl]
    x_train = x_obs[sl]
    g_train = g_recovered[sl]
    g_true_train = g_true[sl]
    
    # if system_name == "lorenz63":
    #     augmented_train = np.column_stack(
    #         [x_train, g_train]
    #     )

    #     dgdt_train = savgol_filter(
    #         g_train[:, 0],
    #         window_length=101,
    #         polyorder=3,
    #         deriv=1,
    #         delta=physical.dt,
    #         mode="interp",
    #     )[:, None]
    
   

    results = []

    for spec in model_specs:
        print(f"\nFITTING: {spec.name}")
        
        x_fit = x_train[::analysis.mcmc_time_stride]
        g_fit = g_train[::analysis.mcmc_time_stride]

        # if system_name == "lorenz63":
        #     x_fit = augmented_train[::analysis.mcmc_time_stride]
        #     g_fit = dgdt_train[::analysis.mcmc_time_stride]
        # else:
            
    

        phi = build_design_matrix(x_fit, spec, problem)
        scaler = fit_feature_scaler(phi)
        phi_scaled = scaler.transform(phi)

        posterior = run_mcmc_regression(
            phi_scaled,
            g_fit,
            warmup_steps=analysis.warmup_steps,
            num_samples=analysis.num_samples,
            seed=physical.seed,
        )

        print("  MCMC finished")

        t0 = time.perf_counter()

        g_mean = posterior_closure_mean(
            x_train, spec, scaler, posterior["beta"], problem
        )
        
            
    

        print(
            f"  posterior_closure_mean finished "
            f"in {time.perf_counter() - t0:.2f} s"
        )

        beta_mean = posterior["beta"].mean(axis=0)

        t0 = time.perf_counter()

        try:
            x_online = integrate_learned_closure(
                t, x_obs[0], beta_mean, spec, scaler, physical, problem
            )
            
            print(
                f"  integrate_learned_closure finished "
                f"in {time.perf_counter() - t0:.2f} s"
            )

        except RuntimeError as err:
            print(f"  A-posteriori integration FAILED: {err}")
            print(f"  Skipping remaining evaluation for {spec.name}.")
            continue
        

    
        
        t0 = time.perf_counter()

        evaluation = evaluate_closure_model(
            t=t,
            dt=physical.dt,
            x_reference=x_true,
            x_prediction=x_online,
            g_reference=g_train,
            g_prediction=g_mean,
            x_for_posterior=x_train,
            spec=spec,
            scaler=scaler,
            posterior=posterior,
            problem=problem,
            config=evaluation_cfg,
        )
        
        
        
        

        print(
            f"  evaluate_closure_model finished "
            f"in {time.perf_counter() - t0:.2f} s"
        )

        t0 = time.perf_counter()

        truth_closure = closure_fidelity(
            g_true_train,
            g_mean,
        )
    
    

        print(
            f"  truth closure metrics finished "
            f"in {time.perf_counter() - t0:.2f} s"
        )
        
        results.append({
            "spec": spec,
            "posterior": posterior,
            "scaler": scaler,
            "g_mean": g_mean,
            "x_online": x_online,
            "evaluation": evaluation,
            "truth_closure": truth_closure,
        })

        print(f"  COMPLETED MODEL: {spec.name}")
    


    # 4) Independent-trajectory test, no refit.
    print("\nStarting independent-trajectory evaluation...")
    physical_test = replace(
        physical, seed=physical.seed + analysis.test_seed_offset
    )
    t_test, x_true_test, hidden_test, g_true_test = problem.generate_truth(
        physical_test
    )
    x_obs_test = problem.add_observation_noise(
        x_true_test,
        analysis.observation_noise_std,
        seed=physical_test.seed + 1,
    )
    g_recovered_test = problem.recover_closure_from_x(
        x_obs_test,
        physical_test,
        derivative=analysis.derivative_method,
        sg_window=analysis.savgol_window,
        sg_polyorder=analysis.savgol_polyorder,
    )
    

        
    model_target_test = g_recovered_test
    

    test_results = []
    for trained in results:
        spec = trained["spec"]
        posterior = trained["posterior"]
        scaler = trained["scaler"]
        beta_mean = posterior["beta"].mean(axis=0)

        print(f"\nTESTING: {spec.name}")

        t0 = time.perf_counter()

        g_test_mean = posterior_closure_mean(
            x_obs_test,
            spec,
            scaler,
            posterior["beta"],
            problem,
        )
    

        print(
            f"  test posterior_closure_mean finished "
            f"in {time.perf_counter() - t0:.2f} s"
        )

        t0 = time.perf_counter()



        x_test_online = integrate_learned_closure(
            t_test,
            x_obs_test[0],
            beta_mean,
            spec,
            scaler,
            physical_test,
            problem,
        )
    

        print(
            f"  test integrate_learned_closure finished "
            f"in {time.perf_counter() - t0:.2f} s"
        )

        t0 = time.perf_counter()

        evaluation = evaluate_closure_model(
            t=t_test,
            dt=physical_test.dt,
            x_reference=x_true_test,
            x_prediction=x_test_online,
            g_reference=model_target_test,
            g_prediction=g_test_mean,
            x_for_posterior=x_obs_test,
            spec=spec,
            scaler=scaler,
            posterior=posterior,
            problem=problem,
            config=evaluation_cfg,
        )
        



        print(
            f"  test evaluate_closure_model finished "
            f"in {time.perf_counter() - t0:.2f} s"
        )
        
        if system_name == "lorenz63":
            dgdt_true_test = savgol_filter(
                g_true_test[:, 0],
                window_length=101,
                polyorder=3,
                deriv=1,
                delta=physical_test.dt,
                mode="interp",
            )[:, None]

            truth_closure_test = closure_fidelity(
                dgdt_true_test,
                g_test_mean,
            )
        else:
            truth_closure_test = closure_fidelity(
                g_true_test,
                g_test_mean,
            )
    

        test_results.append({
            "spec": spec,
            "posterior": posterior,
            "scaler": scaler,
            "g_mean": g_test_mean,
            "x_online": x_test_online,
            "evaluation": evaluation,
            "truth_closure": truth_closure_test,
        })

        print(f"  COMPLETED TEST MODEL: {spec.name}")

    

    print_metric_table(results, heading="TRAINING-REALIZATION METRICS")
    print_metric_table(
        test_results,
        heading="INDEPENDENT-TRAJECTORY GENERALIZATION (NO REFIT)",
    )

    filename = f"closure_analysis_results_{problem.SYSTEM_NAME}.pkl"

    save_results(
        filename,
        system_name=problem.SYSTEM_NAME,
        display_name=problem.DISPLAY_NAME,
        state_labels=problem.state_labels(physical),
        closure_label=problem.closure_label(physical),

        physical=physical,
        physical_test=physical_test,
        analysis=analysis,
        evaluation_cfg=evaluation_cfg,

        t=t,
        x_true=x_true,
        hidden_true=hidden_true,
        g_true=g_true,
        g_recovered=g_recovered,
        t_train=t_train,
        x_train=x_train,
        g_train=g_train,
        results=results,

        t_test=t_test,
        x_true_test=x_true_test,
        hidden_test=hidden_test,
        g_true_test=g_true_test,
        g_recovered_test=g_recovered_test,
        test_results=test_results,
    )

    return filename


def main():
    systems = ["linear",  "lorenz96"] #"lorenz63",
    rerun_flags = [False,  True] #False,

    for system, should_rerun in zip(systems, rerun_flags):
        filename = f"closure_analysis_results_{system}.pkl"

        if os.path.exists(filename) and not should_rerun:
            print(f"\n{filename} already exists. Skipping {system}.")
            continue

        run_analysis(system)

#%%
if __name__ == "__main__":
    main()
