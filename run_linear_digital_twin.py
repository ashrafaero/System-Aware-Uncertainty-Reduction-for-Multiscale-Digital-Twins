"""Run an EnKF digital twin from a saved slow-fast linear closure pickle."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from enkf_digital_twin import EnKFConfig, run_sequential_digital_twin
from problem_registry import get_problem


def _load(filename: str | Path) -> dict[str, Any]:
    with Path(filename).open("rb") as stream:
        return pickle.load(stream)


def select_training_model(
    results: list[dict[str, Any]], model_name: str | None = None
) -> dict[str, Any]:
    """Select the closure using training information only."""
    if not results:
        raise RuntimeError("The source pickle contains no successful models.")
    if model_name is None:
        return min(
            results,
            key=lambda item: item["evaluation"]["closure"]["nrmse_std"],
        )
    selected = next(
        (item for item in results if item["spec"].name == model_name), None
    )
    if selected is None:
        names = ", ".join(item["spec"].name for item in results)
        raise ValueError(f"Unknown model '{model_name}'. Available: {names}")
    return selected


def default_linear_enkf_config() -> EnKFConfig:
    """Conservative defaults for the scalar resolved state."""
    return EnKFConfig(
        ensemble_size=40,
        assimilation_stride=25,
        observed_state_stride=1,
        observation_noise_std=0.02,
        initial_state_std=0.15,
        process_noise_std=0.002,
        parameter_process_noise_std=1.0e-5,
        state_inflation=1.01,
        parameter_inflation=1.002,
        localization_radius=None,
        estimate_closure_parameters=True,
        seed=2026,
    )


def run_from_pickle(
    input_filename: str | Path = "closure_analysis_results_linear.pkl",
    output_filename: str | Path = "digital_twin_results_linear.pkl",
    *,
    model_name: str | None = None,
    config: EnKFConfig | None = None,
) -> Path:
    data = _load(input_filename)
    if data.get("system_name") != "linear":
        raise ValueError("This runner requires a slow-fast linear-system pickle.")
    problem = get_problem("linear")
    trained = select_training_model(data["results"], model_name)
    trained_test = next(
        (
            item for item in data["test_results"]
            if item["spec"].name == trained["spec"].name
        ),
        None,
    )
    if trained_test is None:
        raise RuntimeError(
            f"Model '{trained['spec'].name}' has no independent-test result."
        )

    t = np.asarray(data["t_test"], dtype=float)
    x_truth = np.asarray(data["x_true_test"], dtype=float)
    result = run_sequential_digital_twin(
        t=t,
        x_truth=x_truth,
        x0_mean=np.asarray(trained_test["x_online"][0], dtype=float),
        beta_samples=np.asarray(trained["posterior"]["beta"], dtype=float),
        spec=trained["spec"],
        scaler=trained["scaler"],
        physical=data["physical_test"],
        problem=problem,
        config=config or default_linear_enkf_config(),
        observation_indices=np.array([0], dtype=int),
    )

    ids = result["assimilation_indices"]
    truth = result["truth_at_assimilation"]
    free_run = np.asarray(trained_test["x_online"], dtype=float)[ids]
    closure_truth = np.asarray(data["g_true_test"], dtype=float)[ids]
    result.update(
        {
            "source_filename": str(input_filename),
            "system_name": "linear",
            "display_name": data.get("display_name", "Linear slow-fast system"),
            "model_name": trained["spec"].name,
            "state_labels": data.get("state_labels", [r"$x$"]),
            "free_run_mean": free_run,
            "free_run_rmse": np.sqrt(np.mean((free_run - truth) ** 2, axis=1)),
            "closure_truth_at_assimilation": closure_truth,
            "closure_forecast_rmse": np.sqrt(np.mean((result["closure_forecast_mean"] - closure_truth) ** 2, axis=1)),
            "closure_analysis_rmse": np.sqrt(np.mean((result["closure_analysis_mean"] - closure_truth) ** 2, axis=1)),
        }
    )
    output_path = Path(output_filename)
    with output_path.open("wb") as stream:
        pickle.dump(result, stream)
    print(f"Digital-twin results saved to: {output_path}")
    print(f"Training-selected closure model: {trained['spec'].name}")
    print(f"Mean free-run RMSE: {np.mean(result['free_run_rmse']):.6e}")
    print(f"Mean forecast RMSE: {np.mean(result['forecast_rmse']):.6e}")
    print(f"Mean analysis RMSE: {np.mean(result['analysis_rmse']):.6e}")
    print(f"Mean analysis coverage: {np.mean(result['analysis_coverage']):.3f}")
    print(f"Mean closure-analysis RMSE: {np.mean(result['closure_analysis_rmse']):.6e}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="closure_analysis_results_linear.pkl")
    parser.add_argument("--output", default="digital_twin_results_linear.pkl")
    parser.add_argument("--model", default=None)
    parser.add_argument("--ensemble-size", type=int, default=40)
    parser.add_argument("--assimilation-stride", type=int, default=25)
    parser.add_argument("--observation-noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fixed-closure", action="store_true")
    args = parser.parse_args()
    base = default_linear_enkf_config()
    cfg = EnKFConfig(
        ensemble_size=args.ensemble_size,
        assimilation_stride=args.assimilation_stride,
        observed_state_stride=1,
        observation_noise_std=args.observation_noise,
        initial_state_std=base.initial_state_std,
        process_noise_std=base.process_noise_std,
        parameter_process_noise_std=base.parameter_process_noise_std,
        state_inflation=base.state_inflation,
        parameter_inflation=base.parameter_inflation,
        localization_radius=None,
        estimate_closure_parameters=not args.fixed_closure,
        seed=args.seed,
    )
    run_from_pickle(args.input, args.output, model_name=args.model, config=cfg)


if __name__ == "__main__":
    main()
