"""Run an EnKF digital twin from a saved Lorenz-96 closure-analysis pickle."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from enkf_digital_twin import EnKFConfig, run_sequential_digital_twin
from problem_registry import get_problem


def load_pickle(filename: str | Path) -> dict[str, Any]:
    with open(filename, "rb") as stream:
        return pickle.load(stream)


# def select_training_model(
#     results: list[dict[str, Any]], model_name: str | None = None
# ) -> dict[str, Any]:
#     """Select using training metrics only, preventing test-data leakage."""
#     if not results:
#         raise RuntimeError("The source pickle contains no successful models.")
#     if model_name is not None:
#         match = next(
#             (item for item in results if item["spec"].name == model_name), None
#         )
#         if match is None:
#             available = ", ".join(item["spec"].name for item in results)
#             raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
#         return match

#     def score(item: dict[str, Any]) -> float:
#         value = item["evaluation"]["long_time"].get("spatial_corr_rmse", np.nan)
#         if np.isfinite(value):
#             return float(value)
#         return float(item["evaluation"]["closure"]["nrmse_std"])

#     return min(results, key=score)

def select_training_model(
    results: list[dict[str, Any]],
    x_true: np.ndarray,
    t: np.ndarray,
    t_train: np.ndarray,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Select closure using resolved-state reconstruction RMSE."""

    if not results:
        raise RuntimeError("The source pickle contains no successful models.")

    # Manual override remains available
    if model_name is not None:
        match = next(
            (item for item in results if item["spec"].name == model_name),
            None,
        )
        if match is None:
            available = ", ".join(item["spec"].name for item in results)
            raise ValueError(
                f"Unknown model '{model_name}'. Available: {available}"
            )
        return match

    # Same window used by the closure-results plotting code:
    # start at beginning of training and retain subsequent trajectory.
    tmin = float(t_train[0])
    mask = t >= tmin

    x_ref = np.asarray(x_true, dtype=float)[mask]

    def score(item: dict[str, Any]) -> float:
        x_model = np.asarray(item["x_online"], dtype=float)[mask]

        rmse = np.sqrt(
            np.mean((x_model - x_ref) ** 2)
        )

        if not np.isfinite(rmse):
            return np.inf

        return float(rmse)

    print("\nTraining state-reconstruction RMSE:")
    for item in results:
        print(
            f"  {item['spec'].name:<20s}: "
            f"{score(item):.6e}"
        )

    return min(results, key=score)

def run_from_pickle(
    input_filename: str | Path = "closure_analysis_results_lorenz96.pkl",
    output_filename: str | Path = "digital_twin_results_lorenz96.pkl",
    *,
    model_name: str | None = None,
    config: EnKFConfig | None = None,
) -> Path:
    data = load_pickle(input_filename)
    if data.get("system_name") != "lorenz96":
        raise ValueError("This runner currently supports only two-level Lorenz-96.")

    problem = get_problem("lorenz96")
    
    # trained = select_training_model(data["results"], model_name=model_name)
    trained = select_training_model(
        data["results"],
        x_true=np.asarray(data["x_true"], dtype=float),
        t=np.asarray(data["t"], dtype=float),
        t_train=np.asarray(data["t_train"], dtype=float),
        model_name=model_name,
    )
    
    
    
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

    # The independent trajectory is the synthetic physical asset.  Its hidden
    # fast variables and exact G_k are deliberately withheld from the EnKF.
    t = np.asarray(data["t_test"], dtype=float)
    x_truth = np.asarray(data["x_true_test"], dtype=float)
    x0_mean = np.asarray(trained_test["x_online"][0], dtype=float)
    dt_result = run_sequential_digital_twin(
        t=t,
        x_truth=x_truth,
        x0_mean=x0_mean,
        beta_samples=np.asarray(trained["posterior"]["beta"], dtype=float),
        spec=trained["spec"],
        scaler=trained["scaler"],
        physical=data["physical_test"],
        problem=problem,
        config=config,
    )

    obs_ids = dt_result["assimilation_indices"]
    free_run = np.asarray(trained_test["x_online"], dtype=float)[obs_ids]
    truth = dt_result["truth_at_assimilation"]
    closure_truth = np.asarray(data["g_true_test"], dtype=float)[obs_ids]
    dt_result.update(
        {
            "source_filename": str(input_filename),
            "system_name": "lorenz96",
            "display_name": data.get("display_name", "Two-level Lorenz-96"),
            "model_name": trained["spec"].name,
            "state_labels": data.get("state_labels", []),
            "free_run_mean": free_run,
            "free_run_rmse": np.sqrt(np.mean((free_run - truth) ** 2, axis=1)),
            # Verification only: this hidden closure was not passed to EnKF.
            "closure_truth_at_assimilation": closure_truth,
            "closure_forecast_rmse": np.sqrt(np.mean((dt_result["closure_forecast_mean"] - closure_truth) ** 2, axis=1)),
            "closure_analysis_rmse": np.sqrt(np.mean((dt_result["closure_analysis_mean"] - closure_truth) ** 2, axis=1)),
        }
    )

    output_path = Path(output_filename)
    with output_path.open("wb") as stream:
        pickle.dump(dt_result, stream)
    print(f"Digital-twin results saved to: {output_path}")
    print(f"Training-selected closure model: {trained['spec'].name}")
    print(f"Mean free-run RMSE: {np.mean(dt_result['free_run_rmse']):.6f}")
    print(f"Mean forecast RMSE: {np.mean(dt_result['forecast_rmse']):.6f}")
    print(f"Mean analysis RMSE: {np.mean(dt_result['analysis_rmse']):.6f}")
    print(f"Mean analysis coverage: {np.mean(dt_result['analysis_coverage']):.3f}")
    print(f"Mean closure-analysis RMSE: {np.mean(dt_result['closure_analysis_rmse']):.6f}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="closure_analysis_results_lorenz96.pkl")
    parser.add_argument("--output", default="digital_twin_results_lorenz96.pkl")
    parser.add_argument("--model", default=None)
    parser.add_argument("--ensemble-size", type=int, default=150)
    parser.add_argument("--assimilation-stride", type=int, default=100)
    parser.add_argument("--observed-state-stride", type=int, default=2)
    parser.add_argument("--observation-noise", type=float, default=0.50)
    parser.add_argument("--localization-radius", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    cfg = EnKFConfig(
        ensemble_size=args.ensemble_size,
        assimilation_stride=args.assimilation_stride,
        observed_state_stride=args.observed_state_stride,
        observation_noise_std=args.observation_noise,
        localization_radius=args.localization_radius,
        seed=args.seed,
    )
    run_from_pickle(
        args.input, args.output, model_name=args.model, config=cfg
    )


if __name__ == "__main__":
    main()
