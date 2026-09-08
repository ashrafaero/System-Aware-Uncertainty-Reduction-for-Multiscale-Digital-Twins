"""Publication diagnostics for the Lorenz-63 dynamic-closure EnKF twin."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Restrained, print-friendly palette.
C_TRUTH = "#222222"
C_FORECAST = "#B56A2D"
C_ANALYSIS = "#1F5A94"
C_OBS = "#6F6F6F"
C_SPREAD = "#4E7A6A"
C_FREE = "#8A8A8A"

STYLE = {
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 12.0,
    "axes.labelsize": 13.5,
    "axes.titlesize": 14.0,
    "legend.fontsize": 9.0,
    "xtick.labelsize": 10.8,
    "ytick.labelsize": 10.8,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.35,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
}


def _panel(ax, label):
    ax.text(
        0.012, 0.955, label, transform=ax.transAxes,
        ha="left", va="top", fontsize=12.5, fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.0},
        zorder=10,
    )


def _style(ax):
    ax.grid(True, color="0.88", lw=0.5)
    ax.tick_params(direction="in", top=True, right=True)


# def _legend(ax, *, ncol=1, loc="upper right"):
#     return ax.legend(
#         frameon=False, ncol=ncol, loc=loc,
#         handlelength=2.0, columnspacing=0.9,
#         handletextpad=0.45, borderaxespad=0.35,
#         labelspacing=0.28,
#     )

def _legend(ax, *, ncol=1, loc="best",
            bbox_to_anchor=None):
    return ax.legend(
        frameon=True,
        framealpha=0.72,
        facecolor="white",
        edgecolor="0.75",
        ncol=ncol,
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        handlelength=2.0,
        columnspacing=0.9,
        handletextpad=0.45,
        borderaxespad=0.35,
        labelspacing=0.28,
    )

def rolling_mean(v, w):
    v = np.asarray(v, dtype=float)
    w = max(1, int(w))
    k = np.ones(w)
    return np.convolve(v, k, mode="same") / np.convolve(np.ones_like(v), k, mode="same")


def _coverage_ylim(*series):
    """Zoom coverage panels while leaving a small margin below the observed range."""
    vals = np.concatenate([np.asarray(s, dtype=float).reshape(-1) for s in series])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.90, 1.005
    lower = max(0.75, float(np.nanmin(vals)) - 0.03)
    # Keep 0.95 visible even for near-perfect coverage.
    lower = min(lower, 0.93)
    return lower, 1.005


def _small_error_ylim(*series):
    """Choose a publication-focused RMSE range from DA curves, excluding free-run scale."""
    vals = np.concatenate([np.asarray(s, dtype=float).reshape(-1) for s in series])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    upper = float(np.nanpercentile(vals, 99.5))
    upper = max(upper * 1.25, 0.05)
    return 0.0, upper


def validate_payload(data):
    required = (
        "assimilation_times", "truth_at_assimilation",
        "forecast_mean", "analysis_mean", "analysis_lower", "analysis_upper",
        "forecast_rmse", "analysis_rmse", "analysis_spread", "free_run_rmse",
        "closure_truth_at_assimilation", "closure_forecast_mean", "closure_analysis_mean",
        "closure_forecast_rmse", "closure_analysis_rmse",
        "truth_q_at_assimilation", "latent_q_forecast_rmse", "latent_q_analysis_rmse",
        "forecast_coverage", "analysis_coverage", "beta_mean", "beta_std",
        "qoi_truth", "qoi_analysis_mean",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing DT result keys: {missing}")

    t = np.asarray(data["assimilation_times"])
    n = len(t)
    for key in ("forecast_mean", "analysis_mean", "analysis_lower", "analysis_upper"):
        arr = np.asarray(data[key])
        if arr.shape != (n, 3):
            raise ValueError(f"{key} shape {arr.shape}; expected {(n, 3)}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite {key}.")
    if np.any(np.asarray(data["analysis_lower"]) > np.asarray(data["analysis_mean"])) or np.any(
        np.asarray(data["analysis_mean"]) > np.asarray(data["analysis_upper"])
    ):
        raise ValueError("Analysis interval ordering failure.")

    print("\nLorenz-63 dynamic-DT plotting payload checks: PASS")


def add_observations(ax, data, state_index):
    obs_idx = np.asarray(data["observation_indices"])
    hit = np.flatnonzero(obs_idx == state_index)
    if hit.size:
        ax.scatter(
            data["assimilation_times"],
            np.asarray(data["observations"])[:, hit[0]],
            s=5, facecolors="none", edgecolors=C_OBS,
            linewidths=0.55, alpha=0.55,
            label=rf"Observation, $y_{{{state_index+1},k}}$",
            zorder=2,
        )


def plot_state_assimilation(data):
    t = np.asarray(data["assimilation_times"])
    truth = np.asarray(data["truth_at_assimilation"])

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            4, 1, figsize=(7.2, 9.3), sharex=True,
            constrained_layout=True,
        )

        for j, sym in enumerate(("x", "y")):
            ax = axes[j]
            ax.plot(
                t, truth[:, j], color=C_TRUTH, lw=1.15,
                label=rf"Truth, ${sym}_k$",
            )
            ax.plot(
                t, np.asarray(data["forecast_mean"])[:, j],
                "--", color=C_FORECAST, lw=1.20,
                label=rf"Forecast, ${sym}_k^-$",
            )
            ax.plot(
                t, np.asarray(data["analysis_mean"])[:, j],
                color=C_ANALYSIS, lw=1.30,
                label=rf"Analysis, ${sym}_k^+$",
            )
            ax.fill_between(
                t,
                np.asarray(data["analysis_lower"])[:, j],
                np.asarray(data["analysis_upper"])[:, j],
                color=C_ANALYSIS, alpha=0.10, linewidth=0,
            )
            add_observations(ax, data, j)
            ax.set_ylabel(rf"${sym}$")
            _style(ax); _panel(ax, f"({chr(97 + j)})")
            _legend(ax, ncol=2, loc="best")

        ax = axes[2]
        ax.plot(
            t, data["truth_q_at_assimilation"], color=C_TRUTH, lw=1.15,
            label=r"Hidden truth, $z_k$ (verification)",
        )
        ax.plot(
            t, np.asarray(data["forecast_mean"])[:, 2],
            "--", color=C_FORECAST, lw=1.20,
            label=r"Forecast, $q_k^-$",
        )
        ax.plot(
            t, np.asarray(data["analysis_mean"])[:, 2],
            color=C_ANALYSIS, lw=1.30,
            label=r"Analysis, $q_k^+$",
        )
        ax.fill_between(
            t,
            np.asarray(data["analysis_lower"])[:, 2],
            np.asarray(data["analysis_upper"])[:, 2],
            color=C_ANALYSIS, alpha=0.10, linewidth=0,
        )
        ax.set_ylabel(r"Latent state, $q$")
        _style(ax); _panel(ax, "(c)")
        _legend(ax, ncol=2, loc="best")

        ax = axes[3]
        # Free run is intentionally allowed to clip so the DA curves near zero
        # remain visible.  This is stated directly in the legend.
        ax.plot(
            t, data["free_run_rmse"], color=C_FREE, lw=1.0, alpha=0.2,
            label="Free-run RMSE (clipped)",
        )
        ax.plot(
            t, data["forecast_rmse"], "-", color=C_FORECAST, lw=1.20, alpha=0.2,
            label=r"Forecast RMSE, $\varepsilon_k^-$",
        )
        ax.plot(
            t, data["analysis_rmse"], color=C_ANALYSIS, lw=1.30, alpha=0.2,
            label=r"Analysis RMSE, $\varepsilon_k^+$",
        )
        ax.plot(
            t, data["analysis_spread"], "--", color=C_SPREAD, lw=1.20,
            label=r"Analysis spread, $s_k^+$",
        )

        ymin, ymax = _small_error_ylim(
            data["forecast_rmse"],
            data["analysis_rmse"],
            data["analysis_spread"],
        )
        ax.set_ylim(ymin, ymax)

        ax.set_xlabel(r"Assimilation time, $t_k$")
        ax.set_ylabel(r"State RMSE / spread")
        _style(ax); _panel(ax, "(d)")
        _legend(ax, ncol=2, loc="best")

    return fig, axes


def plot_closure_and_goal(data):
    t = np.asarray(data["assimilation_times"])

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            3, 1, figsize=(7.2, 7.6), sharex=True,
            constrained_layout=True,
        )

        ax = axes[0]
        ax.plot(
            t, np.asarray(data["closure_truth_at_assimilation"])[:, 0],
            color=C_TRUTH, lw=1.15,
            label=r"Truth, $G_k=-x_kz_k$",
        )
        ax.plot(
            t, np.asarray(data["closure_forecast_mean"])[:, 0],
            "--", color=C_FORECAST, lw=1.20,
            label=r"Forecast, $\widehat G_k^-=-x_k^-q_k^-$",
        )
        ax.plot(
            t, np.asarray(data["closure_analysis_mean"])[:, 0],
            color=C_ANALYSIS, lw=1.30,
            label=r"Analysis, $\widehat G_k^+=-x_k^+q_k^+$",
        )
        ax.set_ylabel(r"Closure, $G$")
        _style(ax); _panel(ax, "(a)")
        _legend(ax, ncol=1, loc="best")

        # ax = axes[1]
        # ax.plot(
        #     t, data["closure_forecast_rmse"],
        #     "--", color=C_FORECAST, lw=1.20,
        #     label=r"Forecast closure error",
        # )
        # ax.plot(
        #     t, data["closure_analysis_rmse"],
        #     color=C_ANALYSIS, lw=1.30,
        #     label=r"Analysis closure error",
        # )
        # ax.plot(
        #     t, data["latent_q_analysis_rmse"],
        #     "-.", color=C_SPREAD, lw=1.20,
        #     label=r"Latent-$q$ analysis error",
        # )
        # ax.set_ylabel(r"Absolute error")
        # _style(ax); _panel(ax, "(b)")
        # _legend(ax, ncol=1, loc="best")
        
        ax = axes[1]
        forecast_err = np.asarray(data["closure_forecast_rmse"])
        analysis_err = np.asarray(data["closure_analysis_rmse"])
        q_err = np.asarray(data["latent_q_analysis_rmse"])

        ax.plot(
            t,
            forecast_err,
            "-",
            color=C_FORECAST,
            lw=1.0,
            alpha=0.2,
            label=r"Forecast closure error",
        )

        ax.plot(
            t,
            analysis_err,
            color=C_ANALYSIS,
            lw=1.05,
            alpha=0.2,
            label=r"Analysis closure error",
        )

        ax.plot(
            t,
            q_err,
            "--",
            color=C_SPREAD,
            lw=1.15,
            alpha=1,
            label=r"Latent-$q$ analysis error",
        )

        # Focus the y-axis on the typical small-error region.
        # Large intermittent spikes are intentionally clipped.
        all_err = np.concatenate([
            forecast_err,
            analysis_err,
            q_err,
        ])

        upper = np.nanpercentile(all_err, 50)

        ax.set_ylim(
            0.0,
            1.15 * upper,
        )

        ax.set_ylabel(r"Absolute error")

        _style(ax)
        _panel(ax, "(b)")

        ax.legend(
            loc="best",
            frameon=True,
            framealpha=0.75,
            facecolor="white",
            edgecolor="0.75",
        )



        ax = axes[2]
        ax.plot(
            t, data["qoi_truth"], color=C_TRUTH, lw=1.15,
            label=r"Truth, $Q(X_k)$",
        )
        ax.plot(
            t, data["qoi_analysis_mean"], color=C_ANALYSIS, lw=1.30,
            label=r"$\mathrm{E}[Q\mid D_k]$",
        )
        ax.fill_between(
            t, data["qoi_analysis_lower"], data["qoi_analysis_upper"],
            color=C_ANALYSIS, alpha=0.10, linewidth=0,
        )
        ax.set_xlabel(r"Assimilation time, $t_k$")
        ax.set_ylabel(r"$Q=(x^2+y^2)/4$")
        _style(ax); _panel(ax, "(c)")
        _legend(ax, ncol=1, loc="best")

    return fig, axes


def plot_posterior_diagnostics(data, coverage_window=21):
    t = np.asarray(data["assimilation_times"])
    beta_mean = np.asarray(data["beta_mean"])
    beta_std = np.asarray(data["beta_std"])
    fixed_parameters = not bool(
        data.get("config", {}).get("estimate_closure_parameters", True)
    )

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            2, 1,
            figsize=(7.2, 5.2),
            sharex=True,
            constrained_layout=True,
        )

        ax = axes[0]
        fc = rolling_mean(np.asarray(data["forecast_coverage"]), coverage_window)
        ac = rolling_mean(np.asarray(data["analysis_coverage"]), coverage_window)
        ax.plot(t, fc, "--", color=C_FORECAST, lw=1.20, label=r"Forecast, $x,y$")
        ax.plot(t, ac, color=C_ANALYSIS, lw=1.30, label=r"Analysis, $x,y$")
        ax.axhline(
            0.95, color=C_TRUTH, ls=":", lw=1.0,
            label=r"$p_{\mathrm{nom}}=0.95$",
        )
        ax.set_ylim(*_coverage_ylim(fc, ac))
        ax.set_ylabel(r"Coverage")
        _style(ax); _panel(ax, "(a)")
        _legend(ax, ncol=3, loc="best")

        # ax = axes[1]
        # qcov = rolling_mean(
        #     np.asarray(data["latent_q_analysis_coverage"]),
        #     coverage_window,
        # )
        # ax.plot(
        #     t, qcov, color=C_ANALYSIS, lw=1.30,
        #     label=r"Latent-$q$ analysis coverage",
        # )
        # ax.axhline(
        #     0.95, color=C_TRUTH, ls=":", lw=1.0,
        #     label=r"$p_{\mathrm{nom}}=0.95$",
        # )
        # ax.set_ylim(*_coverage_ylim(qcov))
        # ax.set_ylabel(r"$q$ coverage")
        # _style(ax); _panel(ax, "(b)")
        # _legend(ax, ncol=2, loc="best")

        # ------------------------------------------------------------------
        # (c) Physically important closure coefficients: q and xy terms
        # ------------------------------------------------------------------
        ax = axes[1]

        labels = list(
            data.get(
                "feature_names",
                [f"beta_{j}" for j in range(beta_mean.shape[1])]
            )
        )

        # Locate the two terms of physical interest
        idx_q = labels.index("q")
        idx_xy = labels.index("xy")

        selected_indices = [idx_q, idx_xy]
        selected_labels = [
            r"$\beta_q$",
            r"$\beta_{xy}$",
        ]

        # Professional colors: blue + muted orange
        selected_colors = [
            C_ANALYSIS,
            C_FORECAST,
        ]

        selected_linestyles = [
            "-",
            "--",
        ]

        if fixed_parameters:

            # Fixed-parameter DT:
            # coefficients do not evolve, so show their fixed posterior values.
            for j, lab, color, ls in zip(
                selected_indices,
                selected_labels,
                selected_colors,
                selected_linestyles,
            ):
                ax.plot(
                    t,
                    beta_mean[:, j],
                    color=color,
                    linestyle=ls,
                    linewidth=1.6,
                    label=lab,
                )

                # Initial/offline posterior uncertainty
                ax.fill_between(
                    t,
                    beta_mean[:, j] - 1.96 * beta_std[:, j],
                    beta_mean[:, j] + 1.96 * beta_std[:, j],
                    color=color,
                    alpha=0.10,
                    linewidth=0,
                )

            ax.set_ylabel(r"Closure coefficient, $\beta$")

        else:

            # Adaptive-parameter DT:
            # show coefficient changes normalized by the initial posterior std.
            shift = (
                (beta_mean - beta_mean[0])
                / np.maximum(beta_std[0], np.finfo(float).eps)
            )

            for j, lab, color, ls in zip(
                selected_indices,
                selected_labels,
                selected_colors,
                selected_linestyles,
            ):
                ax.plot(
                    t,
                    shift[:, j],
                    color=color,
                    linestyle=ls,
                    linewidth=1.6,
                    label=lab,
                )

            ax.axhline(
                0.0,
                color="0.35",
                linestyle=":",
                linewidth=1.0,
            )

            ax.set_ylabel(
                r"$(\bar{\beta}_k-\bar{\beta}_0)/s_{\beta,0}$"
            )

        ax.set_xlabel(r"Assimilation time, $t_k$")
        _style(ax)
        _panel(ax, "(b)")

        ax.legend(
            loc="best",
            frameon=True,
            framealpha=0.75,
            facecolor="white",
            edgecolor="0.75",
            ncol=2,
        )


    return fig, axes


def print_metrics(data):
    def m(k):
        return float(np.nanmean(np.asarray(data[k], dtype=float)))

    free = m("free_run_rmse")
    forecast = m("forecast_rmse")
    analysis = m("analysis_rmse")

    print("\n" + "=" * 76)
    print("Lorenz-63 dynamic-closure EnKF digital-twin metrics")
    print("=" * 76)
    print(f"{'Mean free-run state RMSE':<48s}{free:>18.6f}")
    print(f"{'Mean forecast state RMSE':<48s}{forecast:>18.6f}")
    print(f"{'Mean analysis state RMSE':<48s}{analysis:>18.6f}")
    print(f"{'Analysis RMSE reduction vs free run (%)':<48s}{100*(1-analysis/max(free,np.finfo(float).eps)):>18.3f}")
    print(f"{'Analysis RMSE reduction vs forecast (%)':<48s}{100*(1-analysis/max(forecast,np.finfo(float).eps)):>18.3f}")
    print(f"{'Mean latent-q analysis abs. error':<48s}{m('latent_q_analysis_rmse'):>18.6f}")
    print(f"{'Mean closure analysis abs. error':<48s}{m('closure_analysis_rmse'):>18.6f}")
    print(f"{'Mean forecast coverage (x,y)':<48s}{m('forecast_coverage'):>18.6f}")
    print(f"{'Mean analysis coverage (x,y)':<48s}{m('analysis_coverage'):>18.6f}")
    print(f"{'Mean latent-q analysis coverage':<48s}{m('latent_q_analysis_coverage'):>18.6f}")
    print("=" * 76)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?",
        default="digital_twin_results_lorenz63_dynamic.pkl",
    )
    parser.add_argument("--save-prefix", default="lorenz63_dynamic_dt")
    parser.add_argument("--save-dir", default="results/lorenz63/dynamic_dt")
    parser.add_argument("--coverage-window", type=int, default=21)
    args = parser.parse_args()

    with Path(args.input).open("rb") as f:
        data = pickle.load(f)

    validate_payload(data)
    print_metrics(data)

    figs = [
        ("state", plot_state_assimilation(data)[0]),
        ("closure_qoi", plot_closure_and_goal(data)[0]),
        ("posterior", plot_posterior_diagnostics(data, args.coverage_window)[0]),
    ]

    if args.save_prefix:
        out = Path(args.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for suffix, fig in figs:
            fig.savefig(
                out / f"{args.save_prefix}_{suffix}.png",
                dpi=400, bbox_inches="tight",
            )

    plt.show()


if __name__ == "__main__":
    main()
