"""Publication-style diagnostics for the slow-fast linear EnKF digital twin."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

STYLE = {
    "mathtext.fontset": "stix", "font.family": "serif", "font.size": 13,
    "axes.labelsize": 15, "axes.titlesize": 15, "legend.fontsize": 11.5,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.linewidth": 0.8, "savefig.dpi": 300, "savefig.bbox": "tight",
}
TRUTH, FORECAST, ANALYSIS = "#202020", "#E69F00", "#0072B2"
OBSERVATION, SPREAD = "#D55E00", "#009E73"


def _panel(ax, label):
    ax.text(0.012, 0.96, label, transform=ax.transAxes, ha="left", va="top",
            fontweight="bold", fontsize=13, bbox={"facecolor": "white", "edgecolor": "none",
            "alpha": 0.8, "pad": 1.2})


def _style(ax):
    ax.grid(True, color="0.87", linewidth=0.55)
    ax.tick_params(direction="in", top=True, right=True)


def plot_linear_tracking(data):
    """State, closure, and state-error diagnostics."""
    t = np.asarray(data["assimilation_times"])
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.6), sharex=True,
                                 constrained_layout=True)
        ax = axes[0]
        ax.plot(t, np.asarray(data["truth_at_assimilation"])[:, 0], color=TRUTH,
                lw=1.4, label=r"Truth, $x_k$")
        ax.plot(t, np.asarray(data["forecast_mean"])[:, 0], color=FORECAST,
                label=r"Forecast, $x_k^-$")
        ax.plot(t, np.asarray(data["analysis_mean"])[:, 0], color=ANALYSIS,
                label=r"Analysis, $x_k^+$")
        ax.fill_between(t, np.asarray(data["analysis_lower"])[:, 0],
                        np.asarray(data["analysis_upper"])[:, 0], color=ANALYSIS,
                        alpha=0.16, linewidth=0, label=r"$95\%$ interval")
        ax.scatter(t, np.asarray(data["observations"])[:, 0], s=8,
                   color=OBSERVATION, alpha=0.55, label=r"Observation, $y_k$")
        ax.set_ylabel(r"Resolved state, $x$")
        _style(ax); _panel(ax, "(a)")
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.55, 1.0), columnspacing=1.1, handlelength=2.0)

        ax = axes[1]
        ax.plot(t, np.asarray(data["closure_truth_at_assimilation"])[:, 0],
                color=TRUTH, lw=1.4, label=r"Truth, $G_k=b\,y_k$")
        ax.plot(t, np.asarray(data["closure_forecast_mean"])[:, 0],
                color=FORECAST, label=r"Forecast, $\widehat G_k^-$")
        ax.plot(t, np.asarray(data["closure_analysis_mean"])[:, 0],
                color=ANALYSIS, label=r"Analysis, $\widehat G_k^+$")
        ax.set_ylabel(r"Closure, $G$")
        _style(ax); _panel(ax, "(b)")
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.55, 1.0), columnspacing=1.1, handlelength=2.0)

        ax = axes[2]
        ax.plot(t, data["free_run_rmse"], color="0.50", label="Free run")
        ax.plot(t, data["forecast_rmse"], color=FORECAST, label=r"Forecast, $|e_k^-|$")
        ax.plot(t, data["analysis_rmse"], color=ANALYSIS, label=r"Analysis, $|e_k^+|$")
        ax.plot(t, data["analysis_spread"], "--", color=SPREAD,
                label=r"Analysis spread, $s_k^+$")
        ax.set_xlabel(r"Assimilation time, $t_k$")
        ax.set_ylabel(r"State error / spread")
        _style(ax); _panel(ax, "(c)")
        ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.6, 1.0), columnspacing=1.2, handlelength=2.0)
    return fig, axes


def plot_linear_posterior(data):
    """QoI, calibration, and closure-parameter posterior evolution."""
    t = np.asarray(data["assimilation_times"])
    beta_mean, beta_std = np.asarray(data["beta_mean"]), np.asarray(data["beta_std"])
    beta_scale = np.maximum(beta_std[0], np.finfo(float).eps)
    beta_shift = (beta_mean - beta_mean[0]) / beta_scale
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.6), sharex=True,
                                 constrained_layout=True)
        ax = axes[0]
        ax.plot(t, data["qoi_truth"], color=TRUTH, label=r"Truth, $Q(x)$")
        ax.plot(t, data["qoi_analysis_mean"], color=ANALYSIS,
                label=r"$\mathbb{E}[Q\mid\mathcal{D}_k]$")
        ax.fill_between(t, data["qoi_analysis_lower"], data["qoi_analysis_upper"],
                        color=ANALYSIS, alpha=0.16, linewidth=0)
        ax.set_ylabel(r"QoI, $Q=x^2/2$")
        _style(ax); _panel(ax, "(a)"); ax.legend(frameon=False, loc="upper right", ncol=2, columnspacing=1.2, handlelength=2.0)

        ax = axes[1]
        ax.plot(t, data["forecast_coverage"], color=FORECAST, label=r"Forecast, $p_k^-$")
        ax.plot(t, data["analysis_coverage"], color=ANALYSIS, label=r"Analysis, $p_k^+$")
        ax.axhline(0.95, color="0.2", ls="--", lw=1.0,
                   label=r"Nominal, $p_{\rm nom}=0.95$")
        ax.set_ylim(-0.03, 1.03); ax.set_ylabel("Empirical coverage")
        _style(ax); _panel(ax, "(b)"); ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), columnspacing=1.1, handlelength=2.0)

        ax = axes[2]
        for j in range(beta_shift.shape[1]):
            ax.plot(t, beta_shift[:, j], label=rf"$\beta_{{{j}}}$")
        ax.axhline(0.0, color="0.3", lw=0.8)
        ax.set_xlabel(r"Assimilation time, $t_k$")
        ax.set_ylabel(r"$(\bar\beta_k-\bar\beta_0)/s_{\beta,0}$")
        _style(ax); _panel(ax, "(c)")
        ax.legend(frameon=False, ncol=min(4, beta_shift.shape[1]), loc="upper left", bbox_to_anchor=(0.1, 1.05), columnspacing=1.2, handlelength=2.0)
    return fig, axes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="digital_twin_results_linear.pkl")
    parser.add_argument("--save-prefix", default=True)
    args = parser.parse_args()
    with Path(args.input).open("rb") as stream:
        data = pickle.load(stream)
    figures = [plot_linear_tracking(data)[0], plot_linear_posterior(data)[0]]
    if args.save_prefix:
        for suffix, fig in zip(("tracking", "posterior"), figures):
            fig.savefig("results/linear/"+
                f"{args.save_prefix}_{suffix}.png",
                dpi=400,
                bbox_inches="tight",
            )
    
    plt.show()


if __name__ == "__main__":
    main()
