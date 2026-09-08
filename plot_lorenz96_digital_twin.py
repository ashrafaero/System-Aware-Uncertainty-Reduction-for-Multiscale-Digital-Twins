"""Publication-style diagnostics for the Lorenz-96 EnKF digital twin."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


STYLE = {
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 12.0,
    "axes.labelsize": 13.5,
    "axes.titlesize": 14.0,
    "legend.fontsize": 11,
    "xtick.labelsize": 10.8,
    "ytick.labelsize": 10.8,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.3,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
}

C_TRUTH = "#222222"
C_FORECAST = "#B56A2D"
C_ANALYSIS = "#1F5A94"
C_OBS = "#6F6F6F"
C_SPREAD = "#4E7A6A"
C_FREE = "#888888"


def _panel(ax, label: str) -> None:
    ax.text(
        0.012, 0.96, label, transform=ax.transAxes, ha="left", va="top",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
    )


def _style(ax) -> None:
    ax.grid(True, color="0.88", lw=0.5)
    ax.tick_params(direction="in", top=True, right=True)


def _legend(ax, ncol=1, loc="best", bbox_to_anchor=None):
    return ax.legend(
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        frameon=True,
        framealpha=0.75,
        facecolor="white",
        edgecolor="0.75",
        handlelength=2.0,
        handletextpad=0.45,
        columnspacing=0.9,
        labelspacing=0.28,
        borderaxespad=0.35,
    )


def _small_error_ylim(*series, percentile=99.0):
    vals = np.concatenate([np.asarray(v, dtype=float).ravel() for v in series])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    upper = max(1.2 * float(np.nanpercentile(vals, percentile)), 0.05)
    return 0.0, upper


def _set_time_window(axes, t, tmin=None, tmax=None):
    """Apply a common publication time window without changing the data."""
    if tmin is None:
        tmin = float(t[0])
    if tmax is None:
        tmax = float(t[-1])
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    for ax in axes:
        ax.set_xlim(tmin, tmax)


def plot_digital_twin_summary(data: dict, state_index: int = 0, tmin=None, tmax=None):
    t = np.asarray(data["assimilation_times"])
    truth = np.asarray(data["truth_at_assimilation"])
    obs_indices = np.asarray(data["observation_indices"])
    state_index = int(state_index)
    if not 0 <= state_index < truth.shape[1]:
        raise ValueError(f"state_index must lie in [0, {truth.shape[1] - 1}].")

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            3, 1, figsize=(7.2, 7.8), constrained_layout=True
        )
        ax = axes[0]
        ax.plot(t, truth[:, state_index], color=C_TRUTH, lw=1.35, label=rf"Truth, $X_{{{state_index+1},k}}$")
        ax.plot(t, data["forecast_mean"][:, state_index], color=C_FORECAST, lw=1.0, label=rf"Forecast, $X_{{{state_index+1},k}}^-$")
        ax.plot(t, data["analysis_mean"][:, state_index], color=C_ANALYSIS, lw=1.15, label=rf"Analysis, $X_{{{state_index+1},k}}^+$")
        ax.fill_between(
            t, data["analysis_lower"][:, state_index], data["analysis_upper"][:, state_index],
            color=C_ANALYSIS, alpha=0.16, linewidth=0,
            label=r"$95\%$ analysis interval",
        )
        if state_index in obs_indices:
            column = int(np.flatnonzero(obs_indices == state_index)[0])
            ax.scatter(
                t, data["observations"][:, column],
                s=6, facecolors="none", edgecolors=C_OBS,
                linewidths=0.55, alpha=0.55,
                label=rf"Observation, $y_{{{state_index+1},k}}$",
            )
        ax.set_ylabel(rf"State, $X_{{{state_index+1}}}$")
        _style(ax); _panel(ax, "(a)")
        _legend(ax, ncol=2, loc="best")

        ax = axes[1]
        ax.plot(t, data["free_run_rmse"], color=C_FREE, label=r"Free-run RMSE")
        ax.plot(t, data["forecast_rmse"], color=C_FORECAST, label=r"Forecast, $X_k^{-}$")
        ax.plot(t, data["analysis_rmse"], color=C_ANALYSIS, label=r"Analysis, $X_k^{+}$")
        ax.plot(t, data["analysis_spread"], "--", color=C_SPREAD, label=r"Analysis spread")
        ax.set_ylim(*_small_error_ylim(
            data["forecast_rmse"],
            data["analysis_rmse"],
            data["analysis_spread"],
            percentile=99.0,
        ))
        ax.set_ylabel(r"State RMSE / spread")
        _style(ax); _panel(ax, "(b)")
        _legend(ax, ncol=2, loc="best")

        ax = axes[2]
        q_truth = np.asarray(data["qoi_truth"])
        q_mean = np.asarray(data["qoi_analysis_mean"])
        q_std = np.asarray(data["qoi_analysis_std"])
        ax.plot(t, q_truth, color=C_TRUTH, label=r"Truth, $Q(X)$")
        ax.plot(t, q_mean, color=C_ANALYSIS, label=r"Analysis, $\widehat{Q}_k$")
        ax.fill_between(t, data["qoi_analysis_lower"], data["qoi_analysis_upper"], color=C_ANALYSIS, alpha=0.16, linewidth=0)
        ax.set_xlabel(r"Assimilation time, $t_k$")
        ax.set_ylabel(r"$Q=\frac{1}{2K}\sum_{j=1}^{K}X_j^2$")
        _style(ax); _panel(ax, "(c)")
        _legend(ax, ncol=1, loc="best")
    _set_time_window(axes, t, tmin, tmax)
    return fig, axes


def plot_closure_tracking(data: dict, state_index: int = 0, tmin=None, tmax=None):
    """Verification of inferred closure tracking; truth is evaluation-only."""
    t = np.asarray(data["assimilation_times"])
    truth = np.asarray(data["closure_truth_at_assimilation"])
    k = int(state_index)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.4), constrained_layout=True)
        ax = axes[0]
        ax.plot(t, truth[:, k], color=C_TRUTH, lw=1.3, label=rf"Truth, $G_{{{k+1}}}$")
        ax.plot(t, data["closure_forecast_mean"][:, k], color=C_FORECAST, label=rf"Forecast, $\widehat{{G}}_{{{k+1}}}^-$")
        ax.plot(t, data["closure_analysis_mean"][:, k], color=C_ANALYSIS, label=rf"Analysis, $\widehat{{G}}_{{{k+1}}}^+$")
        ax.set_ylabel(rf"Closure, $G_{{{k+1}}}$")
        _style(ax); _panel(ax, "(a)")
        _legend(ax, ncol=2, loc="upper right")

        ax = axes[1]
        forecast_err = np.asarray(data["closure_forecast_rmse"])
        analysis_err = np.asarray(data["closure_analysis_rmse"])
        
        ax.plot(t, forecast_err,  color=C_FORECAST, lw=1.2, alpha=1,
                label=r"Forecast")
        ax.plot(t, analysis_err, "--", color=C_ANALYSIS, lw=1.3, alpha=1,
                label=r"Analysis")
        
        
        
        upper = np.nanpercentile(
            np.concatenate([forecast_err.ravel(), analysis_err.ravel()]), 97.5
        )
        if np.isfinite(upper) and upper > 0:
            ax.set_ylim(0.2, 1.15 * upper)
        ax.set_xlabel(r"Assimilation time, $t_k$")
        ax.set_ylabel(r"Closure RMSE")
        _style(ax); _panel(ax, "(b)")
        _legend(ax, ncol=1, loc="lower right")
    _set_time_window(axes, t, tmin, tmax)
    return fig, axes


def plot_digital_state_fields(data: dict, tmin=None, tmax=None):
    t = np.asarray(data["assimilation_times"])
    truth = np.asarray(data["truth_at_assimilation"])
    analysis = np.asarray(data["analysis_mean"])
    raw_error = analysis - truth
    # Normalize by the RMS magnitude of the truth state
    state_rms = np.sqrt(np.nanmean(truth**2))

    if state_rms > np.finfo(float).eps:
        error = raw_error / state_rms
    else:
        error = np.zeros_like(raw_error)



    k = np.arange(1, truth.shape[1] + 1)
    # state_limit = np.percentile(np.abs(np.concatenate([truth.ravel(), analysis.ravel()])), 99.0)
    # error_limit = max(np.percentile(np.abs(error), 99.0), np.finfo(float).eps)
    # state_levels = np.linspace(-state_limit, state_limit, 81)
    # error_levels = np.linspace(-error_limit, error_limit, 81)
    
    # ------------------------------------------------------------------
    # State limits: actual extrema rounded outward to nearest integer
    # ------------------------------------------------------------------
    state_min = min(np.nanmin(truth), np.nanmin(analysis))
    state_max = max(np.nanmax(truth), np.nanmax(analysis))

    state_lower = np.floor(state_min)
    state_upper = np.ceil(state_max)

    # Symmetric around zero
    state_limit = max(abs(state_lower), abs(state_upper))

    # ------------------------------------------------------------------
    # Error limits: actual extrema rounded outward to nearest integer
    # ------------------------------------------------------------------
    error_min = np.nanmin(error)
    error_max = np.nanmax(error)

    error_lower = np.floor(10.0 * error_min) / 10.0
    error_upper = np.ceil(10.0 * error_max) / 10.0

    # Symmetric around zero
    error_limit = max(abs(error_lower), abs(error_upper))

    # Avoid zero-width normalization in the unlikely case of zero error
    error_limit = max(error_limit, np.finfo(float).eps)

    # Contour levels
    state_levels = np.linspace(-state_limit, state_limit, 81)
    error_levels = np.linspace(-error_limit, error_limit, 81)
    
    # error_limit = 1.0
    # error_levels = np.linspace(-1.0, 1.0, 81)


    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            3, 1, figsize=(7.2, 7.8), sharex=True, sharey=True,
            constrained_layout=True,
        )
        m0 = axes[0].contourf(t, k, truth.T, levels=state_levels, cmap="viridis", extend="both")
        axes[1].contourf(t, k, analysis.T, levels=state_levels, cmap="viridis", extend="both")
        norm_error = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
        m2 = axes[2].contourf(t, k, error.T, levels=error_levels, cmap="BrBG", norm=norm_error, extend="both")
        for ax, label in zip(axes, (r"(a) Truth, $X_j(t_k)$", r"(b) Analysis, $X_j^+(t_k)$", r"(c) Error, $e_j^+(t_k)$")):
            ax.set_ylabel(r"State index, $j$")
            _panel(ax, label)
            ax.tick_params(direction="in", top=True, right=True)
        axes[-1].set_xlabel(r"Assimilation time, $t_k$")
        # cb_state = fig.colorbar(m0, ax=axes[:2], pad=0.015, aspect=35)
        # cb_state.set_label(r"Resolved state")
        # cb_error = fig.colorbar(m2, ax=axes[2], pad=0.015, aspect=18, format="%.2f")
        # cb_error.set_label(r"Analysis error, $e_j^{+}=X_j^{+}-X_j$")
        cb_state = fig.colorbar(
            m0,
            ax=axes[:2],
            pad=0.015,
            aspect=35,
            ticks=np.linspace(-state_limit, state_limit, 5),
            format="%.2f",
        )
        cb_state.set_label(r"Resolved state")

        cb_error = fig.colorbar(
            m2,
            ax=axes[2],
            pad=0.015,
            aspect=18,
            ticks=np.linspace(-error_limit, error_limit, 5),
            format="%.2f",
        )
        # cb_error.set_label(r"Analysis error, $e_j^{+}=X_j^{+}-X_j$")
        cb_error.set_label(r"Normalized analysis error, $e_j^{+}/X_{\rm rms}$", fontsize=11)
    
    
    _set_time_window(axes, t, tmin, tmax)
    return fig, axes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="digital_twin_results_lorenz96.pkl")
    parser.add_argument("--state", type=int, default=0, help="Zero-based state index.")
    parser.add_argument("--save-prefix", default="lorenz96_dt")
    parser.add_argument(
        "--tmin", type=float, default=None,
        help="First assimilation time displayed."
    )
    parser.add_argument(
        "--tmax", type=float, default=None,
        help="Last assimilation time displayed."
    )
    parser.add_argument("--save-dir", default="results/lorenz96/digital_twin")
    args = parser.parse_args()
    with Path(args.input).open("rb") as stream:
        data = pickle.load(stream)
    figures = [
        plot_digital_twin_summary(
            data, state_index=args.state, tmin=args.tmin, tmax=args.tmax
        )[0],
        plot_digital_state_fields(
            data, tmin=args.tmin, tmax=args.tmax
        )[0],
        plot_closure_tracking(
            data, state_index=args.state, tmin=args.tmin, tmax=args.tmax
        )[0],
    ]
    if args.save_prefix:
        out = Path(args.save_dir)
        out.mkdir(parents=True, exist_ok=True)
        for suffix, fig in zip(("summary", "fields", "closure"), figures):
            fig.savefig(
                out / f"{args.save_prefix}_{suffix}.png",
                dpi=400,
                bbox_inches="tight",
            )            
    plt.show()


if __name__ == "__main__":
    main()
