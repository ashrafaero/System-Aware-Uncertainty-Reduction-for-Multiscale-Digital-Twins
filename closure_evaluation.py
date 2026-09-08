"""Shared metrics, result persistence, and plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from typing import Mapping, Sequence, Any
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import norm, wasserstein_distance

from closure_inference import (
    ClosureModelSpec,
    FeatureScaler,
    build_design_matrix,
)

_EPS = np.finfo(float).eps

# Self-contained plotting style: these figures no longer inherit oversized
# fonts or other presentation settings from the calling analysis script.
_JOURNAL_STYLE = {
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 12.0,
    "axes.labelsize": 13.5,
    "legend.fontsize": 11,
    "xtick.labelsize": 10.8,
    "ytick.labelsize": 10.8,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
}

        
# Okabe-Ito-inspired, colorblind-safe colors. A model keeps the same color in
# every diagnostic, making separate legends and repeated text unnecessary.
_MODEL_COLORS = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#6B6B6B",  # gray
    "#8C564B",  # brown
    "#000000",  # black
)

_TRUTH_COLOR = "#202020"
_PREDICTION_COLOR = "#0072B2"


def _journal_plot(func):
    """Apply a stable publication style regardless of caller ``rcParams``."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        with plt.rc_context(_JOURNAL_STYLE):
            return func(*args, **kwargs)
    return wrapped


def _model_colors(n):
    return [_MODEL_COLORS[i % len(_MODEL_COLORS)] for i in range(n)]


@dataclass(frozen=True)
class EvaluationConfig:
    credible_mass: float = 0.95
    forecast_error_threshold: float = 0.30
    forecast_persistence: int = 1
    acf_max_lag: int = 2000
    psd_nperseg: int | None = 2048
    spatial_max_lag: int | None = None
    posterior_chunk_time: int = 200

@dataclass(frozen=True)
class AnalysisConfig:
    observation_noise_std: float = 0.00
    train_start_fraction: float = 1.0 / 6.0
    train_end_fraction: float = 3.0 / 4.0
    derivative_method: str = "savgol"
    savgol_window: int = 31 #101
    savgol_polyorder: int = 3

    mcmc_time_stride: int = 5 #20
    warmup_steps: int = 100
    num_samples: int = 500

    test_seed_offset: int = 100

    forecast_error_threshold: float = 0.30
    forecast_persistence: int = 1
    acf_max_lag: int = 2000
    psd_nperseg: int | None = 2048
    spatial_max_lag: int | None = None
    posterior_chunk_time: int = 200
    
def closure_fidelity(g_reference, g_prediction):
    ref = np.asarray(g_reference, dtype=float)
    pred = np.asarray(g_prediction, dtype=float)
    if ref.shape != pred.shape:
        raise ValueError(
            f"Closure shape mismatch: {ref.shape} vs {pred.shape}"
        )
    ref = ref.reshape(-1)
    pred = pred.reshape(-1)
    err = pred - ref
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    std = float(np.std(ref))
    nrmse = rmse / std if std > _EPS else np.nan
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((ref - ref.mean())**2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > _EPS else np.nan
    corr = (
        float(np.corrcoef(ref, pred)[0, 1])
        if np.std(ref) > _EPS and np.std(pred) > _EPS
        else np.nan
    )
    bias = float(err.mean())
    return {
        "rmse": rmse,
        "nrmse_std": float(nrmse),
        "mae": mae,
        "r2": float(r2),
        "corr": corr,
        "bias": bias,
        "normalized_bias": float(bias / std) if std > _EPS else np.nan,
    }


def posterior_predictive_calibration(
    x,
    g_reference,
    spec,
    scaler,
    beta_samples,
    sigma_samples,
    problem,
    *,
    credible_mass=0.95,
    chunk_time=200,
):
    x = np.asarray(x)
    g_reference = np.asarray(g_reference)
    if x.shape[0] != g_reference.shape[0]:
        raise ValueError("x and g_reference must have the same number of times.")

    beta_samples = np.asarray(beta_samples)
    sigma_samples = np.asarray(sigma_samples).reshape(-1)
    if beta_samples.shape[0] != sigma_samples.size:
        raise ValueError("beta and sigma posterior draws do not match.")

    z = float(norm.ppf(1.0 - (1.0 - credible_mass) / 2.0))
    n_covered = 0
    n_total = 0
    width_sum = 0.0

    for i0 in range(0, x.shape[0], chunk_time):
        i1 = min(i0 + chunk_time, x.shape[0])
        phi = scaler.transform(
            build_design_matrix(x[i0:i1], spec, problem)
        )
        target = g_reference[i0:i1].reshape(-1)
        mu = phi @ beta_samples.T
        lower = np.mean(mu - z * sigma_samples[None, :], axis=1)
        upper = np.mean(mu + z * sigma_samples[None, :], axis=1)
        n_covered += int(np.count_nonzero((target >= lower) & (target <= upper)))
        n_total += target.size
        width_sum += float(np.sum(upper - lower))

    empirical = n_covered / n_total
    return {
        "nominal_coverage": float(credible_mass),
        "empirical_coverage": float(empirical),
        "coverage_error": float(abs(empirical - credible_mass)),
        "mean_interval_width": float(width_sum / n_total),
    }


def normalized_state_error(x_reference, x_prediction):
    x_reference = np.asarray(x_reference, dtype=float)
    x_prediction = np.asarray(x_prediction, dtype=float)
    if x_reference.shape != x_prediction.shape:
        raise ValueError("State arrays must have matching shapes.")
    num = np.sqrt(np.mean((x_prediction - x_reference)**2, axis=1))
    den = np.sqrt(np.mean(x_reference**2, axis=1))
    return num / np.maximum(den, _EPS)


def forecast_horizon(t, error_curve, *, threshold=0.30, persistence=1):
    above = np.asarray(error_curve) >= threshold
    if persistence == 1:
        hits = np.flatnonzero(above)
    else:
        running = np.convolve(
            above.astype(int),
            np.ones(persistence, dtype=int),
            mode="valid",
        )
        hits = np.flatnonzero(running == persistence)
    return (
        float(t[-1] - t[0])
        if hits.size == 0
        else float(t[int(hits[0])] - t[0])
    )


def short_time_dynamics_metrics(
    t, x_reference, x_prediction, *, threshold=0.30, persistence=1
):
    error = normalized_state_error(x_reference, x_prediction)
    return {
        "normalized_error_curve": error,
        "mean_normalized_error": float(np.mean(error)),
        "max_normalized_error": float(np.max(error)),
        "forecast_horizon": forecast_horizon(
            t, error, threshold=threshold, persistence=persistence
        ),
        "forecast_threshold": float(threshold),
    }


def mean_temporal_acf(x, max_lag):
    x = np.asarray(x, dtype=float)
    nt = x.shape[0]
    max_lag = min(int(max_lag), nt - 1)
    z = x - np.mean(x, axis=0, keepdims=True)
    nfft = 1 << int(np.ceil(np.log2(2 * nt - 1)))
    ft = np.fft.rfft(z, n=nfft, axis=0)
    acov = np.fft.irfft(ft * np.conjugate(ft), n=nfft, axis=0)[:max_lag+1]
    acov /= np.arange(nt, nt-max_lag-1, -1, dtype=float)[:, None]
    acf = acov / np.maximum(acov[0], _EPS)[None, :]
    return np.mean(acf, axis=1)


def mean_psd(x, dt, nperseg):
    x = np.asarray(x, dtype=float)
    nperseg = min(nperseg or 2048, x.shape[0])
    f, pxx = welch(
        x, fs=1.0/dt, axis=0, nperseg=nperseg,
        detrend="constant", scaling="density"
    )
    return f, np.mean(pxx, axis=1)


def mean_spatial_correlation(x, max_lag=None):
    """Periodic spatial correlation.

    This is meaningful for homogeneous multi-state systems such as Lorenz-96.
    For one/two-state canonical systems it returns NaNs and is not used as a
    primary diagnostic.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[1]
    if n < 3:
        return np.full(1, np.nan)
    max_lag = n//2 if max_lag is None else min(int(max_lag), n-1)
    z = x - np.mean(x)
    variance = float(np.mean(z**2))
    corr = np.empty(max_lag+1)
    for lag in range(max_lag+1):
        corr[lag] = np.mean(z * np.roll(z, -lag, axis=1)) / max(variance, _EPS)
    return corr


def long_time_statistics(
    x_reference, x_prediction, *, dt,
    acf_max_lag=2000, psd_nperseg=2048, spatial_max_lag=None,
):
    x_reference = np.asarray(x_reference, dtype=float)
    x_prediction = np.asarray(x_prediction, dtype=float)
    if x_reference.shape != x_prediction.shape:
        raise ValueError("State arrays must have matching shapes.")

    mean_ref = np.mean(x_reference, axis=0)
    mean_pred = np.mean(x_prediction, axis=0)
    std_ref = np.std(x_reference, axis=0)
    std_pred = np.std(x_prediction, axis=0)
    typical_std = max(float(np.mean(std_ref)), _EPS)

    wd = [
        wasserstein_distance(x_reference[:, k], x_prediction[:, k])
        for k in range(x_reference.shape[1])
    ]

    acf_ref = mean_temporal_acf(x_reference, acf_max_lag)
    acf_pred = mean_temporal_acf(x_prediction, acf_max_lag)
    f_ref, psd_ref = mean_psd(x_reference, dt, psd_nperseg)
    f_pred, psd_pred = mean_psd(x_prediction, dt, psd_nperseg)

    spatial_ref = mean_spatial_correlation(x_reference, spatial_max_lag)
    spatial_pred = mean_spatial_correlation(x_prediction, spatial_max_lag)
    if np.all(np.isnan(spatial_ref)):
        spatial_rmse = np.nan
    else:
        spatial_rmse = float(
            np.sqrt(np.nanmean((spatial_pred-spatial_ref)**2))
        )

    return {
        "mean_bias_by_std": float(
            np.mean(np.abs(mean_pred-mean_ref))/typical_std
        ),
        "std_relative_error": float(
            np.mean(np.abs(std_pred-std_ref)/np.maximum(std_ref, _EPS))
        ),
        "pdf_wasserstein": float(np.mean(wd)/typical_std),
        "acf_rmse": float(np.sqrt(np.mean((acf_pred-acf_ref)**2))),
        "log_psd_rmse": float(np.sqrt(np.mean(
            (np.log10(np.maximum(psd_pred,_EPS))
             - np.log10(np.maximum(psd_ref,_EPS)))**2
        ))),
        "spatial_corr_rmse": spatial_rmse,
        "acf_reference": acf_ref,
        "acf_prediction": acf_pred,
        "psd_frequency": f_ref,
        "psd_reference": psd_ref,
        "psd_prediction": psd_pred,
        "spatial_reference": spatial_ref,
        "spatial_prediction": spatial_pred,
    }


def evaluate_closure_model(
    *,
    t, dt, x_reference, x_prediction,
    g_reference, g_prediction,
    x_for_posterior, spec, scaler, posterior,
    problem, config=None,
):
    cfg = config or EvaluationConfig()
    return {
        "closure": closure_fidelity(g_reference, g_prediction),
        "calibration": posterior_predictive_calibration(
            x_for_posterior, g_reference, spec, scaler,
            posterior["beta"], posterior["sigma"], problem,
            credible_mass=cfg.credible_mass,
            chunk_time=cfg.posterior_chunk_time,
        ),
        "short_time": short_time_dynamics_metrics(
            t, x_reference, x_prediction,
            threshold=cfg.forecast_error_threshold,
            persistence=cfg.forecast_persistence,
        ),
        "long_time": long_time_statistics(
            x_reference, x_prediction, dt=dt,
            acf_max_lag=cfg.acf_max_lag,
            psd_nperseg=cfg.psd_nperseg,
            spatial_max_lag=cfg.spatial_max_lag,
        ),
    }


def print_metric_table(results, *, heading="Closure evaluation"):
    print(f"\n{heading}")
    print(
        f"{'model':<20}{'NRMSE_G':>10}{'R2_G':>10}{'corr_G':>10}"
        f"{'coverage':>11}{'T_forecast':>12}{'PDF_WD':>10}"
        f"{'ACF_RMSE':>11}{'PSD_RMSE':>11}"
    )
    print("-"*105)
    for item in results:
        e = item["evaluation"]
        print(
            f"{item['spec'].name:<20}"
            f"{e['closure']['nrmse_std']:>10.3f}"
            f"{e['closure']['r2']:>10.3f}"
            f"{e['closure']['corr']:>10.3f}"
            f"{e['calibration']['empirical_coverage']:>11.3f}"
            f"{e['short_time']['forecast_horizon']:>12.3f}"
            f"{e['long_time']['pdf_wasserstein']:>10.3f}"
            f"{e['long_time']['acf_rmse']:>11.3f}"
            f"{e['long_time']['log_psd_rmse']:>11.3f}"
        )


def save_results(filename, **kwargs):
    with open(filename, "wb") as f:
        pickle.dump(kwargs, f)
    print(f"Results saved to: {filename}")


def load_results(filename):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    print(f"Results loaded from: {filename}")
    return data


def _panel_label(ax, label):
    """Add a compact journal-style panel identifier without a title."""
    ax.text(
        0.012, 0.96, label, transform=ax.transAxes,
        ha="left", va="top", fontsize=10.5, fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
    )


def _style_axis(ax, *, grid_axis="y"):
    ax.grid(True, axis=grid_axis, color="0.86", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", top=True, right=True)


def _bar(ax, names, values, ylabel, panel_label=None, colors=None):
    """Draw one metric bar panel on a caller-owned axis."""
    xpos = np.arange(len(names))
    colors = _model_colors(len(names)) if colors is None else colors
    ax.bar(
        xpos, values, width=0.68, color=colors,
        edgecolor="0.20", linewidth=0.45,
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    _style_axis(ax)
    if panel_label is not None:
        _panel_label(ax, panel_label)


@_journal_plot
def plot_apriori_metrics(results, prefix=""):
    names = [r["spec"].name for r in results]
    ev = [r["evaluation"] for r in results]
    fig, axes = plt.subplots(
        3, 1, figsize=(7.0, 7.8), sharex=True, constrained_layout=True
    )
    _bar(
        axes[0], names, [e["closure"]["nrmse_std"] for e in ev],
        r"$\mathrm{NRMSE}(G,\widehat{G})$", "(a)",
    )
    _bar(
        axes[1], names, [e["closure"]["r2"] for e in ev],
        r"$R^2(G,\widehat{G})$", "(b)",
    )
    _bar(
        axes[2], names, [e["closure"]["corr"] for e in ev],
        r"$\rho(G,\widehat{G})$", "(c)",
    )
    axes[2].set_xlabel(r"Closure model, $\mathcal{M}_m$")
    return fig, axes


@_journal_plot
def plot_calibration_metrics(results, prefix=""):
    names = [r["spec"].name for r in results]
    ev = [r["evaluation"] for r in results]
    coverage = [e["calibration"]["empirical_coverage"] for e in ev]
    nominal = ev[0]["calibration"]["nominal_coverage"]
    fig, ax = plt.subplots(figsize=(7.0, 3.7), constrained_layout=True)
    xpos = np.arange(len(names))
    ax.bar(
        xpos, coverage, width=0.68, color=_model_colors(len(names)),
        edgecolor="0.20", linewidth=0.45,
    )
    ax.axhline(
        nominal, color="0.15", lw=1.1, ls="--",
        label=rf"$p_{{\mathrm{{nom}}}}={nominal:.2f}$",
    )
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylim(0,1.05)
    ax.set_xlabel(r"Closure model, $\mathcal{M}_m$")
    ax.set_ylabel(r"Empirical coverage, $p_{\mathrm{emp}}$")
    _style_axis(ax)
    ax.legend(frameon=False, loc="lower right")
    return fig, ax


@_journal_plot
def plot_short_time_metrics(t, results, prefix=""):
    names = [r["spec"].name for r in results]
    ev = [r["evaluation"] for r in results]
    fig, axes = plt.subplots(
        2, 1, figsize=(7.0, 5.9), constrained_layout=True
    )
    ax = axes[0]
    colors = _model_colors(len(names))
    for name, e, color in zip(names, ev, colors):
        ax.plot(t, e["short_time"]["normalized_error_curve"], color=color, label=name)
    threshold = ev[0]["short_time"]["forecast_threshold"]
    ax.axhline(
        threshold, color="k", lw=1.0, ls="--",
        label=rf"Threshold, $\varepsilon_{{\mathrm{{th}}}}={threshold:g}$",
    )
    ax.set_xlabel(r"Time, $t$")
    ax.set_ylabel(r"Normalized state error, $\varepsilon_x(t)$")
    _style_axis(ax, grid_axis="both")
    _panel_label(ax, "(a)")
    ax.legend(ncol=2, frameon=False, loc="upper center")

    _bar(
        axes[1], names,
        [e["short_time"]["forecast_horizon"] for e in ev],
        r"Forecast horizon, $T_f$", "(b)", colors=colors,
    )
    axes[1].set_xlabel(r"Closure model, $\mathcal{M}_m$")
    return fig, axes


@_journal_plot
def plot_long_time_metrics(results, prefix=""):
    names = [r["spec"].name for r in results]
    ev = [r["evaluation"] for r in results]
    info = [
        ("mean_bias_by_std", r"Mean error, $E_\mu$"),
        ("std_relative_error", r"Std. error, $E_\sigma$"),
        ("pdf_wasserstein", r"PDF distance, $W_1/\bar{\sigma}_X$"),
        ("acf_rmse", r"ACF error, $E_R$"),
        ("log_psd_rmse", r"Log-PSD error, $E_S$"),
    ]
    # Spatial correlation only if meaningful/non-NaN.
    if not all(np.isnan(e["long_time"]["spatial_corr_rmse"]) for e in ev):
        info.append(("spatial_corr_rmse", r"Spatial-corr. error, $E_C$"))

    figures = []
    for start in range(0, len(info), 3):
        group = info[start:start + 3]
        fig, axes = plt.subplots(
            len(group), 1, figsize=(7.0, 2.45 * len(group) + 0.45),
            sharex=True, constrained_layout=True, squeeze=False,
        )
        axes = axes[:, 0]
        for j, (ax, (key, ylabel)) in enumerate(zip(axes, group)):
            _bar(
                ax, names, [e["long_time"][key] for e in ev],
                ylabel, f"({chr(97 + start + j)})",
            )
        axes[-1].set_xlabel(r"Closure model, $\mathcal{M}_m$")
        figures.append((fig, axes))
    return figures


@_journal_plot
def plot_model_statistical_curves(
    x_reference, model_result, *, dt, state_labels=None, prefix=""
):
    name = model_result["spec"].name
    x_prediction = model_result["x_online"]
    stats = model_result["evaluation"]["long_time"]

    fig, axes = plt.subplots(
        3, 1, figsize=(7.0, 7.9), constrained_layout=True
    )
    ax = axes[0]
    ax.hist(x_reference.reshape(-1), bins=80, density=True,
            histtype="step", linewidth=1.5, color=_TRUTH_COLOR,
            label=r"Truth, $X_k$")
    ax.hist(x_prediction.reshape(-1), bins=80, density=True,
            histtype="step", linewidth=1.5, color=_PREDICTION_COLOR,
            label=rf"{name}, $\widehat{{X}}_k$")
    ax.set_xlabel(r"Resolved state, $X$")
    ax.set_ylabel(r"Density, $p_X(X)$")
    _style_axis(ax, grid_axis="both"); _panel_label(ax, "(a)")
    ax.legend(frameon=False, loc="upper right")

    lags = np.arange(stats["acf_reference"].size)*dt
    ax = axes[1]
    ax.plot(lags, stats["acf_reference"], color=_TRUTH_COLOR, label=r"Truth, $X_k$")
    ax.plot(
        lags, stats["acf_prediction"], "--", color=_PREDICTION_COLOR,
        label=rf"{name}, $\widehat{{X}}_k$",
    )
    ax.set_xlabel(r"Time lag, $\tau$")
    ax.set_ylabel(r"Mean temporal ACF, $R(\tau)$")
    _style_axis(ax, grid_axis="both"); _panel_label(ax, "(b)")

    ax = axes[2]
    ax.semilogy(
        stats["psd_frequency"], stats["psd_reference"],
        color=_TRUTH_COLOR, label=r"Truth, $X_k$",
    )
    ax.semilogy(
        stats["psd_frequency"], stats["psd_prediction"], "--",
        color=_PREDICTION_COLOR, label=rf"{name}, $\widehat{{X}}_k$",
    )
    ax.set_xlabel(r"Frequency, $f$")
    ax.set_ylabel(r"Mean PSD, $S_X(f)$")
    _style_axis(ax, grid_axis="both"); _panel_label(ax, "(c)")

    figures = [(fig, axes)]

    if not np.all(np.isnan(stats["spatial_reference"])):
        lag = np.arange(stats["spatial_reference"].size)
        fig_spatial, ax = plt.subplots(
            figsize=(7.0, 3.8), constrained_layout=True
        )
        ax.plot(
            lag, stats["spatial_reference"], marker="o",
            color=_TRUTH_COLOR, label=r"Truth, $X_k$",
        )
        ax.plot(
            lag, stats["spatial_prediction"], "--", marker="o",
            color=_PREDICTION_COLOR, label=rf"{name}, $\widehat{{X}}_k$",
        )
        ax.set_xlabel(r"Spatial lag, $\ell$")
        ax.set_ylabel(r"Periodic spatial correlation, $C(\ell)$")
        _style_axis(ax, grid_axis="both")
        ax.legend(frameon=False)
        figures.append((fig_spatial, np.asarray([ax])))
    return figures
