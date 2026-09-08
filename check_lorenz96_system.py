# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon Sep  7 20:31:03 2026

# @author: akassem
# """

"""Independent diagnostic checks for the two-level Lorenz-96 system."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from lorenz96_system import (
    default_config,
    generate_truth,
    recover_closure_from_x,
    default_model_specs,
    build_design_matrix,
)


# ============================================================================
# Utilities
# ============================================================================

def regression_metrics(y_true, y_pred):
    """Basic deterministic regression metrics."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    error = y_pred - y_true

    rmse = np.sqrt(np.mean(error**2))
    std = np.std(y_true)
    nrmse = rmse / std if std > 0.0 else np.nan

    ss_res = np.sum(error**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else np.nan

    corr = (
        np.corrcoef(y_true, y_pred)[0, 1]
        if np.std(y_pred) > 0.0 and std > 0.0
        else np.nan
    )

    return {
        "rmse": rmse,
        "nrmse": nrmse,
        "r2": r2,
        "corr": corr,
    }


def print_metrics(name, metrics):
    print(f"\n{name}")
    print("-" * len(name))
    print(f"RMSE:        {metrics['rmse']:.6f}")
    print(f"NRMSE(std):  {metrics['nrmse']:.6f}")
    print(f"R^2:         {metrics['r2']:.6f}")
    print(f"Correlation: {metrics['corr']:.6f}")


# ============================================================================
# 1. Solve true two-level Lorenz-96 system
# ============================================================================

cfg = default_config()

print("\n" + "=" * 72)
print("TWO-LEVEL LORENZ-96: INDEPENDENT SYSTEM CHECK")
print("=" * 72)

print("\nPhysical parameters")
print("-------------------")
print(f"K = {cfg.n_slow}")
print(f"J = {cfg.n_fast}")
print(f"F = {cfg.forcing}")
print(f"h = {cfg.h}")
print(f"b = {cfg.b}")
print(f"c = {cfg.c}")
print(f"hc/b = {cfg.coupling}")
print(f"dt = {cfg.dt}")
print(f"t_end = {cfg.t_end}")
print(f"seed = {cfg.seed}")

print("\nSolving the full two-level system...")

t, x, y, g_true = generate_truth(cfg)

print("Simulation completed.")


# ============================================================================
# 2. Numerical sanity checks
# ============================================================================

print("\n" + "=" * 72)
print("1. NUMERICAL / PHYSICAL SANITY CHECKS")
print("=" * 72)

print(f"\nTime shape:       {t.shape}")
print(f"Slow-state shape: {x.shape}")
print(f"Fast-state shape: {y.shape}")
print(f"Closure shape:    {g_true.shape}")

finite_x = np.all(np.isfinite(x))
finite_y = np.all(np.isfinite(y))
finite_g = np.all(np.isfinite(g_true))

print("\nFinite-value check")
print("------------------")
print(f"X finite: {finite_x}")
print(f"Y finite: {finite_y}")
print(f"G finite: {finite_g}")

if finite_x and finite_y and finite_g:
    print("PASS: No NaN or Inf values detected.")
else:
    print("WARNING: Numerical instability detected.")


print("\nSlow-state statistics")
print("---------------------")
print(f"Minimum: {np.min(x): .6f}")
print(f"Maximum: {np.max(x): .6f}")
print(f"Mean:    {np.mean(x): .6f}")
print(f"Std:     {np.std(x): .6f}")


print("\nFast-state statistics")
print("---------------------")
print(f"Minimum: {np.min(y): .6f}")
print(f"Maximum: {np.max(y): .6f}")
print(f"Mean:    {np.mean(y): .6f}")
print(f"Std:     {np.std(y): .6f}")


print("\nTrue closure statistics")
print("-----------------------")
print(f"Minimum: {np.min(g_true): .6f}")
print(f"Maximum: {np.max(g_true): .6f}")
print(f"Mean:    {np.mean(g_true): .6f}")
print(f"Std:     {np.std(g_true): .6f}")


# ============================================================================
# 3. Fast-variable synchronization check
# ============================================================================

# Spread across j for each slow-state location k.
fast_spread = np.mean(np.std(y, axis=1), axis=1)

initial_fast_spread = fast_spread[0]
final_fast_spread = fast_spread[-1]
mean_fast_spread = np.mean(fast_spread)

print("\nFast-variable spread")
print("--------------------")
print(f"Initial: {initial_fast_spread:.6e}")
print(f"Final:   {final_fast_spread:.6e}")
print(f"Mean:    {mean_fast_spread:.6e}")

if mean_fast_spread < 1.0e-5:
    print("WARNING: Fast variables appear nearly synchronized.")
else:
    print("PASS: Fast variables contain non-trivial independent dynamics.")


# ============================================================================
# 4. Exact closure identity check
# ============================================================================

# generate_truth() defines G_k = sum_j Y_{j,k}.
g_direct = np.sum(y, axis=1)

closure_identity_error = np.max(np.abs(g_true - g_direct))

print("\nClosure identity check")
print("----------------------")
print(f"max |G_true - sum_j(Y)| = {closure_identity_error:.6e}")

if closure_identity_error < 1.0e-12:
    print("PASS: Exact closure definition is internally consistent.")
else:
    print("WARNING: Closure construction is inconsistent.")


# ============================================================================
# 5. Can G be recovered from X?
# ============================================================================

print("\n" + "=" * 72)
print("2. CLOSURE RECOVERY FROM RESOLVED STATE")
print("=" * 72)

# With a noise-free synthetic trajectory, gradient differentiation is a useful
# baseline because it does not deliberately smooth the signal.
g_gradient = recover_closure_from_x(
    x,
    cfg,
    derivative="gradient",
)

g_savgol = recover_closure_from_x(
    x,
    cfg,
    derivative="savgol",
    sg_window=31,
    sg_polyorder=3,
)

gradient_metrics = regression_metrics(g_true, g_gradient)
savgol_metrics = regression_metrics(g_true, g_savgol)

print_metrics("Gradient recovery: G_recovered vs G_true", gradient_metrics)
print_metrics("Savitzky-Golay recovery: G_recovered vs G_true", savgol_metrics)


# ============================================================================
# 6. Is X -> G inference actually possible?
# ============================================================================
#
# This is the important pre-MCMC check.
#
# For every closure family, perform ordinary least squares using the EXACT
# hidden closure G_true.
#
# If even least squares cannot predict G_true from the chosen X features,
# MCMC cannot be expected to magically produce an almost exact deterministic
# closure using the same feature library.
# ============================================================================

print("\n" + "=" * 72)
print("3. X -> G IDENTIFIABILITY / INFERENCE CHECK")
print("=" * 72)

# Ignore early transient.
i0 = int(0.15 * len(t))

x_use = x[i0:]
g_use = g_true[i0:]

# Chronological split:
# first 70% for fitting, remaining 30% for out-of-sample testing.
n = len(x_use)
n_train = int(0.70 * n)

x_train = x_use[:n_train]
g_train = g_use[:n_train]

x_test = x_use[n_train:]
g_test = g_use[n_train:]

# Reduce cost while preserving many samples.
stride = 20

x_train_fit = x_train[::stride]
g_train_fit = g_train[::stride]

print(f"\nTraining snapshots used: {len(x_train_fit)}")
print(f"Testing snapshots:       {len(x_test)}")

identifiability_results = []

for spec in default_model_specs():

    phi_train = build_design_matrix(x_train_fit, spec)
    target_train = g_train_fit.reshape(-1)

    # Feature standardization.
    mean = phi_train.mean(axis=0)
    scale = phi_train.std(axis=0)

    # Intercept should not be centered/scaled.
    mean[0] = 0.0
    scale[0] = 1.0

    scale[scale < 1.0e-8] = 1.0

    phi_train_scaled = (phi_train - mean) / scale

    # Deterministic best-fit model.
    beta, *_ = np.linalg.lstsq(
        phi_train_scaled,
        target_train,
        rcond=None,
    )

    # Independent test trajectory segment.
    phi_test = build_design_matrix(x_test, spec)
    phi_test_scaled = (phi_test - mean) / scale

    g_pred_test = phi_test_scaled @ beta

    metrics = regression_metrics(
        g_test.reshape(-1),
        g_pred_test,
    )

    identifiability_results.append((spec.name, metrics))

    print_metrics(
        f"{spec.name}: independent X -> G prediction",
        metrics,
    )


# ============================================================================
# 7. Interpretation / automatic checker
# ============================================================================

print("\n" + "=" * 72)
print("4. AUTOMATIC INTERPRETATION")
print("=" * 72)

best_name, best_metrics = max(
    identifiability_results,
    key=lambda item: item[1]["r2"],
)

print(f"\nBest deterministic feature model: {best_name}")
print(f"Test R^2:                         {best_metrics['r2']:.4f}")
print(f"Test correlation:                 {best_metrics['corr']:.4f}")
print(f"Test NRMSE:                       {best_metrics['nrmse']:.4f}")


if gradient_metrics["r2"] > 0.98:
    print(
        "\nPASS: G can be accurately recovered from the resolved trajectory X(t)."
    )
else:
    print(
        "\nWARNING: Closure extraction from X(t) itself introduces substantial error."
    )


if best_metrics["r2"] >= 0.90:
    print(
        "PASS: The selected X-only feature library contains enough information "
        "for strong deterministic closure inference."
    )

elif best_metrics["r2"] >= 0.70:
    print(
        "MODERATE: The closure is learnable from X, but an exact instantaneous "
        "match should not be expected from the current deterministic feature set."
    )

else:
    print(
        "LIMITATION: The instantaneous closure is only weakly identifiable from "
        "the current X-only feature library. MCMC using the same features should "
        "not be expected to recover G exactly."
    )


# ============================================================================
# 8. Plot: true resolved-state contour + first three slow states
# ============================================================================

plt.rcParams.update({
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})


fig = plt.figure(figsize=(10.0, 9.0), constrained_layout=True)

gs = fig.add_gridspec(
    4,
    1,
    height_ratios=[2.0, 1.0, 1.0, 1.0],
)

# --------------------------------------------------------------------------
# (a) Full resolved-state contour
# --------------------------------------------------------------------------

ax0 = fig.add_subplot(gs[0])

k = np.arange(1, cfg.n_slow + 1)

levels = np.linspace(
    np.percentile(x, 1.0),
    np.percentile(x, 99.0),
    101,
)

contour = ax0.contourf(
    t,
    k,
    x.T,
    levels=levels,
    cmap="viridis",
    extend="both",
)

ax0.set_ylabel(r"State index, $k$")
ax0.set_title("Two-level Lorenz--96 truth")

ax0.text(
    0.012,
    0.94,
    r"(a) $X_k(t)$",
    transform=ax0.transAxes,
    ha="left",
    va="top",
    bbox={
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.85,
        "pad": 2.0,
    },
)

cbar = fig.colorbar(
    contour,
    ax=ax0,
    pad=0.015,
    aspect=35,
)

cbar.set_label(r"Resolved state, $X_k$")


# --------------------------------------------------------------------------
# (b-d) First three resolved states
# --------------------------------------------------------------------------

for i in range(3):

    ax = fig.add_subplot(gs[i + 1])

    ax.plot(
        t,
        x[:, i],
        color="black",
        lw=1.1,
    )

    ax.set_ylabel(rf"$X_{{{i+1}}}$")
    ax.grid(True, alpha=0.20, lw=0.6)

    ax.text(
        0.012,
        0.90,
        f"({chr(ord('b') + i)})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        bbox={
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.8,
            "pad": 1.5,
        },
    )

    if i < 2:
        ax.tick_params(labelbottom=False)

    if i == 2:
        ax.set_xlabel(r"Time, $t$")


plt.show()


# ============================================================================
# 9. Optional diagnostic: true vs recovered closure G1
# ============================================================================

fig, ax = plt.subplots(
    figsize=(10.0, 4.5),
    constrained_layout=True,
)

ax.plot(
    t,
    g_true[:, 0],
    color="black",
    lw=1.2,
    label=r"True, $G_1=\sum_jY_{j,1}$",
)

ax.plot(
    t,
    g_gradient[:, 0],
    "--",
    lw=1.1,
    label=r"Recovered from $X$, gradient",
)

ax.plot(
    t,
    g_savgol[:, 0],
    lw=1.0,
    label=r"Recovered from $X$, Savitzky--Golay",
)

ax.set_xlabel(r"Time, $t$")
ax.set_ylabel(r"Closure, $G_1$")

ax.grid(True, alpha=0.20, lw=0.6)
ax.legend(
    frameon=True,
    facecolor="white",
    framealpha=0.95,
    loc="best",
)

plt.show()