"""Publication-style plots for the two-level Lorenz-96 closure problem."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from closure_evaluation import (
    load_results,
    plot_apriori_metrics,
    plot_calibration_metrics,
    plot_long_time_metrics,
    plot_model_statistical_curves,
    plot_short_time_metrics,
)


plt.rcParams.update(
    {
        "mathtext.fontset": "stix",
        "font.family": "serif",
        "font.size": 12.0,
        "axes.labelsize": 13.5,
        "legend.fontsize": 11,
        "xtick.labelsize": 10.8,
        "ytick.labelsize": 10.8,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "figure.dpi": 120,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
    }
)

C_TRUTH = "#222222"
C_MODEL = "#1F5A94"
C_AUX = "#B56A2D"
C_SECONDARY = "#4E7A6A"
C_NEUTRAL = "#777777"

MODEL_COLORS = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermillion
    "#56B4E9",  # sky blue
    "#6B6B6B",  # gray
    "#8C564B",  # brown
    "#000000",  # black
)

def _style(ax):
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


def _restyle_external_figure(fig):
    """Apply publication styling to figures returned by closure_evaluation."""
    for ax in fig.axes:
        ax.tick_params(direction="in", top=True, right=True)
        if ax.lines or ax.patches:
            ax.grid(True, color="0.88", lw=0.5)
        leg = ax.get_legend()
        if leg is not None:
            leg.set_frame_on(True)
            frame = leg.get_frame()
            frame.set_facecolor("white")
            frame.set_edgecolor("0.75")
            frame.set_alpha(0.75)


def _finite_symmetric_limit(*arrays: np.ndarray, percentile: float = 99.0) -> float:
    """Return a robust symmetric color limit from one or more arrays."""
    values = np.concatenate([np.asarray(a)[np.isfinite(a)].ravel() for a in arrays])
    if values.size == 0:
        return 1.0
    limit = float(np.percentile(np.abs(values), percentile))
    return limit if limit > np.finfo(float).eps else 1.0


def plot_state_time_contours(
    t: np.ndarray,
    x_true: np.ndarray,
    x_model: np.ndarray,
    *,
    training_interval: tuple[float, float] | None = None,
    state_cmap: str = "viridis",
    error_cmap: str = "BrBG",
    percentile: float = 99.0,
):
    """Plot truth, modeled state, and error fields over state index and time.

    Notation
    --------
    X_k(t)       : resolved-state truth
    Xhat_k(t)    : a-posteriori trajectory using the inferred closure
    e_k(t)       : Xhat_k(t) - X_k(t)
    """
    t = np.asarray(t)
    x_true = np.asarray(x_true)
    x_model = np.asarray(x_model)

    if x_true.ndim != 2 or x_model.ndim != 2:
        raise ValueError("x_true and x_model must have shape (n_time, n_state).")
    if x_true.shape != x_model.shape:
        raise ValueError(
            f"Truth/model shapes differ: {x_true.shape} versus {x_model.shape}."
        )
    if t.ndim != 1 or t.size != x_true.shape[0]:
        raise ValueError("t must be one-dimensional and match the time dimension.")

    # error = x_model - x_true
    
    
    # state_limit = 10.0
    # error_limit = _finite_symmetric_limit(error, percentile=percentile)
    # state_norm = TwoSlopeNorm(vmin=-10.0, vcenter=0.0, vmax=10.0)
    # error_norm = TwoSlopeNorm(vmin=-error_limit, vcenter=0.0, vmax=error_limit)
    
    # error = x_model - x_true
    
    # error = x_model - x_true
    # # Normalize error by its maximum absolute value
    # error_scale = np.nanmax(np.abs(error))

    # if error_scale > np.finfo(float).eps:
    #     error = error / error_scale
    # else:
    #     error = np.zeros_like(error)
    raw_error = x_model - x_true
    
    # Normalize by the RMS magnitude of the truth state
    state_rms = np.sqrt(np.nanmean(x_true**2))

    if state_rms > np.finfo(float).eps:
        error = raw_error / state_rms
    else:
        error = np.zeros_like(raw_error)




    
    # ------------------------------------------------------------------
    # State limits: use actual extrema and round outward to nearest integer
    # ------------------------------------------------------------------
    state_min = min(np.nanmin(x_true), np.nanmin(x_model))
    state_max = max(np.nanmax(x_true), np.nanmax(x_model))

    state_lower = np.floor(state_min)
    state_upper = np.ceil(state_max)

    # Use a symmetric color scale around zero
    state_limit = max(abs(state_lower), abs(state_upper))

    state_norm = TwoSlopeNorm(
        vmin=-state_limit,
        vcenter=0.0,
        vmax=state_limit,
    )

    # ------------------------------------------------------------------
    # Error limits: use actual extrema and round outward to nearest integer
    # ------------------------------------------------------------------
    # error_limit = 1.0
    error_min = np.nanmin(error)
    error_max = np.nanmax(error)

    error_lower = np.floor(error_min)
    error_upper = np.ceil(error_max)

    # Use a symmetric color scale around zero
    error_limit = max(abs(error_lower), abs(error_upper))

    error_norm = TwoSlopeNorm(
        vmin=-error_limit,
        vcenter=0.0,
        vmax=error_limit,
    )




    # pcolormesh preserves the sampled Lorenz-96 state indices without imposing
    # artificial interpolation in the discrete spatial coordinate k.
    k = np.arange(1, x_true.shape[1] + 1)
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.6, 7.9),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    fields = (x_true, x_model, error)
    norms = (state_norm, state_norm, error_norm)
    cmaps = (state_cmap, state_cmap, error_cmap)
    panel_labels = (r"(a) $X_k(t)$", r"(b) $\widehat{X}_k(t)$", r"(c) $e_k(t)$")

    meshes = []
    for ax, field, norm, cmap, label in zip(
        axes, fields, norms, cmaps, panel_labels, strict=True
    ):
        # mesh = ax.pcolormesh(t, k, field.T, shading="auto", cmap=cmap, norm=norm)
        levels = np.linspace(norm.vmin, norm.vmax, 101) 
        mesh = ax.contourf(t, k, field.T, levels=levels, cmap=cmap, norm=norm, extend="both")
        
        meshes.append(mesh)
        ax.set_ylabel(r"State index, $k$")
        ax.text(
            0.012,
            0.94,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.0},
        )
        if training_interval is not None:
            ax.axvline(
                training_interval[1],
                color="0.15",
                lw=1.0,
                ls="--",
                alpha=0.85,
            )

    axes[-1].set_xlabel(r"Time, $t$")
    axes[0].set_ylim(0.5, x_true.shape[1] + 0.5)

    # Truth and prediction deliberately share one colorbar and normalization.
    # cbar_state = fig.colorbar(meshes[1], ax=axes[:2], pad=0.015, aspect=35, format="%.2f")
    # cbar_state = fig.colorbar(meshes[1], ax=axes[:2], pad=0.015, aspect=35, 
    #                           ticks=np.linspace(-10, 10, 9), format="%.1f")
    # cbar_state.set_label(r"Resolved state, $X_k$")
    # cbar_error = fig.colorbar(meshes[2], ax=axes[2], pad=0.015, aspect=18, format="%.2f")
    # cbar_error.set_label(r"State error, $e_k=\widehat{X}_k-X_k$")
    
    cbar_state = fig.colorbar(
        meshes[1],
        ax=axes[:2],
        pad=0.015,
        aspect=35,
        ticks=np.linspace(-state_limit, state_limit, 5),
        format="%.2f",
    )
    cbar_state.set_label(r"Resolved state, $X_k$")

    cbar_error = fig.colorbar(
        meshes[2],
        ax=axes[2],
        pad=0.015,
        aspect=18,
        ticks=np.linspace(-error_limit, error_limit, 5),
        format="%.2f",
    )
    # cbar_error.set_label(r"State error, $e_k=\widehat{X}_k-X_k$")
    cbar_error.set_label(
        r"Normalized state error,  $e_k/X_{\rm rms}$", fontsize=11,
    )
    
    
    return fig, axes


def plot_lorenz96_results(
    filename: str = "closure_analysis_results_lorenz96.pkl",
    state_index: int = 0,
):
    data = load_results(filename)
    display_name = data.get("display_name", "Two-level Lorenz-96")
    state_labels = data.get("state_labels", [])
    closure_label = data.get("closure_label", r"$G_k$")

    physical = data["physical"]
    physical_test = data["physical_test"]
    t = data["t"]
    x_true = data["x_true"]
    t_train = data["t_train"]
    g_train = data["g_train"]
    results = data["results"]
    t_test = data["t_test"]
    x_true_test = data["x_true_test"]
    test_results = data["test_results"]

    # Publication plotting windows: exclude the pre-training transient/spin-up.
    # Keep the complete training and post-training forecast/testing intervals.
    plot_tmin = float(t_train[0])
    plot_tmax = float(t[-1])
    test_tmin = float(t_test[0])
    test_tmax = float(t_test[-1])

    if not results:
        raise RuntimeError("No successful Lorenz-96 training models are available.")

    state_index = int(state_index)
    if state_index < 0 or state_index >= x_true.shape[1]:
        raise ValueError(
            f"state_index={state_index} is invalid for {x_true.shape[1]} states."
        )
    state_label = (
        state_labels[state_index]
        if state_index < len(state_labels)
        else rf"$X_{{{state_index + 1}}}$"
    )
    closure_component = state_index if g_train.shape[1] == x_true.shape[1] else 0

    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    ax.plot(
        t_train,
        g_train[:, closure_component],
        color=C_TRUTH, linestyle="--",
        lw=1.6,
        label=rf"Target, $G_{{{state_index + 1}}}$",
    )
    for i, item in enumerate(results):
        ax.plot(
            t_train,
            item["g_mean"][:, closure_component],
            color=MODEL_COLORS[i % len(MODEL_COLORS)],
            lw=1.1,
            label=rf"{item['spec'].name}, $\widehat{{G}}_{{{state_index + 1}}}$",
        )
    
    
    ax.set_xlabel(r"Time, $t$")
    ax.set_ylabel(closure_label)
    ax.set_xlim(float(t_train[0]), float(t_train[-1]))
    ax.grid(True, alpha=0.22, lw=0.6)
    _legend(ax, ncol=2, loc="best")
    
    path_results='results/lorenz96/closure_results/'
    fig.savefig(path_results+
        "lorenz96_training_closure_hierarchy.png",
        dpi=400,
        bbox_inches="tight",
    )
        
    

    # plot_apriori_metrics(results, prefix=f"{display_name} — training: ")
    # plot_calibration_metrics(results, prefix=f"{display_name} — training: ")
    # plot_short_time_metrics(t, results, prefix=f"{display_name} — training: ")
    # plot_long_time_metrics(results, prefix=f"{display_name} — training: ")
    fig_apriori = plot_apriori_metrics(
        results, prefix=f"{display_name} — training: "
    )[0]
    _restyle_external_figure(fig_apriori)
    fig_apriori.savefig(path_results+
        "lorenz96_training_apriori_metrics.png",
        dpi=400,
        bbox_inches="tight",
    )

    fig_calibration = plot_calibration_metrics(
        results, prefix=f"{display_name} — training: "
    )[0]
    _restyle_external_figure(fig_calibration)
    fig_calibration.savefig(path_results+
        "lorenz96_training_calibration.png",
        dpi=400,
        bbox_inches="tight",
    )

    fig_short = plot_short_time_metrics(
        t, results, prefix=f"{display_name} — training: "
    )[0]
    _restyle_external_figure(fig_short)
    fig_short.savefig(path_results+
        "lorenz96_training_short_time.png",
        dpi=400,
        bbox_inches="tight",
    )

    training_long_time = plot_long_time_metrics(
        results, prefix=f"{display_name} — training: "
    )
    for i, (fig_metric, _) in enumerate(training_long_time, start=1):
        _restyle_external_figure(fig_metric)
        fig_metric.savefig(path_results+
            f"lorenz96_training_long_time_{i}.png",
            dpi=400,
            bbox_inches="tight",
        )
    
    # ==================================================================
    # Compare state reconstruction for all closure models
    # ==================================================================

    # Exclude the initial transient from the reconstruction comparison.
    recon_mask = (t >= plot_tmin) & (t <= plot_tmax)

    t_recon = t[recon_mask]
    x_true_recon = x_true[recon_mask]

    n_models = len(results)
    ncols = 2
    nrows = int(np.ceil(n_models / ncols))

    fig_compare, axes_compare = plt.subplots(
        nrows,
        ncols,
        figsize=(12.0, 3.2 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes_compare = np.atleast_1d(axes_compare).ravel()

    for i, item in enumerate(results):

        x_model_recon = np.asarray(item["x_online"])[recon_mask]

        # RMSE of the first resolved state, X_1
        rmse_x1 = np.sqrt(
            np.mean(
                (
                    x_model_recon[:, 0]
                    - x_true_recon[:, 0]
                ) ** 2
            )
        )

        ax = axes_compare[i]

        ax.plot(
            t_recon,
            x_true_recon[:, 0],
            color=C_TRUTH,
            lw=1.3,
            label=r"Truth, $X_1$",
        )

        ax.plot(
            t_recon,
            x_model_recon[:, 0],
            "--",
            color=MODEL_COLORS[i % len(MODEL_COLORS)],
            lw=1.2,
            label=r"Reconstruction, $\widehat{X}_1$",
        )

        ax.set_title(
            rf"{item['spec'].name}: RMSE$_{{X_1}}$ = {rmse_x1:.3f}",
            fontsize=11,
        )

        ax.set_xlim(plot_tmin, plot_tmax)
        ax.set_xlabel(r"Time, $t$")
        ax.set_ylabel(r"$X_1$")
        _style(ax)

        if i == 0:
            _legend(ax, ncol=1, loc="best")


    # Remove unused panels, if any
    for j in range(n_models, len(axes_compare)):
        fig_compare.delaxes(axes_compare[j])


    fig_compare.savefig(
        path_results + "lorenz96_all_model_X1_reconstruction.png",
        dpi=400,
        bbox_inches="tight",
    )

    print("\nResolved-state reconstruction RMSE:")
    
        
    
    # best = min(
    #     results,
    #     key=lambda item: item["evaluation"]["long_time"]["spatial_corr_rmse"],
    # )
    # score = best["evaluation"]["long_time"]["spatial_corr_rmse"]
    # print(
    #     f"\nTraining-selected Lorenz-96 closure: {best['spec'].name} "
    #     f"(spatial-correlation RMSE={score:.6e})"
    # )
    # ==================================================================
    
    # Select model using resolved-state reconstruction RMSE
    # ==================================================================

    def reconstruction_rmse(item):
        """RMSE of reconstructed resolved states over the plotting window."""

        x_model_recon = np.asarray(item["x_online"])[recon_mask]

        rmse = np.sqrt(
            np.mean(
                (x_model_recon - x_true_recon) ** 2
            )
        )

        # Prevent NaN/Inf trajectories from being selected
        if not np.isfinite(rmse):
            return np.inf

        return float(rmse)
    
    print("\nResolved-state reconstruction RMSE:")
    for item in results:
        print(
            f"  {item['spec'].name:<20s} "
            f"{reconstruction_rmse(item):.6e}"
        )



    best = min(
        results,
        key=reconstruction_rmse,
    )

    score = reconstruction_rmse(best)

    print(
        f"\nTraining-selected Lorenz-96 closure: {best['spec'].name} "
        f"(state-reconstruction RMSE={score:.6e})"
    )


    fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
    ax.plot(t, x_true[:, state_index], color=C_TRUTH, lw=1.3, label=rf"Truth, ${state_label.strip('$')}$")
    ax.plot(
        t,
        best["x_online"][:, state_index],
        "--",
        color=C_MODEL,
        lw=1.3,
        label=rf"Model, $\widehat{{X}}_{{{state_index + 1}}}$",
    )
    ax.axvspan(t_train[0], t_train[-1], color="0.5", alpha=0.10, label=r"Training, $\mathcal{T}_{\rm tr}$")
    ax.set_xlabel(r"Time, $t$")
    ax.set_ylabel(state_label)
    ax.set_xlim(plot_tmin, plot_tmax)
    ax.grid(True, alpha=0.22, lw=0.6)
    _legend(ax, ncol=1, loc="best")
    
    fig.savefig(path_results+
        "lorenz96_training_state_trajectory.png",
        dpi=400,
        bbox_inches="tight",
    )
    

    # fig, axes = plot_state_time_contours(
    #     t,
    #     x_true,
    #     best["x_online"],
    #     training_interval=(float(t_train[0]), float(t_train[-1])),
    # )
    fig, axes = plot_state_time_contours(
        t,
        x_true,
        best["x_online"],
    )
    
    for ax in axes:
        ax.set_xlim(plot_tmin, plot_tmax)
    fig.suptitle("Training", fontweight="semibold")
    
    fig.savefig(path_results+
        "lorenz96_training_state_contours.png",
        dpi=400,
        bbox_inches="tight",
    )
    
    

    
    # plot_model_statistical_curves(
    #     x_true,
    #     best,
    #     dt=physical.dt,
    #     state_labels=state_labels,
    #     prefix=f"{display_name} — training: ",
    # )
    
    training_statistics = plot_model_statistical_curves(
        x_true,
        best,
        dt=physical.dt,
        state_labels=state_labels,
        prefix=f"{display_name} — training: ",
    )

    for i, (fig_stat, _) in enumerate(training_statistics, start=1):
        _restyle_external_figure(fig_stat)
        fig_stat.savefig(path_results+
            f"lorenz96_training_statistics_{i}.png",
            dpi=400,
            bbox_inches="tight",
        )


    # # Keep the model selected using the training trajectory to prevent leakage.
    # plot_apriori_metrics(test_results, prefix=f"{display_name} — independent trajectory: ")
    # plot_calibration_metrics(test_results, prefix=f"{display_name} — independent trajectory: ")
    # plot_short_time_metrics(t_test, test_results, prefix=f"{display_name} — independent trajectory: ")
    # plot_long_time_metrics(test_results, prefix=f"{display_name} — independent trajectory: ")
    
    fig_apriori = plot_apriori_metrics(
        test_results, prefix=f"{display_name} — independent trajectory: "
    )[0]
    _restyle_external_figure(fig_apriori)
    fig_apriori.savefig(path_results+
        "lorenz96_testing_apriori_metrics.png",
        dpi=400,
        bbox_inches="tight",
    )

    fig_calibration = plot_calibration_metrics(
        test_results, prefix=f"{display_name} — independent trajectory: "
    )[0]
    _restyle_external_figure(fig_calibration)
    fig_calibration.savefig(path_results+
        "lorenz96_testing_calibration.png",
        dpi=400,
        bbox_inches="tight",
    )

    fig_short = plot_short_time_metrics(
        t_test, test_results, prefix=f"{display_name} — independent trajectory: "
    )[0]
    _restyle_external_figure(fig_short)
    fig_short.savefig(path_results+
        "lorenz96_testing_short_time.png",
        dpi=400,
        bbox_inches="tight",
    )

    training_long_time = plot_long_time_metrics(
        test_results, prefix=f"{display_name} — independent trajectory: "
    )
    for i, (fig_metric, _) in enumerate(training_long_time, start=1):
        _restyle_external_figure(fig_metric)
        fig_metric.savefig(path_results+
            f"lorenz96_testing_long_time_{i}.png",
            dpi=400,
            bbox_inches="tight",
        )

    best_test = next(
        (item for item in test_results if item["spec"].name == best["spec"].name),
        None,
    )
    if best_test is None:
        print(
            f"Best training model '{best['spec'].name}' was not successful "
            "on the independent trajectory."
        )
    else:
        fig, ax = plt.subplots(figsize=(7.4, 4.4), constrained_layout=True)
        ax.plot(t_test, x_true_test[:, state_index], color=C_TRUTH, lw=1.3, label=rf"Truth, ${state_label.strip('$')}$")
        ax.plot(
            t_test,
            best_test["x_online"][:, state_index],
            "--",
            color=C_MODEL,
            lw=1.3,
            label=rf"Model, $\widehat{{X}}_{{{state_index + 1}}}$",
        )
    
        ax.set_xlabel(r"Time, $t$")
        ax.set_ylabel(state_label)
        ax.set_xlim(test_tmin, test_tmax)
        ax.grid(True, alpha=0.22, lw=0.6)
        _legend(ax, ncol=1, loc="best")
        fig.savefig(path_results+
            "lorenz96_testing_state_trajectory.png",
            dpi=400,
            bbox_inches="tight",
        )
    


        
        fig, axes = plot_state_time_contours(
            t_test,
            x_true_test,
            best_test["x_online"],
        )
        for ax in axes:
            ax.set_xlim(test_tmin, test_tmax)
        fig.suptitle("Testing", fontweight="semibold")
        fig.savefig(path_results+
            "lorenz96_testing_state_contours.png",
            dpi=400,
            bbox_inches="tight",
        )
    
    

        # plot_model_statistical_curves(
        #     x_true_test,
        #     best_test,
        #     dt=physical_test.dt,
        #     state_labels=state_labels,
        #     prefix=f"{display_name} — independent trajectory: ",
        # )
        testing_statistics = plot_model_statistical_curves(
            x_true_test,
            best_test,
            dt=physical_test.dt,
            state_labels=state_labels,
            prefix=f"{display_name} — independent trajectory: ",
        )

        for i, (fig_stat, _) in enumerate(testing_statistics, start=1):
            _restyle_external_figure(fig_stat)
            fig_stat.savefig(path_results+
                f"lorenz96_testing_statistics_{i}.png",
                dpi=400,
                bbox_inches="tight",
            )

    plt.show()
    
    print("Integrator dt:", physical.dt)
    print("Number of integration steps:", int((t[-1] - t[0]) / physical.dt) + 1)
    print("Number of stored points:", len(t))
    print("Stored-data dt:", np.mean(np.diff(t)))


def main():
    filename = "closure_analysis_results_lorenz96.pkl"
    if not os.path.exists(filename):
        print(f"\n{filename} does not exist. Nothing to plot.")
        return
    plot_lorenz96_results(filename, state_index=0)
    
    



if __name__ == "__main__":
    main()
