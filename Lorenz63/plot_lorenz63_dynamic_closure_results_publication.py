"""Publication plots for the Lorenz-63 dynamic latent-closure analysis."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import lorenz63_dynamic_system as problem


# Professional, restrained palette used consistently across all Lorenz-63 figures.
C_TRUTH = "#222222"      # charcoal
C_MODEL = "#1F5A94"      # muted dark blue
C_AUX = "#B56A2D"        # muted burnt orange
C_SECONDARY = "#4E7A6A"  # desaturated green
C_NEUTRAL = "#777777"    # gray

STYLE = {
    "mathtext.fontset": "stix",
    "font.family": "serif",
    "font.size": 12.0,
    "axes.labelsize": 13.5,
    "axes.titlesize": 14.0,
    "legend.fontsize": 9.2,
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
#         handlelength=2.0, columnspacing=1.0,
#         handletextpad=0.5, borderaxespad=0.35,
#         labelspacing=0.3,
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

def validate_payload(data):
    required = ("system_name", "train", "test", "results", "test_results", "selected_model_name")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"Missing dynamic-closure result keys: {missing}")
    if data["system_name"] != "lorenz63_dynamic":
        raise ValueError(f"Unexpected system_name {data['system_name']!r}.")
    names = [r["spec"].name for r in data["results"]]
    test_names = [r["spec"].name for r in data["test_results"]]
    if names != test_names:
        raise ValueError("Training/test model lists differ.")
    if data["selected_model_name"] not in names:
        raise ValueError("Selected model is not in result list.")
    for block in ("train", "test"):
        d = data[block]
        n = len(d["t"])
        if np.asarray(d["x_truth"]).shape != (n, 2):
            raise ValueError(f"{block}: x_truth shape mismatch.")
        if np.asarray(d["q_recovered"]).shape != (n, 1):
            raise ValueError(f"{block}: q_recovered shape mismatch.")
    print("\nDynamic-closure plotting payload checks: PASS")


def _selected(data, testing=False):
    results = data["test_results"] if testing else data["results"]
    return next(r for r in results if r["spec"].name == data["selected_model_name"])


def plot_realization(data, testing=False):
    d = data["test"] if testing else data["train"]
    item = _selected(data, testing)

    t = np.asarray(d["t"])
    truth = np.asarray(d["x_truth"])
    g_truth = np.asarray(d["g_truth"])[:, 0]
    pred = np.asarray(item["x_online"])
    g_pred = np.asarray(item["g_online"])[:, 0]
    q_pred = np.asarray(item["q_online"])[:, 0]
    q_rec = np.asarray(d["q_recovered"])[:, 0]
    hidden = np.asarray(d["hidden"])[:, 0]

    label = "Independent testing" if testing else "Training"

    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(
            4, 1, figsize=(7.2, 9.3), sharex=True,
            constrained_layout=True,
        )

        ax = axes[0]
        ax.plot(t, g_truth, color=C_TRUTH, lw=1.15, label=r"Truth, $G=-xz$")
        ax.plot(
            t, g_pred, "--", color=C_MODEL, lw=1.30,
            label=r"Closure, $\widehat G=-\widehat{x}\widehat{q}$",
        )
        ax.set_ylabel(r"Closure, $G$")
        _style(ax); _panel(ax, "(a)"); _legend(ax, ncol=1)

        for j, sym in enumerate(("x", "y"), start=1):
            ax = axes[j]
            ax.plot(t, truth[:, j - 1], color=C_TRUTH, lw=1.15, label=rf"Truth, ${sym}$")
            ax.plot(
                t, pred[:, j - 1], "--", color=C_MODEL, lw=1.30,
                label=rf"Model, $\widehat{{{sym}}}$",
            )
            ax.set_ylabel(rf"${sym}$")
            _style(ax); _panel(ax, f"({chr(97 + j)})"); _legend(ax, ncol=1)

        ax = axes[3]
        ax.plot(
            t, hidden, color=C_TRUTH, lw=1.05,
            label=r"Hidden truth, $z$ (verification)",
        )
        ax.plot(
            t, q_rec, color=C_AUX, lw=1.05, alpha=0.95,
            label=r"Resolved-data reconstruction, $q_{\mathrm{rec}}$",
        )
        ax.plot(
            t, q_pred, "--", color=C_MODEL, lw=1.30,
            label=r"Model, $\widehat q$",
        )
        ax.set_ylabel(r"Latent state")
        ax.set_xlabel(r"Time, $t$")
        _style(ax); _panel(ax, "(d)")
        _legend(ax, ncol=1, loc="best")

        fig.suptitle(label, fontsize=14.5, fontweight="semibold")

    return fig, axes


def plot_qdot_fit(data):
    """A-priori latent-dynamics fit with clearly distinguishable line styles."""
    d = data["train"]
    item = _selected(data, False)

    t = np.asarray(d["t"])
    qdot = np.asarray(d["qdot_recovered"])[:, 0]
    mean = np.asarray(item["qdot_mean"])[:, 0]
    lo = np.asarray(item["qdot_lower"])[:, 0]
    hi = np.asarray(item["qdot_upper"])[:, 0]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)

        # Truth/recovered target is solid; posterior mean is dashed so two
        # nearly coincident curves remain distinguishable in print.
        ax.plot(
            t, qdot, color=C_TRUTH, lw=1.10,
            label=r"Recovered target, $\dot q_{\mathrm{rec}}$",
            zorder=3,
        )
        ax.plot(
            t, mean, "--", color=C_MODEL, lw=1.45,
            label=r"Posterior mean, $\widehat F(x,y,q)$",
            zorder=4,
        )
        ax.fill_between(
            t, lo, hi, color=C_MODEL, alpha=0.12, linewidth=0,
            label=r"$95\%$ credible interval", zorder=1,
        )

        ax.set_xlabel(r"Time, $t$")
        ax.set_ylabel(r"Latent dynamics, $\dot q$")
        _style(ax)
        _legend(ax, ncol=2, loc="best")

    return fig, ax


def plot_candidate_screen(data):
    """Candidate comparison on a log scale so small quadratic/cubic errors remain visible."""
    rows = data["candidate_screen"]
    names = [r["spec"].name.replace("xyq-", "") for r in rows]
    fit = np.asarray([r["fit"]["nrmse_std"] for r in rows], dtype=float)
    val = np.asarray([r["validation"]["nrmse_std"] for r in rows], dtype=float)

    x = np.arange(len(names))
    w = 0.34

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(6.5, 4.0), constrained_layout=True)

        ax.bar(
            x - w / 2, fit, width=w,
            facecolor="0.78", edgecolor=C_NEUTRAL, linewidth=0.8,
            label="Fit NRMSE",
        )
        ax.bar(
            x + w / 2, val, width=w,
            facecolor="0.90", edgecolor=C_MODEL, linewidth=1.0,
            label="Held-out NRMSE",
        )

        ax.set_yscale("log")
        ax.set_xticks(x, names)
        ax.set_ylabel(r"NRMSE of $\dot q$")
        _style(ax)
        _legend(ax, ncol=2, loc="best")

    return fig, ax


def print_coefficients(data):
    item = _selected(data, False)
    beta = np.asarray(item["posterior"]["beta"])
    physical = np.vstack([
        problem.physical_coefficients(row, item["scaler"]) for row in beta
    ])
    names = problem.feature_names(item["spec"])
    print("\nPosterior coefficients for qdot in unscaled basis")
    print("-" * 64)
    for name, mean, std in zip(names, physical.mean(0), physical.std(0, ddof=1)):
        print(f"{name:>8s}: {mean: .6e} +/- {std:.3e}")


def print_metrics(data):
    name = data["selected_model_name"]
    print("\n" + "=" * 78)
    print(f"Lorenz-63 dynamic closure metrics: {name}")
    print("=" * 78)
    for label, item in (
        ("TRAINING", _selected(data, False)),
        ("INDEPENDENT TEST", _selected(data, True)),
    ):
        e = item["evaluation"]
        print(label)
        print(f"  qdot NRMSE           {e['qdot']['nrmse_std']:.6f}")
        print(f"  qdot R2              {e['qdot']['r2']:.6f}")
        print(f"  autonomous G NRMSE   {e['closure']['nrmse_std']:.6f}")
        print(f"  autonomous G R2      {e['closure']['r2']:.6f}")
        print(f"  state RMSE            {e['state_rmse']:.6f}")
        print(f"  state NRMSE           {e['state_nrmse']:.6f}")
        print(f"  forecast horizon      {e['forecast_horizon']:.6f}")
        print(f"  q vs hidden-z NRMSE*  {e['latent_q']['nrmse_std']:.6f}")
    print("* verification only; hidden z was not used for fitting.")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?",
        default="closure_analysis_results_lorenz63_dynamic.pkl",
    )
    parser.add_argument("--save-prefix", default="lorenz63_dynamic")
    parser.add_argument("--save-dir", default="results/lorenz63/dynamic_closure")
    args = parser.parse_args()

    with Path(args.input).open("rb") as f:
        data = pickle.load(f)

    validate_payload(data)
    print_coefficients(data)
    print_metrics(data)

    figs = [
        ("training", plot_realization(data, False)[0]),
        ("testing", plot_realization(data, True)[0]),
        ("qdot_fit", plot_qdot_fit(data)[0]),
        ("candidate_screen", plot_candidate_screen(data)[0]),
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
