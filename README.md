# Bayesian Closure Inference and Ensemble Data Assimilation for Uncertainty-Aware Digital Twins of Partially Observed Multiscale Systems

## Overview

This repository contains the computational framework accompanying the manuscript **“Bayesian Closure Inference and Ensemble Data Assimilation for Uncertainty-Aware Digital Twins of Partially Observed Multiscale Systems.”**

The framework addresses predictive modeling of multiscale dynamical systems when part of the system dynamics is unresolved or unobserved. It combines two complementary components:

1. **Offline Bayesian closure inference** — unresolved effects are reconstructed from resolved trajectories and represented using prescribed closure models. Markov chain Monte Carlo (MCMC) inference is used to characterize uncertainty in the closure coefficients.
2. **Online ensemble data assimilation** — the posterior-informed reduced model is embedded in an Ensemble Kalman Filter (EnKF) digital twin. As observations become available, the ensemble is forecast and corrected sequentially to improve state reconstruction and propagate model and state uncertainty.

The same workflow is examined using three benchmark problems of increasing complexity:

- a two-state **linear slow–fast system**;
- a partially observed **Lorenz–63 system**;
- the **two-level Lorenz–96 system** with unresolved fast variables.

The repository is organized so that closure identification, posterior analysis, free-running reduced-model evaluation, and online digital-twin experiments can be executed and plotted independently.

---

## Methodology at a Glance

The computational workflow is divided into an **offline closure-learning stage** and an **online digital-twin stage**.

### 1. Offline closure inference

For each benchmark, the influence of unresolved variables on the resolved dynamics is extracted from reference trajectories. Candidate closure representations of increasing complexity are then evaluated.

The Bayesian inference stage estimates a posterior distribution for the closure coefficients rather than relying only on a deterministic best fit. This enables posterior uncertainty quantification and posterior predictive evaluation.

The closure models are assessed using both *a priori* and *a posteriori* diagnostics, including closure reconstruction, state prediction, forecast horizon, statistical behavior, posterior uncertainty, and independent-trajectory performance.

### 2. Online digital twin

The inferred closure is subsequently embedded within the reduced governing equations. The offline posterior supplies uncertainty in the closure coefficients to the ensemble representation used by the digital twin.

At assimilation time $k$, the digital posterior is represented by an ensemble

$$
\pi_k(X_k,\xi) = p(X_k,\xi\mid\mathcal{D}_k), \qquad \\{(X_k^{(n)},\xi^{(n)})\\}_{n=1}^{N_e},
$$

where $X_k$ denotes the resolved state and $\xi$ denotes the learned closure coefficients.

Each ensemble member is propagated according to

$$
X_{k+1}^{-,(n)}
=
\mathcal{F}_k
\left(
X_k^{+,(n)},\xi_k^{(n)}
\right)
+
w_k^{(n)},
$$

while observations satisfy

$$
y_{k+1}
=
H_{k+1}X_{k+1}
+
v_{k+1}.
$$

The EnKF then performs a sequential analysis update. Depending on the experiment, the closure coefficients can either remain fixed at their offline values or be included in the augmented state for joint state–parameter updating.

Importantly, unresolved reference variables and exact closure quantities are used only for retrospective verification and are not supplied to the EnKF forecast or analysis update.

---

## Benchmark Systems

### Linear slow–fast system

The first benchmark provides a controlled setting in which the influence of an unresolved variable can be studied directly. Candidate closure models include:

- linear resolved-state closure;
- quadratic resolved-state closure.

This system provides a baseline for evaluating posterior closure recovery, autonomous reduced-order prediction, and EnKF state/parameter estimation.

### Partially observed Lorenz–63 system

The second benchmark introduces nonlinear chaotic dynamics and partial observability. The framework evaluates progressively richer closure descriptions constructed from available resolved variables.

Candidate hierarchies include:

- $x$-only linear and quadratic representations;
- $(x,y)$ linear and quadratic representations.

For online experiments, the EnKF assimilates only the prescribed resolved measurements. The hidden Lorenz variable and exact unresolved forcing are retained for retrospective assessment rather than being used by the filter.

### Two-level Lorenz–96 system

The third benchmark considers a spatially distributed multiscale chaotic system in which the slow variables are resolved and the fast variables remain unresolved.

Candidate closure hierarchies progressively increase the spatial information supplied to the closure:

- local linear and quadratic models;
- $\pm1$-neighbor linear and quadratic models;
- $\pm2$-neighbor linear and quadratic models.

This hierarchy preserves a common inference methodology while increasing the information available to the closure in a physically meaningful manner.

---

## Repository Structure

The principal modules are organized around system definitions, closure inference/evaluation, and online data assimilation.

```text
.
├── linear_system.py
├── lorenz63_system.py
├── lorenz96_system.py
├── problem_registry.py
│
├── closure_inference.py
├── closure_evaluation.py
├── main_closure_analysis.py
├── plot_closure_results.py
│
├── enkf_digital_twin.py
├── run_linear_digital_twin.py
├── run_lorenz63_digital_twin.py
├── run_lorenz96_digital_twin.py
│
├── plot_linear_digital_twin.py
├── plot_lorenz63_digital_twin.py
└── plot_lorenz96_digital_twin.py
```

### Core files

| File | Purpose |
|---|---|
| `linear_system.py` | Defines the linear slow–fast benchmark. |
| `lorenz63_system.py` | Defines the partially observed Lorenz–63 benchmark. |
| `lorenz96_system.py` | Defines the two-level Lorenz–96 benchmark. |
| `problem_registry.py` | Maps system names to the corresponding problem modules. |
| `closure_inference.py` | Shared feature construction, MCMC inference, posterior closure representation, and reduced-model integration. |
| `closure_evaluation.py` | Shared configuration, metrics, result storage/loading, and evaluation utilities. |
| `main_closure_analysis.py` | System-independent driver for the offline closure analyses. |
| `plot_closure_results.py` | Generates figures from previously saved closure-analysis results. |
| `enkf_digital_twin.py` | Reusable ensemble/augmented-state EnKF implementation. |
| `run_*_digital_twin.py` | Runs the online digital-twin experiment for each benchmark. |
| `plot_*_digital_twin.py` | Generates digital-twin diagnostics and publication-oriented figures. |

---

## Running the Offline Closure Analysis

Run all three benchmark systems with

```bash
python main_closure_analysis.py
```

or execute `main_closure_analysis.py` directly from a Python IDE.

The systems are processed sequentially:

```text
linear → lorenz63 → lorenz96
```

Before each analysis, the driver checks whether the corresponding saved result already exists. Completed systems can therefore be skipped without repeating the MCMC calculation.

Representative result files are

```text
closure_analysis_results_linear.pkl
closure_analysis_results_lorenz63.pkl
closure_analysis_results_lorenz96.pkl
```

To rerun a particular system, remove its corresponding saved result and execute the main analysis again.

---

## Plotting Saved Closure Results

Previously generated results can be visualized independently. For example,

```bash
python plot_closure_results.py closure_analysis_results_linear.pkl
python plot_closure_results.py closure_analysis_results_lorenz63.pkl --state 1
python plot_closure_results.py closure_analysis_results_lorenz96.pkl --state 0
```

The `--state` argument uses zero-based indexing.

The plotting utilities read the metadata stored with each result so that system-specific titles, state labels, closure definitions, dimensions, training diagnostics, and independent-trajectory results can be recovered automatically.

---

## Running the Online Digital Twin

The digital-twin stage consumes the posterior information generated during offline closure analysis. It does **not** require refitting the closure.

### Linear system

```bash
python run_linear_digital_twin.py closure_analysis_results_linear.pkl
python plot_linear_digital_twin.py digital_twin_results_linear.pkl
```

The linear implementation also supports comparison between joint state–parameter updating and state assimilation with fixed closure coefficients when the corresponding runner option is enabled.

### Lorenz–63

A representative command is

```bash
python run_lorenz63_digital_twin.py closure_analysis_results_lorenz63_algebraic.pkl --observe xy
python plot_lorenz63_digital_twin.py digital_twin_results_lorenz63.pkl
```

Available observation configurations include `x`, `y`, and `xy`. Only the selected resolved measurements are supplied to the filter.

### Lorenz–96

```bash
python run_lorenz96_digital_twin.py closure_analysis_results_lorenz96.pkl
python plot_lorenz96_digital_twin.py digital_twin_results_lorenz96.pkl --state 0
```

The Lorenz–96 experiment can use partial observations of the resolved slow variables, allowing the digital twin to be evaluated without feeding the complete resolved state to the filter.

---

## Evaluation Metrics

The framework is designed to assess more than pointwise closure fitting. Depending on the benchmark and analysis stage, diagnostics include:

- normalized root-mean-square error (NRMSE);
- coefficient of determination ($R^2$);
- correlation;
- posterior credible-interval coverage;
- autonomous forecast horizon;
- state reconstruction error;
- free-run, forecast, and analysis RMSE;
- ensemble spread;
- empirical coverage;
- closure-tracking error;
- quantities of interest (QoIs);
- independent-trajectory prediction;
- statistical distribution and temporal-correlation diagnostics.

These diagnostics distinguish between accurately reconstructing unresolved forcing from available data and maintaining predictive skill when the learned reduced model is integrated autonomously or used within a sequential digital twin.

---

## Sensitivity Studies

For online EnKF experiments, useful sensitivity parameters include:

1. ensemble size;
2. assimilation interval;
3. observed-state fraction;
4. sensor-noise level;
5. localization radius for spatially distributed systems;
6. fixed versus online-updated closure coefficients.

When conducting sensitivity studies, changing one parameter at a time provides the clearest interpretation of its influence on forecast error, analysis error, ensemble calibration, closure tracking, and QoI prediction.

---

## Reproducibility and Saved Results

Computationally expensive inference and plotting are separated throughout the repository. Saved `.pkl` files retain the information required for subsequent evaluation and visualization.

A typical workflow is therefore

```text
Reference simulation
        ↓
Closure extraction
        ↓
Candidate closure construction
        ↓
Bayesian/MCMC inference
        ↓
Posterior closure evaluation
        ↓
Saved closure-analysis results
        ↓
Reduced-model / digital-twin initialization
        ↓
EnKF forecast–analysis cycles
        ↓
Saved digital-twin results
        ↓
Figures and diagnostics
```

This structure allows figures and post-processing choices to be revised without repeating the Bayesian inference stage.

---

## Interpretation

The three benchmark systems are intended to expose different levels of difficulty in closure-informed digital twins. The linear system provides a controlled baseline, Lorenz–63 introduces nonlinear chaotic sensitivity and partial observation, and two-level Lorenz–96 extends the problem to a high-dimensional multiscale setting with spatially distributed unresolved dynamics.

A central distinction in the framework is between **offline uncertainty in unresolved model physics** and **online uncertainty in the evolving system state**. Bayesian closure inference addresses the former by providing posterior distributions for the learned closure, while ensemble data assimilation addresses the latter by sequentially incorporating observations. Their combination provides a unified route from closure identification to uncertainty-aware state reconstruction and prediction.

---

## Notes on Hidden/Unresolved Variables

Reference simulations contain the complete state so that exact unresolved forcing can be constructed and the learned closure can be evaluated.

During the online digital-twin experiments, however, hidden variables and exact closure values are not supplied as observations to the EnKF. They are retained only as reference quantities for retrospective error analysis and verification.

This distinction is important when interpreting digital-twin performance under partial observability.

---

## Citation

If you use this repository in academic work, please cite the accompanying manuscript. Citation information will be added following publication.

```bibtex
@article{Kassem2026System,
  title   = {System-Aware Uncertainty Reduction for Multiscale Digital Twins:
From Bayesian Closure Inference to Ensemble Data Assimilation},
  author  = {Kassem, Ashraf and Kara, Kursat, and San, Omer},
  year    = {2026},
  note    = {Manuscript in preparation}
}
```

---

## License

Please refer to the repository license file for terms of use and redistribution.
