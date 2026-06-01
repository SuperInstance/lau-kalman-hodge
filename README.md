# lau-kalman-hodge

**Hodge-Kalman-Spectral bridge: the Kalman filter as a Hodge star operator on observation bundles.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)

---

## What This Does

This crate implements a deep structural theorem from differential geometry applied to state estimation: **the Kalman filter *is* the Hodge star operator on the observation bundle**. It provides:

- **Observation bundles** (vector bundles over a discretized state manifold with fiber metrics)
- **Hodge decomposition** of observation data into harmonic, exact, and coexact components
- **Kalman filtering** whose predict-update cycle is reinterpreted as de Rham differential and codifferential
- **Spectral projection** via FFT that maps frequency bands to Hodge components
- **Observability Gramians** as Čech coboundary operators
- **Green's operators** that connect the Kalman gain to the Hodge Laplacian inverse
- A unified **belief estimator** integrating all components for agent architectures

Every observation sequence is decomposed as:

```
Observation = Harmonic ⊕ Exact(noise) ⊕ CoExact(uncertainty)
```

where the harmonic component is the Kalman steady-state estimate, the exact component is the innovation process, and the coexact component is the model-uncertainty residual.

---

## Key Idea

In classical Kalman filtering, the gain matrix `K` is derived from Riccati equations. This crate shows that `K` is actually the **Green's operator** `G = Δ⁻¹` for the Hodge Laplacian `Δ = dd* + d*d` on the observation bundle:

| Kalman Concept | Geometric Counterpart |
|---|---|
| Predict step | De Rham differential `d` |
| Update step | Codifferential `d*` |
| Kalman gain `K` | Green's operator `G = Δ⁻¹` |
| Steady-state estimate | Harmonic space `H⁰(M; E) = ker(Δ)` |
| Innovation `y - ŷ` | Exact form `dα` |
| Uncertainty residual | Coexact form `d*β` |
| Observability Gramian | Čech coboundary operator |
| Kalman covariance `P` | Information-form Laplacian |

The observation bundle `E → M` attaches a fiber (observation space with inner product) to each point of the discretized state manifold. Sections of this bundle are the observation data, and the Hodge decomposition splits any section into three mutually orthogonal parts.

---

## Install

```toml
[dependencies]
lau-kalman-hodge = "0.1.0"
```

Requires Rust 2021 edition. Dependencies: `nalgebra` (with serde), `rustfft`, `num-complex` (with serde).

---

## Quick Start

```rust
use lau_kalman_hodge::{
    BeliefEstimator, ObservationBundle, KalmanFilter,
    HodgeDecomposition, SpectralProjector, ObservabilityGramian, GreensOperator,
};
use nalgebra::{DMatrix, DVector};

// --- Kalman filtering the standard way ---
let f = DMatrix::identity(2, 2);          // state transition
let h = DMatrix::identity(2, 2);          // observation model
let q = DMatrix::identity(2, 2) * 0.1;    // process noise
let r = DMatrix::identity(2, 2) * 1.0;    // measurement noise
let x0 = DVector::from_vec(vec![0.0, 0.0]);
let p0 = DMatrix::identity(2, 2) * 10.0;

let mut kf = KalmanFilter::new(f, h, q, r, x0, p0);
kf.step(&DVector::from_vec(vec![5.0, -3.0]));
println!("Estimate: {:?}", kf.state.x);

// --- Or use the unified belief estimator ---
let mut belief = BeliefEstimator::new(
    DMatrix::identity(2, 2),
    DMatrix::identity(2, 2),
    DMatrix::identity(2, 2) * 0.1,
    DMatrix::identity(2, 2) * 1.0,
    DVector::from_vec(vec![0.0, 0.0]),
    DMatrix::identity(2, 2) * 10.0,
    20, // number of states for spectral analysis
);

let observations: Vec<DVector<f64>> = (0..20)
    .map(|i| DVector::from_vec(vec![i as f64 * 0.5, (i as f64 * -0.3)]))
    .collect();

let beliefs = belief.batch_estimate(&observations);
for (i, b) in beliefs.iter().enumerate() {
    println!("t={}: est={:?}, harmonic_energy={:.3}, steady={}",
        i, b.estimate, b.harmonic_energy, b.is_steady_state);
}
```

---

## API Reference

### Core Types

| Type | Module | Description |
|---|---|---|
| `Fiber` | `bundle` | Observation fiber with inner product metric |
| `Section` | `bundle` | Section of the observation bundle (obs data over states) |
| `ObservationBundle` | `bundle` | Full bundle `E → M` with observation matrix `H` |
| `HodgeDecomposition` | `hodge` | Result of splitting a section into harmonic ⊕ exact ⊕ coexact |
| `KalmanFilter` | `kalman` | Standard predict-update Kalman filter |
| `KalmanState` | `kalman` | State estimate, covariance, and innovation |
| `SpectralProjector` | `spectral` | FFT-based frequency decomposition into Hodge components |
| `SpectralDecomposition` | `spectral` | Frequency-domain Hodge split with power spectrum |
| `ObservabilityGramian` | `gramian` | Gramian as Čech coboundary; observability analysis |
| `GreensOperator` | `greens` | `G = Δ⁻¹` connecting Kalman gain to Hodge inverse |
| `BeliefState` | `belief` | Unified belief with energies and steady-state flag |
| `BeliefEstimator` | `belief` | Top-level estimator integrating Kalman + Hodge + spectral |

### Key Methods

**ObservationBundle**
- `new(n_states, obs_dim, H)` — create a trivial bundle
- `with_fiber(...)` — create with custom fiber metric
- `differential(&state)` — apply observation matrix `Hx` (de Rham `d`)
- `section_from_data(&data)` — wrap observation matrix as a section

**HodgeDecomposition**
- `hodge_decompose(&section)` — full decomposition
- `harmonic_projection(&section)` — project onto `ker(Δ)`
- `is_harmonic(&section, tol)` — check if a section is harmonic
- `decomp.energies()` — `(harmonic, exact, coexact)` L² energies
- `decomp.verify_orthogonality(tol)` — check mutual orthogonality

**KalmanFilter**
- `new(F, H, Q, R, x0, P0)` — construct with system matrices
- `predict()` — forward propagation (analogous to `d`)
- `update(&y)` — measurement correction (analogous to `d*`)
- `step(&y)` — predict + update
- `filter(&[observations])` — run on observation sequence
- `innovation_decomposition(&states)` — Hodge decompose innovation history
- `steady_state_estimate(&states)` — extract harmonic component

**SpectralProjector**
- `new(cutoff)` — manual frequency cutoff
- `auto(n_points)` — automatic 1/3 split
- `decompose(&section)` — full spectral Hodge decomposition
- `extract_harmonic(&section)` — DC component only
- `power_spectrum(&section)` — power at each frequency

**ObservabilityGramian**
- `compute(&F, &H, horizon)` — build Gramian `Σ(F^i)'H'H(F^i)`
- `is_observable()` — full-rank check
- `rank()` / `condition_number()` — numerical analysis
- `cech_coboundary(n)` — build observability matrix
- `reconstruct_state(&observations)` — invert to recover state

**GreensOperator**
- `from_laplacian(&Δ)` — construct by inverting Laplacian
- `from_kalman(&kf)` — construct from Kalman filter
- `apply(&f)` — solve `Δu = f`
- `harmonic_projector()` — `I - GΔ`
- `compose(&other)` — combine for nested estimation

**BeliefEstimator**
- `new(F, H, Q, R, x0, P0, n_states)` — full setup
- `observe(&y) → BeliefState` — single observation update
- `batch_estimate(&[observations]) → Vec<BeliefState>` — run on sequence
- `observability(horizon)` — compute Gramian
- `greens_operator()` — get `G = Δ⁻¹`
- `innovation_decomposition()` — Hodge decomposition of innovations

---

## How It Works

### 1. Observation Bundles (Vector Bundle Geometry)

The state space is a discrete manifold `M` (a finite set of state points). The **observation bundle** `E → M` attaches to each state point a fiber — the vector space of possible observations — equipped with a metric (inner product matrix). The observation matrix `H` acts as the **de Rham differential** `d`, mapping state vectors to observation vectors.

### 2. Hodge Decomposition

Any section `ω ∈ Γ(E)` (observation data) decomposes into three mutually orthogonal parts:

```
ω = ω_harmonic ⊕ ω_exact ⊕ ω_coexact
```

- **Harmonic** (`ker(Δ)`): constant across states — the steady-state Kalman estimate
- **Exact** (`im(d)`): forward differences of the data — the innovation process
- **Coexact** (`im(d*)`): backward differences — uncertainty residuals

The Laplacian `Δ = dd* + d*d` is the discrete second-difference operator. Harmonic sections are in its kernel.

### 3. Kalman Filter as Hodge Projection

The predict-update cycle maps onto the de Rham complex:

- **Predict**: `x̂⁻ = Fx̂`, `P⁻ = FPF' + Q` — push forward via `d`
- **Update**: `x̂ = x̂⁻ + K(y - Hx̂⁻)`, `P = (I - KH)P⁻` — correct via `d*`
- **Gain**: `K = P⁻H'(HP⁻H' + R)⁻¹` — this IS the Green's operator `Δ⁻¹`

The information-form Laplacian is `Δ_info = H'R⁻¹H + P⁻¹`, and `K = Δ_info⁻¹ H'R⁻¹`.

### 4. Spectral Decomposition (FFT)

The harmonic component corresponds to the **zero-frequency (DC)** component in the frequency domain. Low frequencies map to exact forms (smooth innovation), high frequencies map to coexact forms (rough uncertainty). The `SpectralProjector` uses FFT to split observations into these bands.

### 5. Observability as Čech Cohomology

The observability Gramian `W_O = Σ(F^i)'H'H(F^i)` acts as a Čech coboundary: it measures the obstruction to patching local observations into a consistent global state estimate. Full rank means the system is observable and the harmonic space `H⁰(M; E)` is finite-dimensional.

---

## The Math

### Hodge Theory on Bundles

For a vector bundle `E → M` with metric `g` and connection `∇`, the Hodge Laplacian is:

```
Δ = d d* + d* d
```

where `d` is the exterior derivative and `d*` is its formal adjoint (codifferential). The **Hodge decomposition theorem** states:

```
Γ(E) = ker(Δ) ⊕ im(d) ⊕ im(d*)
```

These three summands are orthogonal with respect to the bundle metric.

### Discrete Setting

On a discrete 1D chain of `n` states:

```
(dω)_i = ω_{i+1} - ω_i           (forward difference)
(d*ω)_i = ω_i - ω_{i+1}          (backward difference / adjoint)
(Δω)_i = ω_{i-1} - 2ω_i + ω_{i+1}  (discrete Laplacian)
```

### Kalman–Hodge Connection

The Kalman filter's information form is:

```
Δ_info = H'R⁻¹H + P₀⁻¹          (information Laplacian)
K = Δ_info⁻¹ H'R⁻¹               (Green's operator)
```

The steady-state Kalman gain satisfies the algebraic Riccati equation, which in this framework is the statement that the optimal estimate lies in the harmonic space.

### Spectral Interpretation

The FFT diagonalizes the circulant Laplacian:

```
Δ ↔ diag(λ_k)   where λ_k = 2 - 2cos(2πk/n)
```

- `k = 0`: `λ = 0` (harmonic, DC component)
- `k = 1, ..., cutoff`: low-frequency exact modes
- `k > cutoff`: high-frequency coexact modes

---

## License

MIT
