//! # Hodge-Kalman-Spectral Bridge
//!
//! Implements the theorem that the Kalman filter IS the Hodge star operator
//! on the observation bundle. Every observation bundle E → M admits a
//! Hodge decomposition:
//!
//!   Observation = Harmonic ⊕ Exact(noise) ⊕ CoExact(uncertainty)
//!
//! Where:
//! - Harmonic = Kalman steady-state estimate
//! - Exact = innovation process (y - ŷ)
//! - CoExact = model-uncertainty residual

pub mod bundle;
pub mod hodge;
pub mod kalman;
pub mod spectral;
pub mod gramian;
pub mod greens;
pub mod belief;

pub use bundle::ObservationBundle;
pub use hodge::HodgeDecomposition;
pub use kalman::KalmanFilter;
pub use spectral::SpectralProjector;
pub use gramian::ObservabilityGramian;
pub use greens::GreensOperator;
pub use belief::BeliefEstimator;
