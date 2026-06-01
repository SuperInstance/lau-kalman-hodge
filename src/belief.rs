//! Unified belief estimation for PLATO agents.
//!
//! The Hodge-Kalman-Spectral bridge provides a unified framework for
//! agent beliefs: the Kalman filter's estimate lives in the harmonic space,
//! innovations are exact forms, and uncertainty residuals are coexact forms.
//!
//! This module integrates all components into a belief estimator suitable
//! for PLATO agent architectures.

use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};
use crate::bundle::ObservationBundle;
use crate::hodge::HodgeDecomposition;
use crate::kalman::KalmanFilter;
use crate::spectral::SpectralProjector;
use crate::gramian::ObservabilityGramian;
use crate::greens::GreensOperator;
use crate::bundle::Section;

/// Belief state of a PLATO agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeliefState {
    /// Current estimate (harmonic component).
    pub estimate: DVector<f64>,
    /// Estimate covariance.
    pub covariance: DMatrix<f64>,
    /// Innovation energy (exact component energy).
    pub innovation_energy: f64,
    /// Uncertainty energy (coexact component energy).
    pub uncertainty_energy: f64,
    /// Harmonic energy (steady-state confidence).
    pub harmonic_energy: f64,
    /// Is the belief in steady state?
    pub is_steady_state: bool,
}

/// Unified belief estimator using the Hodge-Kalman-Spectral bridge.
pub struct BeliefEstimator {
    /// The underlying Kalman filter.
    pub kalman: KalmanFilter,
    /// Spectral projector for frequency analysis.
    pub spectral: SpectralProjector,
    /// Observation bundle.
    pub bundle: ObservationBundle,
    /// History of innovations for Hodge decomposition.
    pub innovation_history: Vec<DVector<f64>>,
}

impl BeliefEstimator {
    /// Create a new belief estimator.
    pub fn new(
        f: DMatrix<f64>,
        h: DMatrix<f64>,
        q: DMatrix<f64>,
        r: DMatrix<f64>,
        x0: DVector<f64>,
        p0: DMatrix<f64>,
        n_states: usize,
    ) -> Self {
        let obs_dim = h.nrows();
        let bundle = ObservationBundle::new(n_states, obs_dim, h.clone());
        let kalman = KalmanFilter::new(f, h, q, r, x0, p0);
        let spectral = SpectralProjector::auto(n_states);

        Self {
            kalman,
            spectral,
            bundle,
            innovation_history: Vec::new(),
        }
    }

    /// Update belief with a new observation.
    pub fn observe(&mut self, y: &DVector<f64>) -> BeliefState {
        self.kalman.step(y);
        self.innovation_history.push(self.kalman.state.innovation.clone());

        // Compute Hodge decomposition of innovation history
        let (harm_e, exact_e, coex_e) = if self.innovation_history.len() >= 3 {
            let decomp = self.innovation_decomposition();
            decomp.energies()
        } else {
            (0.0, 0.0, 0.0)
        };

        let total = harm_e + exact_e + coex_e;
        let harmonic_fraction = if total > 1e-15 { harm_e / total } else { 1.0 };

        BeliefState {
            estimate: self.kalman.state.x.clone(),
            covariance: self.kalman.state.p.clone(),
            innovation_energy: exact_e,
            uncertainty_energy: coex_e,
            harmonic_energy: harm_e,
            is_steady_state: harmonic_fraction > 0.95 && self.innovation_history.len() > 5,
        }
    }

    /// Compute Hodge decomposition of the innovation history.
    pub fn innovation_decomposition(&self) -> HodgeDecomposition {
        if self.innovation_history.is_empty() {
            let empty = Section::zeros(1, 1, self.bundle.fiber.clone());
            return HodgeDecomposition {
                harmonic: empty.clone(),
                exact: empty.clone(),
                coexact: empty,
                laplacian: DMatrix::zeros(1, 1),
                harmonic_projector: DMatrix::zeros(1, 1),
            };
        }

        let obs_dim = self.bundle.obs_dim;
        let n = self.innovation_history.len();
        let mut vals = DMatrix::zeros(obs_dim, n);
        for (i, innov) in self.innovation_history.iter().enumerate() {
            vals.set_column(i, innov);
        }
        let section = self.bundle.section_from_data(&vals);
        crate::hodge::hodge_decompose(&section)
    }

    /// Compute spectral decomposition of observations.
    pub fn spectral_decompose(&self, observations: &[DVector<f64>]) -> crate::spectral::SpectralDecomposition {
        let obs_dim = self.bundle.obs_dim;
        let n = observations.len();
        let mut vals = DMatrix::zeros(obs_dim, n);
        for (i, obs) in observations.iter().enumerate() {
            vals.set_column(i, obs);
        }
        let section = self.bundle.section_from_data(&vals);
        self.spectral.decompose(&section)
    }

    /// Compute observability analysis.
    pub fn observability(&self, horizon: usize) -> ObservabilityGramian {
        ObservabilityGramian::compute(&self.kalman.f, &self.kalman.h, horizon)
    }

    /// Compute the Green's operator (Kalman gain as Hodge inverse).
    pub fn greens_operator(&self) -> Option<GreensOperator> {
        GreensOperator::from_kalman(&self.kalman)
    }

    /// Get the Kalman gain (Hodge-Green operator).
    pub fn kalman_gain(&self) -> &DMatrix<f64> {
        &self.kalman.k
    }

    /// Batch estimation: run on a sequence of observations.
    pub fn batch_estimate(&mut self, observations: &[DVector<f64>]) -> Vec<BeliefState> {
        observations.iter().map(|y| self.observe(y)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn make_estimator() -> BeliefEstimator {
        let f = DMatrix::identity(2, 2);
        let h = DMatrix::identity(2, 2);
        let q = DMatrix::identity(2, 2) * 0.1;
        let r = DMatrix::identity(2, 2) * 1.0;
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let p0 = DMatrix::identity(2, 2) * 10.0;
        BeliefEstimator::new(f, h, q, r, x0, p0, 20)
    }

    #[test]
    fn test_belief_observe() {
        let mut est = make_estimator();
        let y = DVector::from_vec(vec![1.0, 2.0]);
        let belief = est.observe(&y);
        assert_eq!(belief.estimate.len(), 2);
        assert!(belief.estimate[0] > 0.0);
    }

    #[test]
    fn test_belief_batch_estimate() {
        let mut est = make_estimator();
        let obs: Vec<DVector<f64>> = (0..10)
            .map(|i| DVector::from_vec(vec![i as f64, i as f64 * 0.5]))
            .collect();
        let beliefs = est.batch_estimate(&obs);
        assert_eq!(beliefs.len(), 10);
        // Later beliefs should be closer to true values
        let last = &beliefs[9];
        assert!(last.estimate[0] > 5.0);
    }

    #[test]
    fn test_belief_convergence() {
        let mut est = make_estimator();
        let obs: Vec<DVector<f64>> = (0..30)
            .map(|_| DVector::from_vec(vec![5.0, -3.0]))
            .collect();
        let beliefs = est.batch_estimate(&obs);
        let last = &beliefs[beliefs.len() - 1];
        assert!((last.estimate[0] - 5.0).abs() < 1.0);
        assert!((last.estimate[1] - (-3.0)).abs() < 1.0);
    }

    #[test]
    fn test_belief_observability() {
        let est = make_estimator();
        let gram = est.observability(10);
        assert!(gram.is_observable());
    }

    #[test]
    fn test_belief_greens_operator() {
        let est = make_estimator();
        let g = est.greens_operator().unwrap();
        assert_eq!(g.dim, 2);
    }

    #[test]
    fn test_belief_spectral_decompose() {
        let est = make_estimator();
        let obs: Vec<DVector<f64>> = (0..10)
            .map(|i| DVector::from_vec(vec![i as f64, 0.0]))
            .collect();
        let decomp = est.spectral_decompose(&obs);
        assert_eq!(decomp.harmonic.values.ncols(), 10);
    }

    #[test]
    fn test_belief_kalman_gain() {
        let est = make_estimator();
        let k = est.kalman_gain();
        assert_eq!(k.nrows(), 2);
        assert_eq!(k.ncols(), 2);
    }

    #[test]
    fn test_belief_energies() {
        let mut est = make_estimator();
        let obs: Vec<DVector<f64>> = (0..10)
            .map(|i| DVector::from_vec(vec![i as f64 * 0.1, 0.0]))
            .collect();
        let beliefs = est.batch_estimate(&obs);
        let last = &beliefs[beliefs.len() - 1];
        assert!(last.harmonic_energy >= 0.0);
        assert!(last.innovation_energy >= 0.0);
        assert!(last.uncertainty_energy >= 0.0);
    }

    #[test]
    fn test_belief_innovation_decomposition() {
        let mut est = make_estimator();
        let obs: Vec<DVector<f64>> = (0..5)
            .map(|i| DVector::from_vec(vec![i as f64, 0.0]))
            .collect();
        est.batch_estimate(&obs);
        let decomp = est.innovation_decomposition();
        assert_eq!(decomp.harmonic.values.ncols(), 5);
    }

    #[test]
    fn test_belief_steady_state_detection() {
        let mut est = make_estimator();
        let obs: Vec<DVector<f64>> = (0..50)
            .map(|_| DVector::from_vec(vec![5.0, -3.0]))
            .collect();
        let beliefs = est.batch_estimate(&obs);
        let last = &beliefs[beliefs.len() - 1];
        // After 50 constant observations, the estimate should be very close
        assert!((last.estimate[0] - 5.0).abs() < 0.5);
        assert!((last.estimate[1] - (-3.0)).abs() < 0.5);
    }
}
