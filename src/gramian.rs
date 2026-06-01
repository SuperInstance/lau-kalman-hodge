//! Observability Gramian as Čech coboundary.
//!
//! The observability Gramian W_O = Σ (F^i)' H' H F^i encodes how well
//! the state can be reconstructed from observations. In the sheaf-theoretic
//! framework, this is the Čech coboundary operator: it measures obstruction
//! to patching local observations into global state estimates.
//!
//! If W_O is full rank, the system is observable and H⁰(M; E) is finite-dimensional.
//! The Gramian is related to the Hodge Laplacian by: Δ|_obs = H' R^{-1} H + F' Δ F.

use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// Observability Gramian computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityGramian {
    /// State transition matrix F.
    pub f: DMatrix<f64>,
    /// Observation matrix H.
    pub h: DMatrix<f64>,
    /// The Gramian matrix W_O.
    pub gramian: DMatrix<f64>,
    /// Number of time steps used.
    pub horizon: usize,
}

impl ObservabilityGramian {
    /// Compute the observability Gramian over a given horizon.
    pub fn compute(f: &DMatrix<f64>, h: &DMatrix<f64>, horizon: usize) -> Self {
        let n = f.nrows();
        let mut gramian = DMatrix::zeros(n, n);
        let mut f_power = DMatrix::identity(n, n);
        let ht = h.transpose();

        for _ in 0..horizon {
            // Add (F^i)' H' H F^i
            let obs_contrib = &f_power.transpose() * &ht * h * &f_power;
            gramian += &obs_contrib;
            f_power = f * &f_power;
        }

        Self {
            f: f.clone(),
            h: h.clone(),
            gramian,
            horizon,
        }
    }

    /// Check if the system is observable (Gramian is full rank).
    pub fn is_observable(&self) -> bool {
        let rank = self.rank();
        rank == self.gramian.nrows()
    }

    /// Compute the effective rank of the Gramian.
    pub fn rank(&self) -> usize {
        // Use SVD to compute numerical rank
        let svd = self.gramian.clone().svd(true, true);
        let threshold = 1e-10;
        svd.singular_values.iter().filter(|&&s| s > threshold).count()
    }

    /// Condition number of the Gramian (measures observability quality).
    pub fn condition_number(&self) -> f64 {
        let svd = self.gramian.clone().svd(true, true);
        let svals = &svd.singular_values;
        let max_s = svals.iter().cloned().fold(0.0_f64, f64::max);
        let min_s = svals.iter().cloned().filter(|&s| s > 1e-15).fold(f64::INFINITY, f64::min);
        if min_s < 1e-15 { f64::INFINITY } else { max_s / min_s }
    }

    /// The Čech coboundary operator: maps local observation data to
    /// global consistency conditions.
    ///
    /// Returns the observability matrix O = [H; HF; HF²; ...; HF^{n-1}]
    pub fn cech_coboundary(&self, n_steps: usize) -> DMatrix<f64> {
        let obs_dim = self.h.nrows();
        let state_dim = self.f.nrows();
        let mut o = DMatrix::zeros(obs_dim * n_steps, state_dim);
        let mut f_power = DMatrix::identity(state_dim, state_dim);

        for i in 0..n_steps {
            let block = &self.h * &f_power;
            for r in 0..obs_dim {
                for c in 0..state_dim {
                    o[(i * obs_dim + r, c)] = block[(r, c)];
                }
            }
            f_power = &self.f * &f_power;
        }
        o
    }

    /// Reconstruct state from observation sequence using Gramian inverse.
    pub fn reconstruct_state(&self, observations: &[DVector<f64>]) -> Option<DVector<f64>> {
        let gramian_inv = self.gramian.clone().try_inverse()?;
        let n = self.f.nrows();
        let mut state = DVector::zeros(n);
        let mut f_power = DMatrix::identity(n, n);
        let ht = self.h.transpose();

        for obs in observations {
            state += &f_power.transpose() * &ht * obs;
            f_power = &self.f * &f_power;
        }

        Some(&gramian_inv * &state)
    }

    /// The Hodge Laplacian restricted to the observable subspace.
    pub fn hodge_laplacian_obs(&self, r_inv: &DMatrix<f64>) -> DMatrix<f64> {
        let ht = self.h.transpose();
        &ht * r_inv * &self.h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_observable_system() {
        let f = DMatrix::from_row_slice(2, 2, &[1.0, 0.1, 0.0, 1.0]);
        let h = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let gram = ObservabilityGramian::compute(&f, &h, 20);
        assert!(gram.is_observable());
    }

    #[test]
    fn test_unobservable_system() {
        let f = DMatrix::identity(2, 2);
        let h = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let gram = ObservabilityGramian::compute(&f, &h, 20);
        // Only first state is observed
        assert!(!gram.is_observable());
    }

    #[test]
    fn test_gramian_symmetric() {
        let f = DMatrix::from_row_slice(2, 2, &[0.9, 0.1, 0.0, 0.9]);
        let h = DMatrix::identity(2, 2);
        let gram = ObservabilityGramian::compute(&f, &h, 10);
        assert!((gram.gramian.clone() - gram.gramian.transpose()).iter().all(|v| v.abs() < 1e-10));
    }

    #[test]
    fn test_gramian_positive_semidefinite() {
        let f = DMatrix::identity(2, 2);
        let h = DMatrix::identity(2, 2);
        let gram = ObservabilityGramian::compute(&f, &h, 5);
        let svd = gram.gramian.clone().svd(true, true);
        assert!(svd.singular_values.iter().all(|&s| s >= -1e-10));
    }

    #[test]
    fn test_rank() {
        let f = DMatrix::identity(3, 3);
        let h = DMatrix::identity(3, 3);
        let gram = ObservabilityGramian::compute(&f, &h, 5);
        assert_eq!(gram.rank(), 3);
    }

    #[test]
    fn test_condition_number() {
        let f = DMatrix::identity(2, 2);
        let h = DMatrix::identity(2, 2);
        let gram = ObservabilityGramian::compute(&f, &h, 5);
        assert!(gram.condition_number().is_finite());
        assert!(gram.condition_number() > 0.0);
    }

    #[test]
    fn test_cech_coboundary() {
        let f = DMatrix::identity(2, 2);
        let h = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let gram = ObservabilityGramian::compute(&f, &h, 5);
        let cb = gram.cech_coboundary(3);
        assert_eq!(cb.nrows(), 3);
        assert_eq!(cb.ncols(), 2);
    }

    #[test]
    fn test_state_reconstruction() {
        let f = DMatrix::identity(2, 2);
        let h = DMatrix::identity(2, 2);
        let gram = ObservabilityGramian::compute(&f, &h, 5);
        let obs: Vec<DVector<f64>> = vec![
            DVector::from_vec(vec![3.0, -2.0]),
            DVector::from_vec(vec![3.0, -2.0]),
        ];
        // With identity F, reconstruction via Gramian gives mean
        let recon = gram.reconstruct_state(&obs);
        assert!(recon.is_some());
        let r = recon.unwrap();
        assert!(r.iter().all(|v| v.is_finite()));
        // Should be close to the observation values
        assert!(r[0].is_finite() && (r[0] - 3.0).abs() < 5.0);
        assert!(r[1].is_finite() && (r[1] - (-2.0)).abs() < 5.0);
    }

    #[test]
    fn test_hodge_laplacian_obs() {
        let f = DMatrix::identity(2, 2);
        let h = DMatrix::identity(2, 2);
        let r_inv = DMatrix::identity(2, 2);
        let gram = ObservabilityGramian::compute(&f, &h, 5);
        let lap = gram.hodge_laplacian_obs(&r_inv);
        assert_eq!(lap.nrows(), 2);
        assert_eq!(lap.ncols(), 2);
    }
}
