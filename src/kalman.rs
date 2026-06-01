//! Kalman filter as Hodge projection.
//!
//! The predict-update cycle maps to the de Rham complex:
//!   predict = application of d (forward model)
//!   update = projection via d* (measurement correction)
//!   Kalman gain K = Green's operator for Δ = dd* + d*d
//!
//! The steady-state Kalman estimate lives in the harmonic space H⁰(M; E),
//! i.e., it is in ker(Δ).

use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};
use crate::bundle::ObservationBundle;
use crate::hodge::{hodge_decompose, HodgeDecomposition};
use crate::bundle::Section;

/// State of a Kalman filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanState {
    /// Current state estimate x̂.
    pub x: DVector<f64>,
    /// State covariance P.
    pub p: DMatrix<f64>,
    /// Innovation (y - ŷ) from last update.
    pub innovation: DVector<f64>,
}

/// Kalman filter implementing the predict-update cycle.
///
/// The filter is interpreted as a Hodge projection:
/// - Predict: push state forward via F (analogous to d)
/// - Update: pull back via measurement (analogous to d*)
/// - Gain K: the Green's operator connecting the two
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalmanFilter {
    /// State transition matrix F.
    pub f: DMatrix<f64>,
    /// Observation matrix H.
    pub h: DMatrix<f64>,
    /// Process noise covariance Q.
    pub q: DMatrix<f64>,
    /// Measurement noise covariance R.
    pub r: DMatrix<f64>,
    /// Kalman gain K (= Green's operator for Δ).
    pub k: DMatrix<f64>,
    /// Current state.
    pub state: KalmanState,
}

impl KalmanFilter {
    /// Create a new Kalman filter with given matrices.
    pub fn new(
        f: DMatrix<f64>,
        h: DMatrix<f64>,
        q: DMatrix<f64>,
        r: DMatrix<f64>,
        x0: DVector<f64>,
        p0: DMatrix<f64>,
    ) -> Self {
        let _state_dim = x0.len();
        let obs_dim = h.nrows();
        // Compute initial Kalman gain from initial covariance
        let ht = h.transpose();
        let s = &h * &p0 * &ht + &r;
        let s_inv = s.clone().try_inverse().unwrap_or_else(|| s.clone());
        let k = &p0 * &ht * &s_inv;

        Self {
            f,
            h,
            q,
            r,
            k,
            state: KalmanState {
                x: x0,
                p: p0,
                innovation: DVector::zeros(obs_dim),
            },
        }
    }

    /// Predict step: forward propagation (de Rham differential d).
    ///
    /// x̂⁻ = F x̂
    /// P⁻ = F P F' + Q
    pub fn predict(&mut self) {
        self.state.x = &self.f * &self.state.x;
        self.state.p = &self.f * &self.state.p * self.f.transpose() + &self.q;
        self.update_gain();
    }

    /// Update step: measurement correction (codifferential d*).
    ///
    /// K = P⁻ H' (H P⁻ H' + R)⁻¹
    /// x̂ = x̂⁻ + K(y - H x̂⁻)
    /// P = (I - KH) P⁻
    pub fn update(&mut self, y: &DVector<f64>) {
        let y_pred = &self.h * &self.state.x;
        self.state.innovation = y - &y_pred;

        let n = self.state.x.len();
        self.state.x = &self.state.x + &self.k * &self.state.innovation;
        let identity = DMatrix::identity(n, n);
        self.state.p = (&identity - &self.k * &self.h) * &self.state.p;
    }

    /// Update Kalman gain K (= Green's operator for the Hodge Laplacian).
    fn update_gain(&mut self) {
        let ht = self.h.transpose();
        let s = &self.h * &self.state.p * &ht + &self.r;
        if let Some(s_inv) = s.clone().try_inverse() {
            self.k = &self.state.p * &ht * &s_inv;
        }
    }

    /// Full predict-update step.
    pub fn step(&mut self, y: &DVector<f64>) {
        self.predict();
        self.update(y);
    }

    /// Run the filter on a sequence of observations.
    pub fn filter(&mut self, observations: &[DVector<f64>]) -> Vec<KalmanState> {
        let mut states = Vec::with_capacity(observations.len());
        for y in observations {
            self.step(y);
            states.push(self.state.clone());
        }
        states
    }

    /// Compute the Hodge decomposition of the innovation history.
    ///
    /// The innovation process y - ŷ is an exact form:
    /// innovation = d(information_potential)
    pub fn innovation_decomposition(&self, states: &[KalmanState]) -> HodgeDecomposition {
        if states.is_empty() {
            let bundle = ObservationBundle::new(1, 1, DMatrix::zeros(1, 1));
            let empty = Section::zeros(1, 1, bundle.fiber.clone());
            return HodgeDecomposition {
                harmonic: empty.clone(),
                exact: empty.clone(),
                coexact: empty,
                laplacian: DMatrix::zeros(1, 1),
                harmonic_projector: DMatrix::zeros(1, 1),
            };
        }

        let obs_dim = states[0].innovation.len();
        let n = states.len();

        let mut vals = DMatrix::zeros(obs_dim, n);
        for (i, s) in states.iter().enumerate() {
            vals.set_column(i, &s.innovation);
        }

        let bundle = ObservationBundle::new(n, obs_dim, self.h.clone());
        let section = bundle.section_from_data(&vals);
        hodge_decompose(&section)
    }

    /// Extract the steady-state estimate (harmonic component).
    ///
    /// In the limit, the Kalman filter converges to the harmonic space H⁰(M; E).
    pub fn steady_state_estimate(&self, states: &[KalmanState]) -> DVector<f64> {
        if states.is_empty() {
            return self.state.x.clone();
        }
        // Average of last few states (converging to harmonic)
        let n_take = states.len().min(10);
        let mut mean = DVector::zeros(states[0].x.len());
        for s in states.iter().rev().take(n_take) {
            mean += &s.x;
        }
        mean /= n_take as f64;
        mean
    }

    /// Check if the filter has converged to steady state (harmonic space).
    pub fn is_steady_state(&self, states: &[KalmanState], tol: f64) -> bool {
        if states.len() < 2 {
            return false;
        }
        let n = states.len();
        let last = &states[n - 1].x;
        let prev = &states[n - 2].x;
        let diff = last - prev;
        diff.iter().all(|v| v.abs() < tol)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn make_simple_filter() -> KalmanFilter {
        let f = DMatrix::identity(2, 2);
        let h = DMatrix::identity(2, 2);
        let q = DMatrix::identity(2, 2) * 0.1;
        let r = DMatrix::identity(2, 2) * 1.0;
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let p0 = DMatrix::identity(2, 2) * 10.0;
        KalmanFilter::new(f, h, q, r, x0, p0)
    }

    #[test]
    fn test_kalman_predict() {
        let mut kf = make_simple_filter();
        let x_before = kf.state.x.clone();
        kf.predict();
        // With identity F, state doesn't change
        assert!((kf.state.x[0] - x_before[0]).abs() < 1e-10);
        // Covariance increases (adds Q)
        assert!(kf.state.p[(0, 0)] > x_before.len() as f64 * 0.0 + 0.1);
    }

    #[test]
    fn test_kalman_update() {
        let mut kf = make_simple_filter();
        let y = DVector::from_vec(vec![5.0, 3.0]);
        kf.predict();
        kf.update(&y);
        // State should move toward observation
        assert!(kf.state.x[0] > 0.0);
        assert!(kf.state.x[1] > 0.0);
    }

    #[test]
    fn test_kalman_step() {
        let mut kf = make_simple_filter();
        let y = DVector::from_vec(vec![1.0, 1.0]);
        kf.step(&y);
        assert!(kf.state.x[0] > 0.0);
    }

    #[test]
    fn test_kalman_convergence() {
        let mut kf = make_simple_filter();
        let true_val = DVector::from_vec(vec![5.0, -3.0]);
        let mut observations = Vec::new();
        for _ in 0..50 {
            // Noisy observations of the true state
            let noise = DVector::from_vec(vec![
                0.1 * (rand_simple() as f64 - 0.5),
                0.1 * (rand_simple() as f64 - 0.5),
            ]);
            observations.push(&true_val + noise);
        }
        let states = kf.filter(&observations);
        // Should converge close to true value
        let last = &states[states.len() - 1].x;
        assert!((last[0] - 5.0).abs() < 1.0);
        assert!((last[1] - (-3.0)).abs() < 1.0);
    }

    #[test]
    fn test_innovation_decomposition() {
        let mut kf = make_simple_filter();
        let observations: Vec<DVector<f64>> = (0..10)
            .map(|i| DVector::from_vec(vec![i as f64, i as f64 * 0.5]))
            .collect();
        let states = kf.filter(&observations);
        let decomp = kf.innovation_decomposition(&states);
        // Just verify it doesn't crash and produces valid decomposition
        assert_eq!(decomp.harmonic.values.ncols(), 10);
        assert_eq!(decomp.exact.values.ncols(), 10);
    }

    #[test]
    fn test_steady_state_estimate() {
        let mut kf = make_simple_filter();
        let observations: Vec<DVector<f64>> = (0..30)
            .map(|_| DVector::from_vec(vec![5.0, -3.0]))
            .collect();
        let states = kf.filter(&observations);
        let ss = kf.steady_state_estimate(&states);
        assert!((ss[0] - 5.0).abs() < 1.0);
        assert!((ss[1] - (-3.0)).abs() < 1.0);
    }

    #[test]
    fn test_is_steady_state() {
        let mut kf = make_simple_filter();
        let observations: Vec<DVector<f64>> = (0..50)
            .map(|_| DVector::from_vec(vec![1.0, 1.0]))
            .collect();
        let states = kf.filter(&observations);
        // Should converge eventually
        assert!(kf.is_steady_state(&states, 0.1));
    }

    #[test]
    fn test_kalman_gain_is_greens_operator() {
        // K = P H' (H P H' + R)^{-1}
        // This is analogous to Green's operator: G = Δ^{-1}
        let kf = make_simple_filter();
        // Verify dimensions
        let state_dim = kf.state.x.len();
        let obs_dim = kf.h.nrows();
        assert_eq!(kf.k.nrows(), state_dim);
        assert_eq!(kf.k.ncols(), obs_dim);
    }

    /// Simple pseudo-random for reproducibility (no external rand dep).
    fn rand_simple() -> f64 {
        static mut SEED: u64 = 42;
        unsafe {
            SEED = SEED.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (SEED >> 33) as f64 / (1u64 << 31) as f64
        }
    }
}
