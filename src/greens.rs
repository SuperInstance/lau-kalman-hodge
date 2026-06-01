//! Green's operator for the Hodge Laplacian Δ = dd* + d*d.
//!
//! The Kalman gain K IS the Green's operator G = Δ⁻¹ for the restricted
//! Hodge Laplacian on the observation bundle. This connects estimation
//! theory to Hodge theory:
//!
//!   K = G|_{obs} = (H' R⁻¹ H + P⁻¹)⁻¹ H' R⁻¹
//!
//! The Green's operator inverts the Laplacian to recover the potential
//! from the observed data, yielding the optimal estimate.

use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};
use crate::kalman::KalmanFilter;

/// Green's operator for the Hodge Laplacian.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GreensOperator {
    /// The Green's operator matrix G = Δ⁻¹.
    pub g: DMatrix<f64>,
    /// The Laplacian Δ = dd* + d*d.
    pub laplacian: DMatrix<f64>,
    /// Dimension of the operator.
    pub dim: usize,
}

impl GreensOperator {
    /// Construct from the Laplacian by inversion.
    pub fn from_laplacian(laplacian: &DMatrix<f64>) -> Option<Self> {
        let dim = laplacian.nrows();
        let g = laplacian.clone().try_inverse()?;
        Some(Self {
            g,
            laplacian: laplacian.clone(),
            dim,
        })
    }

    /// Construct from Kalman filter (K = Green's operator).
    ///
    /// The information form Laplacian is:
    ///   Δ_info = H' R⁻¹ H + P⁻¹
    /// And the Green's operator is its inverse.
    pub fn from_kalman(kf: &KalmanFilter) -> Option<Self> {
        let ht = kf.h.transpose();
        let r_inv = kf.r.clone().try_inverse()?;
        let p_inv = kf.state.p.clone().try_inverse()?;

        let laplacian = &ht * &r_inv * &kf.h + &p_inv;
        let g = laplacian.clone().try_inverse()?;

        Some(Self {
            g,
            laplacian,
            dim: kf.state.x.len(),
        })
    }

    /// Apply the Green's operator: solve Δu = f → u = Gf.
    pub fn apply(&self, f: &DVector<f64>) -> DVector<f64> {
        &self.g * f
    }

    /// Verify: ΔG = I (identity).
    pub fn verify_inverse(&self, tol: f64) -> bool {
        let product = &self.laplacian * &self.g;
        let identity = DMatrix::identity(self.dim, self.dim);
        (product - identity).iter().all(|v| v.abs() < tol)
    }

    /// Compute the harmonic projection via Green's operator:
    /// P_harmonic = I - GΔ
    pub fn harmonic_projector(&self) -> DMatrix<f64> {
        let identity = DMatrix::identity(self.dim, self.dim);
        &identity - &self.g * &self.laplacian
    }

    /// Spectral decomposition of the Green's operator.
    pub fn eigenvalues(&self) -> Vec<f64> {
        let svd = self.g.clone().svd(true, true);
        svd.singular_values.iter().map(|s| *s).collect()
    }

    /// Condition number of the Green's operator.
    pub fn condition_number(&self) -> f64 {
        let eigs = self.eigenvalues();
        let max = eigs.iter().cloned().fold(0.0_f64, f64::max);
        let min = eigs.iter().cloned().filter(|&e| e > 1e-15).fold(f64::INFINITY, f64::min);
        if min < 1e-15 { f64::INFINITY } else { max / min }
    }

    /// Compose two Green's operators (for nested estimation).
    pub fn compose(&self, other: &GreensOperator) -> Option<GreensOperator> {
        let combined_laplacian = &self.laplacian + &other.laplacian;
        GreensOperator::from_laplacian(&combined_laplacian)
    }
}

/// Build the discrete Hodge Laplacian for a 1D chain complex of length n.
pub fn discrete_hodge_laplacian(n: usize) -> DMatrix<f64> {
    let mut l = DMatrix::zeros(n, n);
    for i in 0..n {
        l[(i, i)] = 2.0;
        if i > 0 {
            l[(i, i - 1)] = -1.0;
        }
        if i < n - 1 {
            l[(i, i + 1)] = -1.0;
        }
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_greens_from_laplacian() {
        let lap = DMatrix::from_row_slice(3, 3,
            &[2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0]);
        let g = GreensOperator::from_laplacian(&lap).unwrap();
        assert_eq!(g.dim, 3);
        assert!(g.verify_inverse(1e-8));
    }

    #[test]
    fn test_greens_apply() {
        let lap = DMatrix::from_row_slice(2, 2, &[2.0, -1.0, -1.0, 2.0]);
        let g = GreensOperator::from_laplacian(&lap).unwrap();
        let f = DVector::from_vec(vec![1.0, 0.0]);
        let u = g.apply(&f);
        // Verify: Δu = f
        let residual = &lap * &u - f;
        assert!(residual.iter().all(|v| v.abs() < 1e-8));
    }

    #[test]
    fn test_greens_harmonic_projector() {
        let lap = DMatrix::from_row_slice(3, 3,
            &[2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0]);
        let g = GreensOperator::from_laplacian(&lap).unwrap();
        let proj = g.harmonic_projector();
        // P² = P (idempotent)
        let p2 = &proj * &proj;
        assert!((&p2 - &proj).iter().all(|v| v.abs() < 1e-8));
    }

    #[test]
    fn test_greens_from_kalman() {
        let f = DMatrix::identity(2, 2);
        let h = DMatrix::identity(2, 2);
        let q = DMatrix::identity(2, 2) * 0.1;
        let r = DMatrix::identity(2, 2) * 1.0;
        let x0 = DVector::from_vec(vec![0.0, 0.0]);
        let p0 = DMatrix::identity(2, 2) * 10.0;
        let kf = KalmanFilter::new(f, h, q, r, x0, p0);
        let g = GreensOperator::from_kalman(&kf).unwrap();
        assert_eq!(g.dim, 2);
    }

    #[test]
    fn test_greens_eigenvalues() {
        let lap = DMatrix::from_row_slice(2, 2, &[2.0, -1.0, -1.0, 2.0]);
        let g = GreensOperator::from_laplacian(&lap).unwrap();
        let eigs = g.eigenvalues();
        assert_eq!(eigs.len(), 2);
        assert!(eigs.iter().all(|&e| e > 0.0));
    }

    #[test]
    fn test_greens_condition_number() {
        let lap = DMatrix::from_row_slice(2, 2, &[2.0, -1.0, -1.0, 2.0]);
        let g = GreensOperator::from_laplacian(&lap).unwrap();
        let cond = g.condition_number();
        assert!(cond.is_finite());
        assert!(cond >= 1.0);
    }

    #[test]
    fn test_discrete_hodge_laplacian() {
        let l = discrete_hodge_laplacian(4);
        assert_eq!(l.nrows(), 4);
        assert_eq!(l.ncols(), 4);
        // Should be symmetric
        assert_eq!(l, l.transpose());
        // Diagonal should be 2
        for i in 0..4 {
            assert!((l[(i, i)] - 2.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_greens_compose() {
        let lap1 = DMatrix::from_row_slice(2, 2, &[2.0, -1.0, -1.0, 2.0]);
        let lap2 = DMatrix::identity(2, 2);
        let g1 = GreensOperator::from_laplacian(&lap1).unwrap();
        let g2 = GreensOperator::from_laplacian(&lap2).unwrap();
        let combined = g1.compose(&g2).unwrap();
        assert_eq!(combined.dim, 2);
    }
}
