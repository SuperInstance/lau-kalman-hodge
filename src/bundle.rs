//! Observation bundle: vector bundle over agent state manifold.
//!
//! An observation bundle E → M associates to each state x ∈ M the vector space
//! of possible observations. The bundle is equipped with a metric (inner product)
//! and a connection (covariant derivative) that enable Hodge theory.

use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

/// Fiber of the observation bundle at a point in the state manifold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fiber {
    /// Dimension of the fiber (observation space).
    pub dim: usize,
    /// Inner product matrix (metric) on the fiber.
    pub metric: DMatrix<f64>,
}

impl Fiber {
    /// Create a fiber with the standard Euclidean metric.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            metric: DMatrix::identity(dim, dim),
        }
    }

    /// Create a fiber with a custom metric.
    pub fn with_metric(metric: DMatrix<f64>) -> Self {
        let dim = metric.nrows();
        assert_eq!(metric.ncols(), dim, "Metric must be square");
        Self { dim, metric }
    }

    /// Inner product of two vectors in this fiber.
    pub fn inner(&self, a: &DVector<f64>, b: &DVector<f64>) -> f64 {
        (a.transpose() * &self.metric * b)[(0, 0)]
    }

    /// Norm of a vector in this fiber.
    pub fn norm(&self, v: &DVector<f64>) -> f64 {
        self.inner(v, v).sqrt()
    }
}

/// Section of the observation bundle: a smooth assignment of an observation
/// vector to each point in a discrete state space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Section {
    /// Values at each state point. Columns are state points, rows are observation dims.
    pub values: DMatrix<f64>,
    /// The fiber structure.
    pub fiber: Fiber,
}

impl Section {
    /// Create a section from a matrix of values.
    pub fn new(values: DMatrix<f64>, fiber: Fiber) -> Self {
        assert_eq!(values.nrows(), fiber.dim, "Row count must match fiber dim");
        Self { values, fiber }
    }

    /// Create a zero section.
    pub fn zeros(obs_dim: usize, n_states: usize, fiber: Fiber) -> Self {
        Self {
            values: DMatrix::zeros(obs_dim, n_states),
            fiber,
        }
    }

    /// Evaluate at state index i.
    pub fn at(&self, i: usize) -> DVector<f64> {
        self.values.column(i).into_owned()
    }

    /// Set value at state index i.
    pub fn set_at(&mut self, i: usize, v: &DVector<f64>) {
        self.values.set_column(i, v);
    }

    /// L² norm over the base manifold (sum of squared fiber norms).
    pub fn l2_norm(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.values.ncols() {
            let col: DVector<f64> = self.values.column(i).into_owned();
            sum += self.fiber.inner(&col, &col);
        }
        sum.sqrt()
    }

    /// Add two sections (pointwise).
    pub fn add(&self, other: &Section) -> Section {
        Section {
            values: &self.values + &other.values,
            fiber: self.fiber.clone(),
        }
    }

    /// Subtract two sections.
    pub fn sub(&self, other: &Section) -> Section {
        Section {
            values: &self.values - &other.values,
            fiber: self.fiber.clone(),
        }
    }

    /// Scale by a scalar.
    pub fn scale(&self, s: f64) -> Section {
        Section {
            values: self.values.scale(s),
            fiber: self.fiber.clone(),
        }
    }
}

/// The observation bundle E → M.
///
/// M is discretized as a finite set of state points. The bundle attaches a
/// fiber (observation space) to each point, with transition maps encoding
/// the observation model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationBundle {
    /// Dimension of the state manifold M (number of state points).
    pub n_states: usize,
    /// Observation dimension (fiber dimension).
    pub obs_dim: usize,
    /// Fiber at each state point (same for trivial bundles).
    pub fiber: Fiber,
    /// Observation matrix H: maps state → observation at each point.
    /// Stored as obs_dim × state_dim (assumed uniform for simplicity).
    pub observation_matrix: DMatrix<f64>,
}

impl ObservationBundle {
    /// Create a new observation bundle.
    pub fn new(n_states: usize, obs_dim: usize, observation_matrix: DMatrix<f64>) -> Self {
        assert_eq!(observation_matrix.nrows(), obs_dim);
        Self {
            n_states,
            obs_dim,
            fiber: Fiber::new(obs_dim),
            observation_matrix,
        }
    }

    /// Create with a custom fiber metric.
    pub fn with_fiber(n_states: usize, obs_dim: usize, observation_matrix: DMatrix<f64>, fiber: Fiber) -> Self {
        assert_eq!(fiber.dim, obs_dim);
        assert_eq!(observation_matrix.nrows(), obs_dim);
        Self {
            n_states,
            obs_dim,
            fiber,
            observation_matrix,
        }
    }

    /// The de Rham differential d: maps a state section to an observation section.
    /// In discrete setting, this is the pullback via the observation matrix.
    pub fn differential(&self, state: &DVector<f64>) -> DVector<f64> {
        &self.observation_matrix * state
    }

    /// Create a section from raw observation data.
    pub fn section_from_data(&self, data: &DMatrix<f64>) -> Section {
        assert_eq!(data.nrows(), self.obs_dim);
        Section::new(data.clone(), self.fiber.clone())
    }

    /// Create a zero section.
    pub fn zero_section(&self) -> Section {
        Section::zeros(self.obs_dim, self.n_states, self.fiber.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    #[test]
    fn test_fiber_euclidean_metric() {
        let f = Fiber::new(3);
        let v = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        assert_eq!(f.inner(&v, &v), 14.0);
        assert!((f.norm(&v) - 14.0_f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_fiber_custom_metric() {
        let g = DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        let f = Fiber::with_metric(g);
        let v = DVector::from_vec(vec![1.0, 1.0]);
        assert_eq!(f.inner(&v, &v), 5.0);
    }

    #[test]
    fn test_section_create_and_access() {
        let f = Fiber::new(2);
        let vals = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let s = Section::new(vals, f);
        let at1 = s.at(1);
        assert_eq!(at1[0], 2.0);
        assert_eq!(at1[1], 5.0);
    }

    #[test]
    fn test_section_l2_norm() {
        let f = Fiber::new(2);
        let vals = DMatrix::from_row_slice(2, 2, &[3.0, 0.0, 4.0, 0.0]);
        let s = Section::new(vals, f);
        assert!((s.l2_norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_section_add_sub() {
        let f = Fiber::new(2);
        let a = DMatrix::from_row_slice(2, 1, &[1.0, 2.0]);
        let b = DMatrix::from_row_slice(2, 1, &[3.0, 4.0]);
        let sa = Section::new(a, f.clone());
        let sb = Section::new(b, f.clone());
        let sum = sa.add(&sb);
        assert_eq!(sum.at(0)[0], 4.0);
        assert_eq!(sum.at(0)[1], 6.0);
        let diff = sa.sub(&sb);
        assert_eq!(diff.at(0)[0], -2.0);
    }

    #[test]
    fn test_section_scale() {
        let f = Fiber::new(2);
        let vals = DMatrix::from_row_slice(2, 1, &[1.0, 2.0]);
        let s = Section::new(vals, f);
        let scaled = s.scale(3.0);
        assert_eq!(scaled.at(0)[0], 3.0);
        assert_eq!(scaled.at(0)[1], 6.0);
    }

    #[test]
    fn test_observation_bundle_creation() {
        let h = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let bundle = ObservationBundle::new(5, 2, h);
        assert_eq!(bundle.n_states, 5);
        assert_eq!(bundle.obs_dim, 2);
    }

    #[test]
    fn test_bundle_differential() {
        let h = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let bundle = ObservationBundle::new(3, 2, h);
        let state = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let obs = bundle.differential(&state);
        assert_eq!(obs[0], 1.0);
        assert_eq!(obs[1], 2.0);
    }

    #[test]
    fn test_bundle_zero_section() {
        let h = DMatrix::identity(2, 2);
        let bundle = ObservationBundle::new(3, 2, h);
        let zs = bundle.zero_section();
        assert_eq!(zs.values.nrows(), 2);
        assert_eq!(zs.values.ncols(), 3);
        assert_eq!(zs.l2_norm(), 0.0);
    }
}
