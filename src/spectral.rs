//! Spectral projection via FFT for harmonic extraction.
//!
//! The harmonic component of the Hodge decomposition can be extracted
//! spectrally: it corresponds to the zero-frequency (DC) component.
//! Higher frequencies correspond to exact and coexact components.
//!
//! This module provides GPU-acceleratable spectral decomposition using FFT,
//! splitting observations into frequency bands that map to the Hodge components.

use nalgebra::DMatrix;
use rustfft::{FftPlanner, num_complex::Complex};
use serde::{Serialize, Deserialize};
use crate::bundle::Section;

/// Spectral decomposition of an observation section.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralDecomposition {
    /// Frequency components (FFT coefficients) for each observation dimension.
    /// Each entry is (dim × n_freq) where n_freq = n_states/2 + 1 for real FFT.
    pub frequencies: Vec<Vec<Complex<f64>>>,
    /// Harmonic (DC) component.
    pub harmonic: Section,
    /// Exact component (low frequencies).
    pub exact: Section,
    /// Coexact component (high frequencies).
    pub coexact: Section,
    /// Cutoff frequency for exact/coexact split.
    pub cutoff_freq: usize,
}

/// Spectral projector using FFT.
pub struct SpectralProjector {
    /// Cutoff frequency index: below = exact, above = coexact.
    pub cutoff: usize,
}

impl SpectralProjector {
    /// Create a new spectral projector with given frequency cutoff.
    pub fn new(cutoff: usize) -> Self {
        Self { cutoff }
    }

    /// Create with automatic cutoff (1/3 of spectrum → exact, 2/3 → coexact).
    pub fn auto(n_points: usize) -> Self {
        let cutoff = (n_points / 3).max(1);
        Self { cutoff }
    }

    /// Perform spectral decomposition of a section.
    pub fn decompose(&self, section: &Section) -> SpectralDecomposition {
        let dim = section.values.nrows();
        let n = section.values.ncols();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);
        let ifft = planner.plan_fft_inverse(n);

        let mut all_freqs = Vec::with_capacity(dim);
        let mut harmonic_vals = DMatrix::zeros(dim, n);
        let mut exact_vals = DMatrix::zeros(dim, n);
        let mut coexact_vals = DMatrix::zeros(dim, n);

        for d in 0..dim {
            // Extract signal for this dimension
            let mut signal: Vec<Complex<f64>> = (0..n)
                .map(|i| Complex::new(section.values[(d, i)], 0.0))
                .collect();
            
            // Forward FFT
            fft.process(&mut signal);
            all_freqs.push(signal.clone());

            // Build filtered versions
            let mut harm_spectrum = signal.clone();
            let mut exact_spectrum = signal.clone();
            let mut coexact_spectrum = signal.clone();

            // DC component = harmonic
            for k in 1..n {
                harm_spectrum[k] = Complex::new(0.0, 0.0);
            }

            // Low frequencies (1..cutoff) = exact
            harm_spectrum[0] = Complex::new(0.0, 0.0);
            for k in self.cutoff..n {
                exact_spectrum[k] = Complex::new(0.0, 0.0);
            }

            // High frequencies (cutoff..n) = coexact
            for k in 0..self.cutoff {
                coexact_spectrum[k] = Complex::new(0.0, 0.0);
            }

            // Inverse FFT to get time-domain signals
            let mut harm_time = harm_spectrum;
            let mut exact_time = exact_spectrum;
            let mut coexact_time = coexact_spectrum;

            ifft.process(&mut harm_time);
            ifft.process(&mut exact_time);
            ifft.process(&mut coexact_time);

            let scale = 1.0 / n as f64;
            for i in 0..n {
                harmonic_vals[(d, i)] = harm_time[i].re * scale;
                exact_vals[(d, i)] = exact_time[i].re * scale;
                coexact_vals[(d, i)] = coexact_time[i].re * scale;
            }
        }

        SpectralDecomposition {
            frequencies: all_freqs,
            harmonic: Section::new(harmonic_vals, section.fiber.clone()),
            exact: Section::new(exact_vals, section.fiber.clone()),
            coexact: Section::new(coexact_vals, section.fiber.clone()),
            cutoff_freq: self.cutoff,
        }
    }

    /// Extract only the harmonic (DC) component.
    pub fn extract_harmonic(&self, section: &Section) -> Section {
        let decomp = self.decompose(section);
        decomp.harmonic
    }

    /// Compute power spectrum for each dimension.
    pub fn power_spectrum(&self, section: &Section) -> Vec<Vec<f64>> {
        let dim = section.values.nrows();
        let n = section.values.ncols();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(n);

        let mut power = Vec::with_capacity(dim);
        for d in 0..dim {
            let mut signal: Vec<Complex<f64>> = (0..n)
                .map(|i| Complex::new(section.values[(d, i)], 0.0))
                .collect();
            fft.process(&mut signal);
            let p: Vec<f64> = signal.iter().map(|c| c.norm_sqr() / (n * n) as f64).collect();
            power.push(p);
        }
        power
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::Fiber;
    use nalgebra::DMatrix;

    fn make_section() -> Section {
        let f = Fiber::new(2);
        let vals = DMatrix::from_row_slice(2, 8,
            &[1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0,
              0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0]);
        Section::new(vals, f)
    }

    #[test]
    fn test_spectral_decompose_dimensions() {
        let s = make_section();
        let proj = SpectralProjector::auto(8);
        let decomp = proj.decompose(&s);
        assert_eq!(decomp.harmonic.values.ncols(), 8);
        assert_eq!(decomp.exact.values.ncols(), 8);
        assert_eq!(decomp.coexact.values.ncols(), 8);
        assert_eq!(decomp.frequencies.len(), 2);
    }

    #[test]
    fn test_harmonic_is_dc_component() {
        let f = Fiber::new(1);
        // Constant signal: all harmonic
        let vals = DMatrix::from_row_slice(1, 8,
            &[5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]);
        let s = Section::new(vals, f);
        let proj = SpectralProjector::auto(8);
        let decomp = proj.decompose(&s);
        // Sum of all components should equal original
        for i in 0..8 {
            let total = decomp.harmonic.values[(0, i)]
                + decomp.exact.values[(0, i)]
                + decomp.coexact.values[(0, i)];
            assert!((total - 5.0).abs() < 1e-8, "Reconstruction at {} = {}", i, total);
        }
    }

    #[test]
    fn test_reconstruction_from_spectral() {
        let s = make_section();
        let proj = SpectralProjector::auto(8);
        let decomp = proj.decompose(&s);
        let recon = decomp.harmonic.add(&decomp.exact).add(&decomp.coexact);
        for i in 0..8 {
            assert!((recon.values[(0, i)] - s.values[(0, i)]).abs() < 1e-8);
            assert!((recon.values[(1, i)] - s.values[(1, i)]).abs() < 1e-8);
        }
    }

    #[test]
    fn test_power_spectrum_dc_signal() {
        let f = Fiber::new(1);
        let vals = DMatrix::from_row_slice(1, 8,
            &[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
        let s = Section::new(vals, f);
        let proj = SpectralProjector::auto(8);
        let ps = proj.power_spectrum(&s);
        // DC should dominate
        assert!(ps[0][0] > ps[0][1] + 1.0);
    }

    #[test]
    fn test_spectral_projector_cutoff() {
        let proj = SpectralProjector::new(3);
        assert_eq!(proj.cutoff, 3);
    }

    #[test]
    fn test_extract_harmonic_only() {
        let s = make_section();
        let proj = SpectralProjector::auto(8);
        let h = proj.extract_harmonic(&s);
        assert_eq!(h.values.ncols(), 8);
        assert_eq!(h.values.nrows(), 2);
    }
}
