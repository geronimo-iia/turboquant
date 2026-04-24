// Math helpers for Lloyd-Max codebook generation.
//
// Inspired by turboquant (MIT, https://github.com/abdelstark/turboquant).
// All functions reimplemented from standard numerical methods:
//   - Lanczos approximation for log-gamma
//   - Beasley-Springer-Moro rational approximation for normal inverse CDF
//   - Simpson's rule for numerical integration
//   - Beta marginal PDF of unit-sphere coordinates

use std::f64::consts::PI;

/// Log-gamma via Lanczos approximation (g=7, n=9).
pub(crate) fn lgamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    #[allow(clippy::excessive_precision)]
    const C: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    const G: f64 = 7.0;

    if x < 0.5 {
        return PI.ln() - (PI * x).sin().ln() - lgamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + G + 0.5;
    let mut a = C[0];
    for (i, &ci) in C[1..].iter().enumerate() {
        a += ci / (x + i as f64 + 1.0);
    }
    0.5 * (2.0 * PI).ln() + a.ln() + (x + 0.5) * t.ln() - t
}

/// PDF of the coordinate marginal distribution of a d-dimensional
/// unit-sphere-uniform vector: f(x) ∝ (1 - x²)^((d-3)/2) for |x| < 1.
///
/// For d > 50, approximated by N(0, 1/d).
pub(crate) fn beta_pdf(x: f64, dim: usize) -> f64 {
    if dim < 2 {
        return 0.0;
    }
    if dim > 50 {
        let sigma2 = 1.0 / dim as f64;
        let sigma = sigma2.sqrt();
        return (-x * x / (2.0 * sigma2)).exp() / (sigma * (2.0 * PI).sqrt());
    }
    if x.abs() >= 1.0 {
        return 0.0;
    }
    let exponent = (dim as f64 - 3.0) / 2.0;
    let unnorm = (1.0 - x * x).powf(exponent);
    let log_c = lgamma(dim as f64 / 2.0) - 0.5 * PI.ln() - lgamma((dim as f64 - 1.0) / 2.0);
    log_c.exp() * unnorm
}

/// Normal inverse CDF (probit) via Beasley-Springer-Moro rational approximation.
#[allow(clippy::excessive_precision)]
pub(crate) fn normal_icdf(p: f64) -> f64 {
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Sample from the coordinate marginal of a d-dim unit-sphere vector
/// at the given quantile via inverse CDF.
pub(crate) fn sample_beta_marginal(dim: usize, u: f64) -> f64 {
    if dim > 50 {
        let sigma = (1.0 / dim as f64).sqrt();
        return sigma * normal_icdf(u);
    }
    numerical_icdf(u, dim)
}

/// Bisection-based inverse CDF for the beta marginal.
fn numerical_icdf(u: f64, dim: usize) -> f64 {
    let mut lo = -1.0_f64;
    let mut hi = 1.0_f64;
    for _ in 0..64 {
        let mid = (lo + hi) / 2.0;
        if numerical_cdf(mid, dim) < u {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    (lo + hi) / 2.0
}

/// Numerical CDF of beta_pdf from -1 to x via Simpson's rule.
fn numerical_cdf(x: f64, dim: usize) -> f64 {
    let n = 200usize;
    let a = -0.9999_f64;
    let b = x.min(0.9999);
    if b <= a {
        return 0.0;
    }
    let h = (b - a) / n as f64;
    let mut sum = beta_pdf(a, dim) + beta_pdf(b, dim);
    for i in 1..n {
        let xi = a + i as f64 * h;
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum += w * beta_pdf(xi, dim);
    }
    sum * h / 3.0
}

/// Simpson's rule returning paired integrals (∫f, ∫g) simultaneously.
pub(crate) fn simpson_integrate<F>(a: f64, b: f64, n: usize, f: F) -> (f64, f64)
where
    F: Fn(f64) -> (f64, f64),
{
    let n = if n.is_multiple_of(2) { n } else { n + 1 };
    let h = (b - a) / n as f64;
    let (f0, g0) = f(a);
    let (fn_, gn) = f(b);
    let mut sum_f = f0 + fn_;
    let mut sum_g = g0 + gn;
    for i in 1..n {
        let x = a + i as f64 * h;
        let (fi, gi) = f(x);
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum_f += w * fi;
        sum_g += w * gi;
    }
    (sum_f * h / 3.0, sum_g * h / 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lgamma_known_values() {
        // Γ(1) = 1 → lgamma(1) = 0
        assert!((lgamma(1.0)).abs() < 1e-10);
        // Γ(2) = 1 → lgamma(2) = 0
        assert!((lgamma(2.0)).abs() < 1e-10);
        // Γ(5) = 24 → lgamma(5) = ln(24)
        assert!((lgamma(5.0) - 24.0_f64.ln()).abs() < 1e-8);
        // Γ(0.5) = √π → lgamma(0.5) = ln(√π)
        assert!((lgamma(0.5) - 0.5 * PI.ln()).abs() < 1e-8);
    }

    #[test]
    fn lgamma_negative_returns_inf() {
        assert!(lgamma(0.0).is_infinite());
        assert!(lgamma(-1.0).is_infinite());
    }

    #[test]
    fn beta_pdf_integrates_to_one() {
        for dim in [5, 10, 20] {
            let n = 1000;
            let h = 2.0 / n as f64;
            let sum: f64 = (0..n)
                .map(|i| beta_pdf(-1.0 + (i as f64 + 0.5) * h, dim) * h)
                .sum();
            assert!((sum - 1.0).abs() < 0.05, "dim={dim}: integral={sum}");
        }
    }

    #[test]
    fn beta_pdf_symmetric() {
        for dim in [5, 10, 100] {
            for &x in &[0.1, 0.3, 0.5] {
                let diff = (beta_pdf(x, dim) - beta_pdf(-x, dim)).abs();
                assert!(diff < 1e-10, "dim={dim}, x={x}: diff={diff}");
            }
        }
    }

    #[test]
    fn beta_pdf_dim_below_2_is_zero() {
        assert_eq!(beta_pdf(0.5, 0), 0.0);
        assert_eq!(beta_pdf(0.5, 1), 0.0);
    }

    #[test]
    fn beta_pdf_at_boundary_is_zero() {
        assert_eq!(beta_pdf(1.0, 10), 0.0);
        assert_eq!(beta_pdf(-1.0, 10), 0.0);
    }

    #[test]
    fn beta_pdf_large_dim_gaussian_path() {
        let dim = 100;
        let pdf = beta_pdf(0.0, dim);
        let expected = (dim as f64 / (2.0 * PI)).sqrt();
        assert!(
            (pdf - expected).abs() / expected < 0.1,
            "pdf={pdf}, expected={expected}"
        );
    }

    #[test]
    fn normal_icdf_median_is_zero() {
        assert!(normal_icdf(0.5).abs() < 1e-6);
    }

    #[test]
    fn normal_icdf_one_sigma() {
        assert!((normal_icdf(0.8413) - 1.0).abs() < 0.01);
        assert!((normal_icdf(0.1587) + 1.0).abs() < 0.01);
    }

    #[test]
    fn normal_icdf_symmetric_tails() {
        let lo = normal_icdf(0.001);
        let hi = normal_icdf(0.999);
        assert!((lo + hi).abs() < 0.01);
    }

    #[test]
    fn sample_beta_marginal_large_dim() {
        let dim = 128;
        assert!(sample_beta_marginal(dim, 0.5).abs() < 0.01);
        let lo = sample_beta_marginal(dim, 0.01);
        let hi = sample_beta_marginal(dim, 0.99);
        assert!(lo < 0.0);
        assert!(hi > 0.0);
        assert!((lo + hi).abs() < 0.01);
    }

    #[test]
    fn sample_beta_marginal_small_dim() {
        let dim = 10;
        assert!(sample_beta_marginal(dim, 0.5).abs() < 0.1);
        assert!(sample_beta_marginal(dim, 0.05) < sample_beta_marginal(dim, 0.95));
    }

    #[test]
    fn simpson_integrate_constant() {
        let (f_int, g_int) = simpson_integrate(0.0, 1.0, 100, |_| (1.0, 2.0));
        assert!((f_int - 1.0).abs() < 1e-10);
        assert!((g_int - 2.0).abs() < 1e-10);
    }

    #[test]
    fn simpson_integrate_x_squared() {
        // ∫₀¹ x² dx = 1/3
        let (val, _) = simpson_integrate(0.0, 1.0, 100, |x| (x * x, 0.0));
        assert!((val - 1.0 / 3.0).abs() < 1e-8);
    }
}
