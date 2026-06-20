//! turbovec_rs — vector indexer STUB (ATANOR `atanor-hologram-core`).
//!
//! Planned role (PRD §3.3 / §8): a fast Rust vector index, exposed to Python
//! via pyo3 + maturin, implementing the `VectorIndexPort` contract so the
//! engine can cull Gaussians by query relevance.
//!
//! This is a stub. No pyo3 bindings yet — uncomment the pyo3 dependency in
//! Cargo.toml and the `#[pymodule]` block below when wiring for real.

/// Squared L2 distance between two equal-length vectors.
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Brute-force top-k nearest indices (placeholder for a real ANN index).
pub fn top_k(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut scored: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2_sq(query, v)))
        .collect();
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    scored.into_iter().take(k).map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn top_k_picks_nearest() {
        let vecs = vec![vec![0.0, 0.0], vec![1.0, 1.0], vec![5.0, 5.0]];
        assert_eq!(top_k(&[0.0, 0.0], &vecs, 2), vec![0, 1]);
    }
}

// --- pyo3 wiring (planned; needs `pyo3` dep + maturin) -----------------------
// use pyo3::prelude::*;
//
// #[pymodule]
// fn turbovec_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
//     // expose build()/query() to satisfy VectorIndexPort
//     Ok(())
// }
