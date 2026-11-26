"""Single-objective, sequential Bayesian optimization with a fixed kernel."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import ndtr

@dataclass
class PrecomputedKernelGP:
    observed_indices: np.ndarray
    y_mean: float
    y_scale: float
    cholesky: np.ndarray
    weights: np.ndarray
    kernel: np.ndarray
    signal_variance: float

    def predict(self, candidate_indices: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        candidate_indices = np.asarray(candidate_indices, dtype=int)
        cross_cov = self.signal_variance * self.kernel[
            np.ix_(candidate_indices, self.observed_indices)
        ]
        mean_normalized = cross_cov @ self.weights
        projected = solve_triangular(
            self.cholesky, cross_cov.T, lower=True, check_finite=False
        )
        prior_variance = self.signal_variance * np.diag(
            self.kernel[np.ix_(candidate_indices, candidate_indices)]
        )
        variance_normalized = np.maximum(
            prior_variance - np.einsum("ij,ij->j", projected, projected), 0.0
        )
        mean = mean_normalized * self.y_scale + self.y_mean
        std = np.sqrt(variance_normalized) * self.y_scale
        return mean, std

def _stable_cholesky(matrix: np.ndarray, jitter: float) -> np.ndarray:
    current = max(float(jitter), 0.0)
    identity = np.eye(matrix.shape[0])
    for _ in range(8):
        try:
            return np.linalg.cholesky(matrix + current * identity)
        except np.linalg.LinAlgError:
            current = 1.0e-12 if current == 0.0 else current * 10.0
    raise np.linalg.LinAlgError("Cholesky decomposition failed after jitter escalation.")

def fit_precomputed_gp(
    kernel: np.ndarray,
    targets: Sequence[float],
    observed_indices: Sequence[int],
    *,
    signal_variance: float = 1.0,
    noise_variance: float = 1.0e-3,
    jitter: float = 1.0e-10,
) -> PrecomputedKernelGP:
    observed_indices = np.asarray(observed_indices, dtype=int)
    y_observed = np.asarray(targets, dtype=float)[observed_indices]
    y_mean = float(np.mean(y_observed))
    y_scale = float(np.std(y_observed, ddof=0))
    if y_scale < 1.0e-12:
        y_scale = 1.0
    normalized = (y_observed - y_mean) / y_scale
    covariance = signal_variance * kernel[np.ix_(observed_indices, observed_indices)]
    covariance = covariance + noise_variance * np.eye(len(observed_indices))
    cholesky = _stable_cholesky(covariance, jitter)
    weights = cho_solve((cholesky, True), normalized, check_finite=False)
    return PrecomputedKernelGP(
        observed_indices=observed_indices,
        y_mean=y_mean,
        y_scale=y_scale,
        cholesky=cholesky,
        weights=weights,
        kernel=kernel,
        signal_variance=float(signal_variance),
    )

def expected_improvement(
    mean: Sequence[float],
    std: Sequence[float],
    best_observed: float,
    *,
    maximize: bool = True,
    xi: float = 0.0,
) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    improvement = mean - best_observed if maximize else best_observed - mean
    improvement = improvement - float(xi)
    acquisition = np.zeros_like(mean)
    valid = std > 1.0e-14
    z_score = improvement[valid] / std[valid]
    acquisition[valid] = (
        improvement[valid] * ndtr(z_score)
        + std[valid] * np.exp(-0.5 * z_score**2) / np.sqrt(2.0 * np.pi)
    )
    return acquisition

def run_search(kernel, targets, initial_indices, n_steps=20, method="BO", seed=0):
    """Observe one candidate per step; return selected indices and best-so-far."""
    targets = np.asarray(targets, dtype=float)
    observed = list(map(int, initial_indices))
    if method not in ("BO", "Random"):
        raise ValueError("method must be BO or Random")
    if len(set(observed)) != len(observed) or not observed:
        raise ValueError("Initial indices must be nonempty and unique")
    if n_steps < 0 or len(observed) + n_steps > len(targets):
        raise ValueError("Observation budget exceeds the candidate set")
    rng = np.random.default_rng(seed)
    best = [float(targets[observed].max())]
    for _ in range(n_steps):
        candidates = np.setdiff1d(np.arange(len(targets)), observed)
        if method == "Random":
            selected = int(rng.choice(candidates))
        else:
            model = fit_precomputed_gp(kernel, targets, observed)
            mean, std = model.predict(candidates)
            ei = expected_improvement(mean, std, best[-1])
            selected = int(candidates[np.argmax(ei)])
        observed.append(selected)
        best.append(max(best[-1], float(targets[selected])))
    return np.asarray(observed), np.asarray(best)
