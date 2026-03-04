# tritonadmix/models/admixture.py

import time
import numpy as np


def initialize(n_individuals: int, n_snps: int, k: int, seed: int = None):
    """Initialize Q and F matrices randomly."""
    if seed is not None:
        np.random.seed(seed)

    Q = np.random.dirichlet(np.ones(k), size=n_individuals)  # (n_individuals, k)
    F = np.random.uniform(0.01, 0.99, size=(k, n_snps))  # (k, n_snps)

    return Q, F


def compute_p(Q: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Compute expected allele frequency: P[i,j] = sum_k Q[i,k] * F[k,j]"""
    return Q @ F  # (n_individuals, n_snps)


def log_likelihood(G: np.ndarray, Q: np.ndarray, F: np.ndarray) -> float:
    """Compute log-likelihood under Hardy-Weinberg."""
    P = compute_p(Q, F)
    P = np.clip(P, 1e-10, 1 - 1e-10)

    prob_0 = (1 - P) ** 2
    prob_1 = 2 * P * (1 - P)
    prob_2 = P ** 2

    mask_valid = G >= 0

    log_prob = np.zeros_like(G, dtype=np.float64)
    log_prob[G == 0] = np.log(prob_0[G == 0])
    log_prob[G == 1] = np.log(prob_1[G == 1])
    log_prob[G == 2] = np.log(prob_2[G == 2])

    return np.sum(log_prob[mask_valid])


def e_step(G: np.ndarray, Q: np.ndarray, F: np.ndarray):
    """E-step: compute posterior ancestry assignments."""
    n_individuals, n_snps = G.shape
    k = Q.shape[1]

    Q_expanded = Q[:, :, np.newaxis]  # (n_individuals, k, 1)
    F_expanded = F[np.newaxis, :, :]  # (1, k, n_snps)

    gamma_alt = Q_expanded * F_expanded  # (n_individuals, k, n_snps)
    gamma_ref = Q_expanded * (1 - F_expanded)

    gamma_alt = gamma_alt / (gamma_alt.sum(axis=1, keepdims=True) + 1e-10)
    gamma_ref = gamma_ref / (gamma_ref.sum(axis=1, keepdims=True) + 1e-10)

    gamma_alt = gamma_alt.transpose(0, 2, 1)  # (n_individuals, n_snps, k)
    gamma_ref = gamma_ref.transpose(0, 2, 1)

    return gamma_alt, gamma_ref


def m_step(G: np.ndarray, gamma_alt: np.ndarray, gamma_ref: np.ndarray):
    """M-step: update Q and F based on expected ancestry assignments."""
    n_individuals, n_snps, k = gamma_alt.shape

    mask = G >= 0
    G_safe = np.where(mask, G, 0)

    alt_counts = G_safe[:, :, np.newaxis] * gamma_alt
    ref_counts = (2 - G_safe)[:, :, np.newaxis] * gamma_ref

    alt_counts = alt_counts * mask[:, :, np.newaxis]
    ref_counts = ref_counts * mask[:, :, np.newaxis]

    Q_new = (alt_counts.sum(axis=1) + ref_counts.sum(axis=1))
    Q_new = Q_new / (Q_new.sum(axis=1, keepdims=True) + 1e-10)

    alt_sum = alt_counts.sum(axis=0)
    ref_sum = ref_counts.sum(axis=0)

    F_new = alt_sum / (alt_sum + ref_sum + 1e-10)
    F_new = F_new.T
    F_new = np.clip(F_new, 1e-6, 1 - 1e-6)

    return Q_new, F_new


def em_step(G, Q, F):
    """One complete EM step: E-step followed by M-step."""
    gamma_alt, gamma_ref = e_step(G, Q, F)
    Q_new, F_new = m_step(G, gamma_alt, gamma_ref)
    return Q_new, F_new


def project_Q(Q):
    """Project Q to valid simplex (rows sum to 1, non-negative)."""
    Q = np.maximum(Q, 1e-10)
    Q = Q / Q.sum(axis=1, keepdims=True)
    return Q


def project_F(F):
    """Project F to valid range [1e-6, 1-1e-6]."""
    return np.clip(F, 1e-6, 1 - 1e-6)


def run_admixture_em(G, k, max_iter, tol, seed, verbose):
    """Standard EM algorithm."""
    n_individuals, n_snps = G.shape

    if verbose:
        print(f"Running ADMIXTURE (EM): {n_individuals} individuals, {n_snps} SNPs, K={k}")

    t_start = time.perf_counter()

    t_init_start = time.perf_counter()
    Q, F = initialize(n_individuals, n_snps, k, seed=seed)
    t_init = time.perf_counter() - t_init_start

    t_estep_total = 0.0
    t_mstep_total = 0.0
    t_ll_total = 0.0

    log_liks = []
    prev_ll = -np.inf
    n_iters = 0

    for iteration in range(max_iter):
        n_iters += 1

        t0 = time.perf_counter()
        gamma_alt, gamma_ref = e_step(G, Q, F)
        t_estep_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        Q, F = m_step(G, gamma_alt, gamma_ref)
        t_mstep_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        ll = log_likelihood(G, Q, F)
        t_ll_total += time.perf_counter() - t0

        log_liks.append(ll)

        if verbose and (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}: log-likelihood = {ll:.2f}")

        # Use relative tolerance for convergence
        rel_change = abs(ll - prev_ll) / (abs(ll) + 1e-10)
        if rel_change < tol:
            if verbose:
                print(f"  Converged at iteration {iteration + 1}")
            break

        prev_ll = ll

    t_total = time.perf_counter() - t_start

    timing = {
        'total': t_total,
        'init': t_init,
        'estep': t_estep_total,
        'mstep': t_mstep_total,
        'loglik': t_ll_total,
        'n_iters': n_iters,
    }

    return Q, F, log_liks, timing


def run_admixture_squarem(G, k, max_iter, tol, seed, verbose):
    """SQUAREM-accelerated EM algorithm (Varadhan & Roland 2008)."""
    n_individuals, n_snps = G.shape

    if verbose:
        print(f"Running ADMIXTURE (SQUAREM): {n_individuals} individuals, {n_snps} SNPs, K={k}")

    t_start = time.perf_counter()

    t_init_start = time.perf_counter()
    Q, F = initialize(n_individuals, n_snps, k, seed=seed)
    t_init = time.perf_counter() - t_init_start

    t_em_total = 0.0
    t_ll_total = 0.0

    log_liks = []
    prev_ll = -np.inf
    n_iters = 0

    for iteration in range(max_iter):
        n_iters += 1

        t0 = time.perf_counter()

        # θ_0 = (Q, F)
        Q0, F0 = Q.copy(), F.copy()

        # θ_1 = EM(θ_0)
        Q1, F1 = em_step(G, Q0, F0)

        # θ_2 = EM(θ_1)
        Q2, F2 = em_step(G, Q1, F1)

        # Compute r = θ_1 - θ_0
        rQ = Q1 - Q0
        rF = F1 - F0

        # Compute v = (θ_2 - θ_1) - r = θ_2 - 2*θ_1 + θ_0
        vQ = Q2 - 2 * Q1 + Q0
        vF = F2 - 2 * F1 + F0

        # Compute step length: α = -||r||² / (r · v)
        r_norm_sq = np.sum(rQ ** 2) + np.sum(rF ** 2)
        v_norm_sq = np.sum(vQ ** 2) + np.sum(vF ** 2)

        if v_norm_sq < 1e-10:
            # Near convergence, just use EM result
            Q, F = Q2, F2
        else:
            # SQUAREM step length: bounded between 1 and 5 to prevent overshooting
            alpha = np.sqrt(r_norm_sq / v_norm_sq)
            alpha = max(1.0, min(alpha, 5.0))

            # Extrapolate: θ_new = θ_0 - 2αr + α²v
            Q_new = Q0 - 2 * alpha * rQ + alpha * alpha * vQ
            F_new = F0 - 2 * alpha * rF + alpha * alpha * vF

            # Project back to constraints
            Q_new = project_Q(Q_new)
            F_new = project_F(F_new)

            # Standard SQUAREM: stabilizing EM step (no likelihood check)
            Q, F = em_step(G, Q_new, F_new)

        t_em_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        ll = log_likelihood(G, Q, F)
        t_ll_total += time.perf_counter() - t0

        log_liks.append(ll)

        if verbose and (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}: log-likelihood = {ll:.2f}")

        # Use relative tolerance for convergence
        rel_change = abs(ll - prev_ll) / (abs(ll) + 1e-10)
        if rel_change < tol:
            if verbose:
                print(f"  Converged at iteration {iteration + 1}")
            break

        prev_ll = ll

    t_total = time.perf_counter() - t_start

    timing = {
        'total': t_total,
        'init': t_init,
        'estep': t_em_total,  # Combined EM time
        'mstep': 0.0,
        'loglik': t_ll_total,
        'n_iters': n_iters,
    }

    return Q, F, log_liks, timing


def run_admixture(G: np.ndarray, k: int, max_iter: int = 100, tol: float = 1e-4,
                  seed: int = None, verbose: bool = True, method: str = 'em'):
    """
    Run ADMIXTURE algorithm.

    method: 'em' (standard) or 'squarem' (accelerated)
    Returns Q, F, log_likelihoods, timing_stats
    """
    if method == 'squarem':
        return run_admixture_squarem(G, k, max_iter, tol, seed, verbose)
    else:
        return run_admixture_em(G, k, max_iter, tol, seed, verbose)
