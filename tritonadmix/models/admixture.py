# tritonadmix/models/admixture.py

import time
import numpy as np


def initialize(n_individuals: int, n_snps: int, k: int, seed: int = None):
    """Initialize Q and F matrices randomly"""
    if seed is not None:
        np.random.seed(seed)

    # Q: (n_individuals, k) - random, rows sum to 1
    Q = np.random.dirichlet(np.ones(k), size=n_individuals)

    # F: (k, n_snps) - random uniform in [0.01, 0.99] to avoid boundary issues
    F = np.random.uniform(0.01, 0.99, size=(k, n_snps))

    return Q, F


def compute_p(Q: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    Compute expected allele frequency for each individual at each SNP.
    p_ij = sum_k q_ik * f_kj
    """
    P = Q @ F  # Q: (n_individuals, k), F: (k, n_snps) -> P: (n_individuals, n_snps)
    return P


def log_likelihood(G: np.ndarray, Q: np.ndarray, F: np.ndarray) -> float:
    """
    Compute log-likelihood of observed genotypes given Q and F.
    Assumes Hardy-Weinberg: P(g=0)=(1-p)^2, P(g=1)=2p(1-p), P(g=2)=p^2
    """
    P = compute_p(Q, F)  # (n_individuals, n_snps)

    P = np.clip(P, 1e-10, 1 - 1e-10) # avoid log(0)

    prob_0 = (1 - P) ** 2
    prob_1 = 2 * P * (1 - P)
    prob_2 = P ** 2

    # Select probability based on observed genotype
    # G: (n_individuals, n_snps), values 0/1/2/-1
    mask_valid = G >= 0

    log_prob = np.zeros_like(G, dtype=np.float64)
    log_prob[G == 0] = np.log(prob_0[G == 0])
    log_prob[G == 1] = np.log(prob_1[G == 1])
    log_prob[G == 2] = np.log(prob_2[G == 2])

    # Sum only valid (non-missing) entries
    return np.sum(log_prob[mask_valid])


def e_step(G: np.ndarray, Q: np.ndarray, F: np.ndarray) -> np.ndarray:
    """
    E-step: compute expected ancestry assignment for each allele copy.

    For each individual i, SNP j, and population k:
    gamma_ijk = posterior probability that an allele copy at (i,j) came from population k

    Returns gamma: (n_individuals, n_snps, k)
    """
    n_individuals, n_snps = G.shape
    k = Q.shape[1]

    Q_expanded = Q[:, :, np.newaxis]  # (n_individuals, k, 1)
    F_expanded = F[np.newaxis, :, :]  # (1, k, n_snps)

    gamma_alt = Q_expanded * F_expanded  # (n_individuals, k, n_snps)

    gamma_ref = Q_expanded * (1 - F_expanded)  # (n_individuals, k, n_snps)

    # Normalize
    gamma_alt = gamma_alt / (gamma_alt.sum(axis=1, keepdims=True) + 1e-10)
    gamma_ref = gamma_ref / (gamma_ref.sum(axis=1, keepdims=True) + 1e-10)

    # Transpose to (n_individuals, n_snps, k)
    gamma_alt = gamma_alt.transpose(0, 2, 1)
    gamma_ref = gamma_ref.transpose(0, 2, 1)

    return gamma_alt, gamma_ref


def m_step(G: np.ndarray, gamma_alt: np.ndarray, gamma_ref: np.ndarray):
    """
    M-step: update Q and F based on expected ancestry assignments.
    """
    n_individuals, n_snps, k = gamma_alt.shape

    # Mask for valid genotypes
    mask = G >= 0  # (n_individuals, n_snps)

    G_safe = np.where(mask, G, 0)  # (n_individuals, n_snps)

    alt_counts = G_safe[:, :, np.newaxis] * gamma_alt  # (n_individuals, n_snps, k)

    ref_counts = (2 - G_safe)[:, :, np.newaxis] * gamma_ref  # (n_individuals, n_snps, k)

    alt_counts = alt_counts * mask[:, :, np.newaxis]
    ref_counts = ref_counts * mask[:, :, np.newaxis]

    Q_new = (alt_counts.sum(axis=1) + ref_counts.sum(axis=1))  # (n_individuals, k)
    Q_new = Q_new / (Q_new.sum(axis=1, keepdims=True) + 1e-10)

    alt_sum = alt_counts.sum(axis=0)  # (n_snps, k)
    ref_sum = ref_counts.sum(axis=0)  # (n_snps, k)

    F_new = alt_sum / (alt_sum + ref_sum + 1e-10)  # (n_snps, k)
    F_new = F_new.T  # (k, n_snps)

    # Clip F to avoid boundary issues
    F_new = np.clip(F_new, 1e-6, 1 - 1e-6)

    return Q_new, F_new


def run_admixture(G: np.ndarray, k: int, max_iter: int = 100, tol: float = 1e-4,
                  seed: int = None, verbose: bool = True):
    """
    Run ADMIXTURE EM algorithm.

    G: genotype matrix (n_individuals, n_snps), values 0/1/2/-1
    k: number of ancestral populations
    max_iter: maximum EM iterations
    tol: convergence tolerance for log-likelihood change

    Returns Q, F, log_likelihoods, timing_stats
    """
    n_individuals, n_snps = G.shape

    if verbose:
        print(f"Running ADMIXTURE: {n_individuals} individuals, {n_snps} SNPs, K={k}")

    # Timing accumulators
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

        # E-step
        t0 = time.perf_counter()
        gamma_alt, gamma_ref = e_step(G, Q, F)
        t_estep_total += time.perf_counter() - t0

        # M-step
        t0 = time.perf_counter()
        Q, F = m_step(G, gamma_alt, gamma_ref)
        t_mstep_total += time.perf_counter() - t0

        # Compute log-likelihood
        t0 = time.perf_counter()
        ll = log_likelihood(G, Q, F)
        t_ll_total += time.perf_counter() - t0

        log_liks.append(ll)

        if verbose and (iteration + 1) % 10 == 0:
            print(f"  Iteration {iteration + 1}: log-likelihood = {ll:.2f}")

        # Check convergence
        if abs(ll - prev_ll) < tol:
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
