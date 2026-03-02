# tests/test_admixture.py

import numpy as np
from tritonadmix.models.admixture import (
    initialize, compute_p, log_likelihood, e_step, m_step, run_admixture
)


def test_initialize():
    Q, F = initialize(10, 100, 3, seed=42)

    assert Q.shape == (10, 3)
    assert F.shape == (3, 100)

    # Q rows sum to 1
    assert np.allclose(Q.sum(axis=1), 1.0)

    # F values in [0.01, 0.99]
    assert np.all(F >= 0.01) and np.all(F <= 0.99)


def test_compute_p():
    Q = np.array([[0.5, 0.5], [1.0, 0.0]])  # (2 individuals, 2 pops)
    F = np.array([[0.2, 0.8], [0.6, 0.4]])  # (2 pops, 2 snps)

    P = compute_p(Q, F)  # (2 individuals, 2 snps)

    assert P.shape == (2, 2)
    # Individual 0: 0.5*0.2 + 0.5*0.6 = 0.4, 0.5*0.8 + 0.5*0.4 = 0.6
    assert np.allclose(P[0], [0.4, 0.6])
    # Individual 1: 1.0*0.2 + 0.0*0.6 = 0.2, 1.0*0.8 + 0.0*0.4 = 0.8
    assert np.allclose(P[1], [0.2, 0.8])


def test_log_likelihood():
    G = np.array([[0, 2], [1, 1]])  # (2 individuals, 2 snps)
    Q = np.array([[0.5, 0.5], [0.5, 0.5]])
    F = np.array([[0.3, 0.7], [0.3, 0.7]])

    ll = log_likelihood(G, Q, F)
    assert np.isfinite(ll)
    assert ll < 0  # log-likelihood should be negative


def test_log_likelihood_missing():
    G = np.array([[0, -1], [1, 1]])  # -1 is missing
    Q = np.array([[0.5, 0.5], [0.5, 0.5]])
    F = np.array([[0.3, 0.7], [0.3, 0.7]])

    ll = log_likelihood(G, Q, F)
    assert np.isfinite(ll)


def test_run_admixture():
    np.random.seed(42)
    G = np.random.randint(0, 3, size=(20, 50)).astype(np.int8)  # (20 individuals, 50 snps)

    Q, F, log_liks, timing = run_admixture(G, k=2, max_iter=20, seed=42, verbose=False)

    assert Q.shape == (20, 2)
    assert F.shape == (2, 50)
    assert len(log_liks) > 0

    # Q rows sum to 1
    assert np.allclose(Q.sum(axis=1), 1.0, atol=1e-5)

    # F values in [0, 1]
    assert np.all(F >= 0) and np.all(F <= 1)

    # Log-likelihood should generally increase (or stay same)
    for i in range(1, len(log_liks)):
        assert log_liks[i] >= log_liks[i-1] - 1e-6

    # Timing dict has expected keys
    assert 'total' in timing
    assert 'estep' in timing
    assert 'mstep' in timing
    assert timing['n_iters'] > 0
