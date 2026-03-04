# tritonadmix/models/cv.py

import numpy as np
from tritonadmix.models.admixture import run_admixture, compute_p


def create_cv_folds(G, n_folds=5, seed=None):
    """
    Create CV fold masks by randomly assigning genotype entries to folds.
    Returns list of boolean masks, one per fold (True = held out).
    """
    if seed is not None:
        np.random.seed(seed)

    n_individuals, n_snps = G.shape
    valid_mask = G >= 0  # only mask valid (non-missing) entries

    # Assign each valid entry to a fold
    valid_indices = np.argwhere(valid_mask)
    n_valid = len(valid_indices)
    fold_assignments = np.random.randint(0, n_folds, size=n_valid)

    # Create mask for each fold
    fold_masks = []
    for fold in range(n_folds):
        mask = np.zeros((n_individuals, n_snps), dtype=bool)
        fold_idx = valid_indices[fold_assignments == fold]
        mask[fold_idx[:, 0], fold_idx[:, 1]] = True
        fold_masks.append(mask)

    return fold_masks


def compute_cv_error(G_true, Q, F, mask):
    """
    Compute prediction error on masked entries.

    G_true: original genotype matrix with true values
    Q, F: learned parameters
    mask: boolean mask indicating held-out entries

    Returns mean squared error on masked entries.
    """
    P = compute_p(Q, F)  # (n_individuals, n_snps)
    P = np.clip(P, 1e-10, 1 - 1e-10)

    # Predicted genotype = 2 * p (expected value under HWE)
    G_pred = 2 * P

    # MSE on masked entries
    errors = (G_pred[mask] - G_true[mask]) ** 2
    return np.mean(errors)


def run_cv_single_k(G, k, n_folds=5, max_iter=100, tol=1e-4, seed=None, verbose=False):
    """
    Run cross-validation for a single K value.
    Returns mean CV error across folds.
    """
    fold_masks = create_cv_folds(G, n_folds=n_folds, seed=seed)
    fold_errors = []

    for fold_idx, mask in enumerate(fold_masks):
        # Create training data: mask held-out entries as missing (-1)
        G_train = G.copy()
        G_train[mask] = -1

        # Train model
        Q, F, _, _ = run_admixture(
            G_train, k=k, max_iter=max_iter, tol=tol,
            seed=seed, verbose=False
        )

        # Compute error on held-out entries
        error = compute_cv_error(G, Q, F, mask)
        fold_errors.append(error)

        if verbose:
            print(f"    Fold {fold_idx + 1}/{n_folds}: error = {error:.4f}")

    return np.mean(fold_errors), np.std(fold_errors)


def run_cv(G, k_min=2, k_max=6, n_folds=5, max_iter=100, tol=1e-4, seed=None, verbose=True):
    """
    Run cross-validation for multiple K values.

    Returns dict with K values, mean errors, and std errors.
    """
    results = {'k': [], 'mean_error': [], 'std_error': []}

    for k in range(k_min, k_max + 1):
        if verbose:
            print(f"  K={k}:")

        mean_err, std_err = run_cv_single_k(
            G, k=k, n_folds=n_folds, max_iter=max_iter, tol=tol,
            seed=seed, verbose=verbose
        )

        results['k'].append(k)
        results['mean_error'].append(mean_err)
        results['std_error'].append(std_err)

        if verbose:
            print(f"    Mean CV error: {mean_err:.4f} (+/- {std_err:.4f})")

    # Find optimal K
    best_idx = np.argmin(results['mean_error'])
    results['best_k'] = results['k'][best_idx]

    return results
