# tritonadmix/viz/plot.py

import numpy as np
import matplotlib.pyplot as plt


def load_q_matrix(q_path):
    return np.loadtxt(q_path)


def load_population_labels(tsv_path, sample_ids):
    """Load population labels and return labels in sample order."""
    sample_to_pop = {}
    with open(tsv_path) as f:
        f.readline()  # skip header
        for line in f:
            fields = line.strip().split('\t')
            sample_id = fields[0]
            superpop = fields[5] if len(fields) > 5 else "Unknown"
            sample_to_pop[sample_id] = superpop

    return [sample_to_pop.get(sid, "Unknown") for sid in sample_ids]


def load_sample_ids(vcf_path):
    import gzip
    opener = gzip.open if vcf_path.endswith('.gz') else open
    with opener(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#CHROM'):
                fields = line.strip().split('\t')
                return fields[9:]
    raise ValueError("No header found in VCF")


def plot_admixture(Q, output_path=None, population_labels=None, sort_by_population=True,
                   figsize=None, colors=None, title=None, dpi=150):
    """
    Plot ADMIXTURE-style stacked bar chart.

    Q: ancestry matrix (n_individuals, k)
    population_labels: list of population labels for each individual
    sort_by_population: if True, group individuals by population
    """
    n_individuals, k = Q.shape

    if figsize is None:
        width = max(10, n_individuals / 50)
        figsize = (width, 4)

    if colors is None:
        cmap = plt.colormaps['tab10']
        colors = [cmap(i) for i in range(k)]

    # Sort by population if labels provided
    if population_labels is not None and sort_by_population:
        sorted_indices = np.argsort(population_labels)
        Q = Q[sorted_indices]
        population_labels = [population_labels[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=figsize)

    # Stacked bar chart
    x = np.arange(n_individuals)
    bottom = np.zeros(n_individuals)

    for i in range(k):
        ax.bar(x, Q[:, i], bottom=bottom, width=1.0, color=colors[i],
               edgecolor='none', label=f'K{i+1}')
        bottom += Q[:, i]

    ax.set_xlim(-0.5, n_individuals - 0.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Ancestry Proportion')
    ax.set_xlabel('Individuals')

    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Ancestry Proportions (K={k})')

    # Add population labels on x-axis if provided
    if population_labels is not None:
        # Find population boundaries
        pop_boundaries = []
        current_pop = population_labels[0]
        start = 0

        for i, pop in enumerate(population_labels):
            if pop != current_pop:
                pop_boundaries.append((start, i - 1, current_pop))
                start = i
                current_pop = pop
        pop_boundaries.append((start, len(population_labels) - 1, current_pop))

        # Add vertical separators and horizontal brackets with labels
        for start, end, pop in pop_boundaries:
            mid = (start + end) / 2
            # Vertical separator at start
            ax.axvline(x=start - 0.5, color='black', linewidth=1.0)
            # Horizontal bracket line under this population's region
            ax.plot([start - 0.3, end + 0.3], [-0.06, -0.06], color='black',
                    linewidth=1.5, transform=ax.get_xaxis_transform(), clip_on=False)
            # Population label
            ax.text(mid, -0.12, pop, ha='center', va='top', fontsize=9, fontweight='bold',
                    transform=ax.get_xaxis_transform())

        ax.axvline(x=n_individuals - 0.5, color='black', linewidth=1.0)
        ax.set_xticks([])
    else:
        ax.set_xticks([])

    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_cv(results, output_path=None, dpi=150):
    """
    Plot CV error vs K.

    results: dict with 'k', 'mean_error', 'std_error', 'best_k'
    """
    k_vals = results['k']
    mean_err = results['mean_error']
    std_err = results['std_error']
    best_k = results['best_k']

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.errorbar(k_vals, mean_err, yerr=std_err, marker='o', capsize=4,
                linewidth=2, markersize=8, color='steelblue')

    # Highlight best K
    best_idx = k_vals.index(best_k)
    ax.scatter([best_k], [mean_err[best_idx]], color='red', s=150,
               zorder=5, label=f'Best K={best_k}')

    ax.set_xlabel('K (number of populations)', fontsize=12)
    ax.set_ylabel('CV Error', fontsize=12)
    ax.set_title('Cross-Validation Error vs K', fontsize=14)
    ax.set_xticks(k_vals)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()
