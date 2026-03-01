# scripts/validate.py
"""Validate TritonAdmix results against known population labels."""

import sys
import numpy as np

def load_q_matrix(q_path):
    """Load Q matrix from ADMIXTURE-format file."""
    return np.loadtxt(q_path)

def load_sample_ids(vcf_path):
    """Extract sample IDs from VCF header."""
    import gzip
    opener = gzip.open if vcf_path.endswith('.gz') else open
    with opener(vcf_path, 'rt') as f:
        for line in f:
            if line.startswith('#CHROM'):
                fields = line.strip().split('\t')
                return fields[9:]  # sample IDs start at column 9
    raise ValueError("No header found in VCF")

def load_population_labels(tsv_path):
    """Load sample -> superpopulation mapping from igsr_samples.tsv."""
    sample_to_pop = {}
    with open(tsv_path) as f:
        header = f.readline()  # skip header
        for line in f:
            fields = line.strip().split('\t')
            sample_id = fields[0]
            superpop = fields[5] if len(fields) > 5 else None  # Superpopulation code
            if superpop:
                sample_to_pop[sample_id] = superpop
    return sample_to_pop

def validate(q_path, vcf_path, tsv_path):
    print(f"Loading Q matrix from {q_path}")
    Q = load_q_matrix(q_path)  # (n_individuals, k)
    n_individuals, k = Q.shape
    print(f"  Shape: {n_individuals} individuals, K={k}")

    print(f"Loading sample IDs from {vcf_path}")
    sample_ids = load_sample_ids(vcf_path)
    print(f"  {len(sample_ids)} samples")

    print(f"Loading population labels from {tsv_path}")
    sample_to_pop = load_population_labels(tsv_path)

    # Match samples to populations
    populations = {}
    for i, sample_id in enumerate(sample_ids):
        pop = sample_to_pop.get(sample_id, "Unknown")
        if pop not in populations:
            populations[pop] = []
        populations[pop].append(i)

    print(f"\nPopulation breakdown:")
    for pop, indices in sorted(populations.items()):
        print(f"  {pop}: {len(indices)} individuals")

    # Compute mean ancestry per population
    print(f"\nMean ancestry proportions by superpopulation:")
    print("-" * (10 + k * 10))
    header = "Pop".ljust(8) + "".join([f"K{i+1}".rjust(9) for i in range(k)])
    print(header)
    print("-" * (10 + k * 10))

    pop_means = {}
    for pop in ["AFR", "AMR", "EAS", "EUR", "SAS"]:
        if pop in populations:
            indices = populations[pop]
            mean_q = Q[indices].mean(axis=0)
            pop_means[pop] = mean_q
            row = pop.ljust(8) + "".join([f"{v:.4f}".rjust(9) for v in mean_q])
            print(row)

    print("-" * (10 + k * 10))

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python validate.py <Q_file> <VCF_file> <igsr_samples.tsv>")
        print("Example: python validate.py output.5.Q data/1000G_chr22_pruned.vcf.gz ~/public/1000Genomes/igsr_samples.tsv")
        sys.exit(1)

    validate(sys.argv[1], sys.argv[2], sys.argv[3])
