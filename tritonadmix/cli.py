# tritonadmix/cli.py

import os
import click

from tritonadmix.io import load_vcf, write_q_matrix, write_p_matrix
from tritonadmix.models.admixture import run_admixture
from tritonadmix.viz.plot import (
    plot_admixture, load_q_matrix, load_sample_ids, load_population_labels
)


@click.group()
def main():
    """TritonAdmix: A pure Python CLI for admixture inference."""
    pass


def print_timing(timing):
    """Print timing summary table."""
    total = timing['total']
    n_iters = timing['n_iters']

    click.echo(click.style("\nTiming Summary:", fg="cyan", bold=True))
    click.echo(f"  Total:           {total:.2f}s")
    click.echo(f"  Initialization:  {timing['init']:.2f}s ({100*timing['init']/total:.1f}%)")
    click.echo(f"  E-step:          {timing['estep']:.2f}s ({100*timing['estep']/total:.1f}%) "
               f"- avg {timing['estep']/n_iters:.3f}s/iter")
    click.echo(f"  M-step:          {timing['mstep']:.2f}s ({100*timing['mstep']/total:.1f}%) "
               f"- avg {timing['mstep']/n_iters:.3f}s/iter")
    click.echo(f"  Log-likelihood:  {timing['loglik']:.2f}s ({100*timing['loglik']/total:.1f}%) "
               f"- avg {timing['loglik']/n_iters:.3f}s/iter")
    click.echo(f"  Iterations:      {n_iters}")


@main.command()
@click.option("--vcf", required=True, type=click.Path(exists=True), help="Path to input VCF file.")
@click.option("-k", "--populations", default=3, type=int, show_default=True, help="Number of ancestral populations (K).")
@click.option("-o", "--output-dir", default="output", type=str, show_default=True, help="Output directory.")
@click.option("--max-iter", default=100, type=int, show_default=True, help="Maximum EM iterations.")
@click.option("--tol", default=1e-4, type=float, show_default=True, help="Convergence tolerance.")
@click.option("--seed", default=None, type=int, help="Random seed for reproducibility.")
@click.option("--profile", is_flag=True, help="Show timing breakdown.")
def run(vcf, populations, output_dir, max_iter, tol, seed, profile):
    """Run ADMIXTURE algorithm on VCF data."""

    click.echo(click.style("TritonAdmix", fg="green", bold=True))

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use input filename as base
    base_name = os.path.splitext(os.path.basename(vcf))[0]
    # Remove .vcf if still present (e.g., from .vcf.gz)
    if base_name.endswith('.vcf'):
        base_name = base_name[:-4]

    click.echo(f"Loading {vcf}...")
    G, sample_ids, variant_ids = load_vcf(vcf)
    click.echo(f"  {len(sample_ids)} individuals, {len(variant_ids)} SNPs")

    Q, F, log_liks, timing = run_admixture(
        G, k=populations, max_iter=max_iter, tol=tol, seed=seed, verbose=True
    )

    q_path = os.path.join(output_dir, f"{base_name}.{populations}.Q")
    p_path = os.path.join(output_dir, f"{base_name}.{populations}.P")

    write_q_matrix(Q, q_path)
    write_p_matrix(F, p_path)

    click.echo(f"Output written to {q_path} and {p_path}")

    if profile:
        print_timing(timing)


@main.command()
@click.option("-q", "--q-matrix", required=True, type=click.Path(exists=True), help="Path to Q matrix file.")
@click.option("-o", "--output", default=None, type=str, help="Output plot path (default: output/<name>.png).")
@click.option("--vcf", default=None, type=click.Path(exists=True), help="VCF file (for sample order).")
@click.option("--labels", default=None, type=click.Path(exists=True), help="Population labels TSV (igsr_samples.tsv format).")
@click.option("--title", default=None, type=str, help="Plot title.")
@click.option("--dpi", default=150, type=int, show_default=True, help="Output resolution.")
def plot(q_matrix, output, vcf, labels, title, dpi):
    """Plot ancestry proportions from Q matrix."""

    click.echo(click.style("TritonAdmix Plot", fg="green", bold=True))

    Q = load_q_matrix(q_matrix)
    click.echo(f"Loaded Q matrix: {Q.shape[0]} individuals, K={Q.shape[1]}")

    # Default output path: output/<q_matrix_name>.png
    if output is None:
        os.makedirs("output", exist_ok=True)
        base_name = os.path.splitext(os.path.basename(q_matrix))[0]
        output = os.path.join("output", f"{base_name}.png")

    population_labels = None
    if vcf and labels:
        sample_ids = load_sample_ids(vcf)
        population_labels = load_population_labels(labels, sample_ids)
        click.echo(f"Loaded population labels for {len(population_labels)} samples")

    plot_admixture(Q, output_path=output, population_labels=population_labels,
                   title=title, dpi=dpi)

    click.echo(f"Plot saved to {output}")


if __name__ == "__main__":
    main()
