# tests/test_cli.py

import os
import tempfile
from click.testing import CliRunner
from tritonadmix.cli import main


def test_help_menu():
    runner = CliRunner()
    result = runner.invoke(main, ['--help'])
    assert result.exit_code == 0
    assert "TritonAdmix" in result.output
    assert "run" in result.output
    assert "plot" in result.output


def test_run_help():
    runner = CliRunner()
    result = runner.invoke(main, ['run', '--help'])
    assert result.exit_code == 0
    assert "--vcf" in result.output
    assert "--output-dir" in result.output


def test_plot_help():
    runner = CliRunner()
    result = runner.invoke(main, ['plot', '--help'])
    assert result.exit_code == 0
    assert "--q-matrix" in result.output


def test_admixture_run():
    runner = CliRunner()
    vcf_path = os.path.join(os.path.dirname(__file__), 'data', 'test.vcf')

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, 'output')
        result = runner.invoke(main, [
            'run',
            '--vcf', vcf_path,
            '-k', '2',
            '-o', output_dir,
            '--max-iter', '10',
            '--seed', '42'
        ])

        assert result.exit_code == 0, f"CLI failed: {result.output}"

        q_path = os.path.join(output_dir, 'test.2.Q')
        p_path = os.path.join(output_dir, 'test.2.P')

        assert os.path.exists(q_path), f"Q file not found: {q_path}"
        assert os.path.exists(p_path), f"P file not found: {p_path}"

        # Check Q file format (5 individuals, 2 columns)
        with open(q_path) as f:
            lines = f.readlines()
            assert len(lines) == 5
            for line in lines:
                values = [float(x) for x in line.strip().split()]
                assert len(values) == 2
                assert abs(sum(values) - 1.0) < 0.01

        # Check P file format (10 SNPs, 2 columns)
        with open(p_path) as f:
            lines = f.readlines()
            assert len(lines) == 10
            for line in lines:
                values = [float(x) for x in line.strip().split()]
                assert len(values) == 2
                assert all(0 <= v <= 1 for v in values)


def test_plot_output():
    runner = CliRunner()
    vcf_path = os.path.join(os.path.dirname(__file__), 'data', 'test.vcf')

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = os.path.join(tmpdir, 'output')
        runner.invoke(main, [
            'run',
            '--vcf', vcf_path,
            '-k', '2',
            '-o', output_dir,
            '--max-iter', '10',
            '--seed', '42'
        ])

        q_path = os.path.join(output_dir, 'test.2.Q')
        plot_path = os.path.join(tmpdir, 'test_plot.png')

        result = runner.invoke(main, [
            'plot',
            '-q', q_path,
            '-o', plot_path
        ])

        assert result.exit_code == 0, f"Plot failed: {result.output}"
        assert os.path.exists(plot_path)


def test_cv_help():
    runner = CliRunner()
    result = runner.invoke(main, ['cv', '--help'])
    assert result.exit_code == 0
    assert "--k-min" in result.output
    assert "--k-max" in result.output
    assert "--folds" in result.output


def test_cv_run():
    runner = CliRunner()
    vcf_path = os.path.join(os.path.dirname(__file__), 'data', 'test.vcf')

    result = runner.invoke(main, [
        'cv',
        '--vcf', vcf_path,
        '--k-min', '2',
        '--k-max', '3',
        '--folds', '2',
        '--max-iter', '5',
        '--seed', '42'
    ])

    assert result.exit_code == 0, f"CV failed: {result.output}"
    assert "Optimal K" in result.output
    assert "CV error" in result.output
