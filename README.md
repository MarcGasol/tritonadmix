# TritonAdmix

Hey! This is the repo for our TritonAdmix CLI tool. We're using `uv` to manage packages because it's way faster and keeps our environments perfectly synced.

Here is the quick-start guide to get your local environment running so we don't break each other's code.

## 1. Get the Environment Set Up

First, make sure you have [uv](https://docs.astral.sh/uv/) installed.

Don't use standard `pip install`. To make sure we are both running the exact same package versions, just pull the repo and run:

```bash
uv sync
```

This reads our uv.lock and pyproject.toml files, creates the .venv folder, and downloads all the exact dependencies we need.

## 2. Activate the Virtual Environment
You have to activate the environment every time you work on the tool.

Mac/Linux:

```Bash
source .venv/bin/activate
```
Windows (PowerShell):

```PowerShell
.venv\Scripts\activate
```
(Note: If Windows yells at you about execution policies, just run ```Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser``` once and try again).

## 3. Install the CLI & Run Tests
To make the tritonadmix command actually work in your terminal while you edit the code, install the package in editable mode:

```Bash
uv pip install -e .
```
To make sure everything is wired up right, run the help command:

```Bash
tritonadmix --help
```
Testing:
We're using pytest to test the CLI parsing and the model logic. Before you push any code, just run:

```Bash
pytest
```
## 4. Usage

**Run ADMIXTURE to get Q and P matrices:**
```bash
tritonadmix run --vcf data/1000G_chr22_pruned.vcf.gz -k 5 -o output/
```

**Run with timing profile:**
```bash
tritonadmix run --vcf data/1000G_chr22_pruned.vcf.gz -k 5 -o output/ --profile
```

This outputs:
- `output/1000G_chr22_pruned.5.Q` — ancestry proportions (n_individuals × k)
- `output/1000G_chr22_pruned.5.P` — allele frequencies (n_snps × k)

**Plot the Q matrix:**
```bash
tritonadmix plot -q output/1000G_chr22_pruned.5.Q
```

**Plot with population labels:**
```bash
tritonadmix plot -q output/1000G_chr22_pruned.5.Q \
    --vcf data/1000G_chr22_pruned.vcf.gz \
    --labels data/igsr_samples.tsv
```

**Full options:**
```bash
tritonadmix run --help
tritonadmix plot --help
```

## 5. How We're Doing Git
The main branch is strictly protected. You literally can't push directly to it.

Whenever you're building a new feature or fixing a bug, branch off main:
```
git checkout -b feature/your-feature-name
```
Write code, test it, and commit.
```
git push -u origin feature/your-feature-name
```
Open a Pull Request (PR) on GitHub so we can review it before it gets merged into main.
