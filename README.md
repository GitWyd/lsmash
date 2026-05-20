# lsmash — Data Smashing 2.0

Python/C++ implementation of the algorithm reported in:

> **"Data Smashing 2.0: Sequence Likelihood (SL) Divergence For Fast Time Series Comparison"**
> Huang, Yi, Victor Rotaru, and Ishanu Chattopadhyay.
> *Knowledge and Information Systems* 65, no. 7 (2023): 3079–3098.
> [arXiv:1909.12243](https://arxiv.org/abs/1909.12243)

---

## Install (PyPI)

```bash
pip install "lsmash @ git+https://github.com/zeroknowledgediscovery/lsmash.git"
```

---

## Usage

```python
import lsmash as ls

opt = ls.LsmashOptions()
opt.data_type = "symbolic"
opt.sae = True

D = ls.from_file("seq.dat", opt)
print(D)
```

See `examples/example.ipynb` for a full walkthrough.

---

## Building from Source

### Linux

Install system build dependencies, then install from source:

**Fedora / RHEL:**
```bash
sudo dnf install boost-devel gsl-devel gcc-c++ libgomp
```

**Debian / Ubuntu:**
```bash
sudo apt-get install libboost-all-dev libgsl-dev g++
```

Then build:
```bash
pip install -e .
```

---

### macOS (Apple Silicon)

The macOS build uses a self-contained [conda](https://docs.conda.io) environment so that
no system-wide packages are modified. All C++ dependencies (Boost, GSL, OpenMP) are
installed into an isolated prefix inside the project directory.

**Prerequisites:** [Homebrew](https://brew.sh). All other dependencies are managed inside
the project directory — nothing is installed system-wide.

If you don't have micromamba:
```bash
brew install micromamba
```

Run all of the following from the **repo root**:

**1. Create the environment** (one-time setup — installs into `.env/` inside the repo):
```bash
micromamba create --prefix ./.env -f environment.yml -y
```

**2. Build the C++ extension:**
```bash
micromamba run -p ./.env pip install -e .
```

**3. Verify:**
```bash
micromamba run -p ./.env python -c "import lsmash; print('ok')"
```

#### Day-to-day development

All commands below should be run from the repo root.

| Task | Command |
|------|---------|
| Run a script | `micromamba run -p ./.env python script.py` |
| Rebuild after C++ edits | `micromamba run -p ./.env pip install -e .` |
| Open JupyterLab | `micromamba run -p ./.env jupyter lab` |
| Interactive shell | `micromamba run -p ./.env python` |
| Remove the environment | `rm -rf ./.env` |

> **Note:** Python-side changes (files under `lsmash/`) take effect immediately without a
> rebuild. Only changes to C++ source (`src/`, `vendor/zbase/`) require re-running
> `pip install -e .`.
>
> This workflow is tested on Apple Silicon (M-series). It should also work on Intel Macs
> with no changes, since all paths are resolved via the conda environment.

---

## API Reference

### `LsmashOptions`

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_type` | `str` | `"symbolic"` | `"symbolic"` or `"continuous"` |
| `data_dir` | `str` | `"row"` | `"row"` (one sample per row) or `"column"` |
| `data_len` | `int` | `10000000` | Maximum data points per sample |
| `partition` | `list` | — | Cut points for discretising continuous data |
| `use_derivative` | `bool` | `False` | Apply discrete derivative before analysis |
| `sae` | `bool` | `True` | Enable self-alignment estimation |
| `num_repeat` | `int` | `20` | Repeats for SAE |
| `depth` | `int` | `8` | SAE depth parameter |

### Functions

```python
lsmash.from_file(path, options)            # → numpy ndarray (dense)
lsmash.from_file_sparse(path, options)     # → dict-of-dicts (sparse)
lsmash.from_sequences(seqs, options)       # → numpy ndarray (dense)
lsmash.from_sequences_sparse(seqs, options)# → dict-of-dicts (sparse)
```
