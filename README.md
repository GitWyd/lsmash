# lsmash

# Data Smashing 2.0

This is an implementation of the algorithm reported in "Data Smashing 2.0: Sequence Likelihood (SL) Divergence For Fast Time Series Comparison"

+ https://arxiv.org/abs/1909.12243

+ Huang, Yi, Victor Rotaru, and Ishanu Chattopadhyay. "Sequence likelihood divergence for fast time series comparison." Knowledge and Information Systems 65, no. 7 (2023): 3079-3098.


# Install
```
python -m pip install "lsmash @ git+https://github.com/zeroknowledgediscovery/lsmash.git"

```


## Linux build dependencies

You still need system development libraries available to the compiler/linker:

- Boost (`program_options`, `timer`, `chrono`, `thread`, `system`)
- GSL / GSL CBLAS
- OpenMP runtime

Typical Fedora/RHEL-style packages:

```bash
sudo dnf install boost-devel gsl-devel gcc-c++ libgomp
```

Typical Debian/Ubuntu-style packages:

```bash
sudo apt-get install libboost-all-dev libgsl-dev g++
```

## Build

```bash
python -m pip install -U build
python -m build
```

## Install from source

```bash
python -m pip install .
```

## Usage

```python
import lsmash as ls

opt = ls.LsmashOptions()
opt.data_type = "symbolic"
opt.sae = True

D = ls.from_file("seq.dat", opt)
print(D)
```
