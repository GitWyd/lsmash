# lsmash

Self-contained `pybind11` wrapper for the `lsmash` core.

This source distribution vendors the required `zbase` C++ sources directly into the package, so it does **not** require `ZUTIL_ROOT` or prebuilt `zbase` archives.

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
