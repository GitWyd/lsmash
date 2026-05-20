from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import platform
import os

system = platform.system()
conda_prefix = os.environ.get('CONDA_PREFIX', '')

if system == 'Darwin':
    extra_compile_args = ['-O3', '-std=c++17', '-fPIC', '-Xpreprocessor', '-fopenmp']
    extra_link_args = []
    openmp_lib = 'omp'    # llvm-openmp (conda-forge / Homebrew libomp)
else:
    extra_compile_args = ['-O3', '-std=c++17', '-fPIC', '-fopenmp']
    extra_link_args = ['-fopenmp', '-Wl,-Bdynamic']
    openmp_lib = 'gomp'   # libgomp shipped with GCC on Linux

library_dirs = []
include_dirs = ['src', 'vendor/zbase']

if conda_prefix:
    library_dirs.insert(0, os.path.join(conda_prefix, 'lib'))
    include_dirs.insert(0, os.path.join(conda_prefix, 'include'))

if system == 'Linux':
    library_dirs.extend([
        '/usr/lib64', '/lib64',
        '/usr/lib/x86_64-linux-gnu', '/lib/x86_64-linux-gnu',
        '/usr/lib/aarch64-linux-gnu', '/lib/aarch64-linux-gnu',
    ])

ext_modules = [
    Pybind11Extension(
        "lsmash._lsmash",
        [
            "src/lsmash_api.cpp",
            "src/pybind_module.cpp",
            "vendor/zbase/semantic.cc",
            "vendor/zbase/config.cc",
        ],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=[
            'boost_program_options',
            'boost_timer',
            'boost_chrono',
            'boost_thread',
            'boost_system',
            'gsl',
            'gslcblas',
            openmp_lib,
            'm',
            'pthread',
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language="c++",
    )
]

setup(
    name="lsmash",
    version="0.1.0",
    description="Python bindings for lsmash",
    packages=["lsmash"],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
