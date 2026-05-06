from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext
import platform

system = platform.system()
if system != "Linux":
    raise RuntimeError(
        "This setup.py is the current Linux build variant. macOS/Windows need platform-specific build settings."
    )

ext_modules = [
    Pybind11Extension(
        "lsmash._lsmash",
        [
            "src/lsmash_api.cpp",
            "src/pybind_module.cpp",
            "vendor/zbase/semantic.cc",
            "vendor/zbase/config.cc",
        ],
        include_dirs=[
            "src",
            "vendor/zbase",
        ],
        library_dirs=[
            "/usr/lib64",
            "/lib64",
            "/usr/lib/x86_64-linux-gnu",
            "/lib/x86_64-linux-gnu",
        ],
        libraries=[
            "boost_program_options",
            "boost_timer",
            "boost_chrono",
            "boost_thread",
            "boost_system",
            "gomp",
            "m",
            "pthread",
        ],
        extra_compile_args=[
            "-O3",
            "-fopenmp",
            "-std=c++17",
            "-fPIC",
        ],
        extra_link_args=[
            "-fopenmp",
            "-Wl,-Bdynamic",
            "-lgsl",
            "-lgslcblas",
        ],
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
