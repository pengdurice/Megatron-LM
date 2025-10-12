import subprocess

from setuptools import Extension, setup, find_packages

setup_args = dict(
    name="megatron-lm",
    packages=find_packages(),
    ext_modules=[
        Extension(
            "megatron.core.datasets.helpers_cpp",
            sources=["megatron/core/datasets/helpers.cpp"],
            language="c++",
            extra_compile_args=(
                subprocess.check_output(["python3", "-m", "pybind11", "--includes"])
                .decode("utf-8")
                .strip()
                .split()
            )
            + ["-O3", "-Wall", "-std=c++17"],
            optional=True,
        )
    ]
)
setup(**setup_args)
