"""Setup configuration for the COVID-19 chest X-ray detection package."""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README = (ROOT / "README.md").read_text(encoding="utf-8")

setup(
    name="covid-xray-detection",
    version="0.1.0",
    description="COVID-19 chest X-ray classification using a convolutional neural network.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Baldari.dev",
    license="MIT",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "streamlit",
    ],
    extras_require={
        "dev": ["pytest", "flake8"],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
