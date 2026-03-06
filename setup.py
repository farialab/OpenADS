from setuptools import setup, find_packages

setup(
    name="ads",
    version="1.0.0",
    description="Acute Stroke Detection (ADS) - A tool for automated brain lesion detection",
    author="Authors: Joshua Shun Liu, Chin-Fu Liu, Andreia V. Faria",
    author_email="andreia.faria@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "ads=ads.cli.main:main",
        ],
    },
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.7.0",
        "numpy>=1.19.0",
        "antspyx>=0.2.7",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "scikit-image>=0.17.0",
        "scipy>=1.5.0",
        "pyyaml>=5.3.0",
        "nibabel",
        "tqdm",
        "scikit-learn",
        "surfa"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
