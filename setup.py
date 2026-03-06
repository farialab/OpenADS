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
    python_requires=">=3.10,<3.13",
    install_requires=[
        "torch>=2.5,<2.6",
        "numpy>=2.0,<2.1",
        "antspyx>=0.5,<0.6",
        "pandas>=2.2,<2.3",
        "matplotlib>=3.9,<3.10",
        "scikit-image>=0.24,<0.25",
        "scipy>=1.13,<1.14",
        "pyyaml>=6.0",
        "nibabel>=5.3,<5.4",
        "tqdm>=4.67,<4.68",
        "scikit-learn>=1.7,<1.8",
        "surfa>=0.6.3,<0.7",
        "shap == 0.50.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
