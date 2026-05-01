from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="genoarmory",
    version="0.1.0",
    description="A DNA sequence Adversial attack and defense benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.0.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
    ],
    entry_points={
        "console_scripts": [
            "genoarmory=genoarmory.cli:main",
        ],
    },
)
