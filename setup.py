from setuptools import find_packages, setup

setup(
    name="analyzer",
    version="0.1",
    packages=find_packages(),
    install_requires=["emd-signal", "pandas", "matplotlib", "openpyxl"],
    extras_require={
        "conda": ["pypdf2"],
        "pip": ["neuronol"],
    },  # this does not install, only documents
)

# Install using: `pip install -e .`
