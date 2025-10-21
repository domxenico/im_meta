from setuptools import setup, find_packages

setup(
    name="immeta",
    description="IM-META: Influence Maximization with Node Metadata",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "networkx>=3.0",
        "numpy>=1.24.0",
    ],
)