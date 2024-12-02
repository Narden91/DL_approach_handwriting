from setuptools import setup, find_packages

setup(
    name="handwriting_analysis",
    version="0.1",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "pytorch-lightning",
        "torch",
        "hydra-core",
        "torchmetrics"
    ]
)
