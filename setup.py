from setuptools import find_packages, setup

setup(
    name="Ames-House-Price-Regression",
    version="0.1.0",
    author="Kawsar Ahmmed",
    description="End-to-end machine learning project for Ames house price regression",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)