from setuptools import find_packages, setup

setup(
    name="Iris classification thingy",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "joblib==1.1.0",
        "numpy==1.22.3",
        "scikit-learn==1.0.2",
        "scipy==1.8.0",
    ],
)
