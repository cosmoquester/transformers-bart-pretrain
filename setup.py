from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="transformers-bart-pretrain",
    version="0.0.1",
    description="Script to train hugginface transformers BART",
    python_requires=">=3.6",
    install_requires=["tensorflow>=2", "tensorflow-text", "transformers"],
    url="https://github.com/cosmoquester/transformers-bart-pretrain.git",
    author="Park Sangjun",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=["tests"]),
)
