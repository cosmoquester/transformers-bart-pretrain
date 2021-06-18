from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="tf2-keras-template",
    version="0.0.1",
    description="This is template repository for tensorflow keras model development.",
    python_requires=">=3.6",
    install_requires=["tensorflow>=2"],
    url="https://github.com/cosmoquester/tf2-keras-template.git",
    author="Park Sangjun",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    packages=find_packages(exclude=["tests"]),
)
