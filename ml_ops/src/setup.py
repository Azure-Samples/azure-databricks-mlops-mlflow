import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


requirements_file_name = "requirements.txt"
with open(requirements_file_name) as f:
    required_packages = f.read().splitlines()
required_packages = [
    package.strip(" ")
    for package in required_packages
    if package.strip(" ") and "#" not in package
]

setup(
    name="diabetes_mlops",
    version="0.0.1",
    author="",
    author_email="",
    description=(""),
    license="",
    keywords="",
    url="",
    package_dir={"": "ml_ops/src"},
    packages=find_packages(where="ml_ops/src"),
    classifiers=[],
    install_requires=required_packages,
)
