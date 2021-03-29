import os

from setuptools import find_packages, setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


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
)
