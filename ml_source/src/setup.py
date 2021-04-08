import os

from setuptools import find_packages, setup


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
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
    name="diabetes",
    version="0.0.1",
    author="",
    author_email="",
    description=(""),
    license="",
    keywords="",
    url="",
    package_dir={"": "ml_source/src"},
    packages=find_packages(where="ml_source/src"),
    classifiers=[],
    install_requires=required_packages,
)
