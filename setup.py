from pathlib import Path
from setuptools import find_packages, setup


core_requirements = []

setup(
    name="demodiffusion",
    version="0.1",
    author="Sungjae Park",
    author_email="sungjae2@andrew.cmu.edu",
    url="https://demodiffusion.github.io/",
    description="Author's implementation of Demodiffusion",
    long_description="",
    long_description_content_type="text/markdown",
    packages=find_packages(include=["demodiffusion.*"]),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=core_requirements,
)