import os

from setuptools import find_packages, setup


def read_requirements():
    """Parse requirements from requirements.txt."""
    reqs_path = os.path.join(".", "requirements.txt")
    with open(reqs_path, "r") as f:
        requirements = [line.rstrip() for line in f]
    return requirements


console_scripts = ["zatsuon=zatsuon.app:main"]

setup(
    name="zatsuon",
    version="0.0.0",
    description="remove noise of audio files",
    author="Hideaki Takahashi",
    author_email="koukyosyumei@hotmail.com",
    license="MIT",
    install_requires=read_requirements(),
    url="https://github.com/Koukyosyumei/Zatsuon",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    entry_points={"console_scripts": console_scripts},
)
