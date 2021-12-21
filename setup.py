from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name="DSWE",
    version="0.0.1",
    author="Pratyush Kumar, Abhinav Prakash, and Yu Ding",
    author_email="yuding@tamu.edu",
    maintainer_email="pratyush19@tamu.edu",
    description="A python package to supplement the Data Science for Wind Energy (DSWE) book.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TAMU-AML/DSWE-Python",
    project_urls={
        "Bug Tracker": "https://github.com/TAMU-AML/DSWE-Python/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    packages=find_packages(exclude=["doc"]),
    python_requires=">=3.6",
)
