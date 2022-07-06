from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setup(
    name="dswe",
    version="0.0.2",
    author="Pratyush Kumar, Abhinav Prakash, and Yu Ding",
    author_email="yuding@tamu.edu",
    maintainer_email="pratyush.019@gmail.com",
    description="A python package to supplement the Data Science for Wind Energy (DSWE) book.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://tamu-aml.github.io/DSWE-Python/",
    project_urls={
        "Source": "https://github.com/TAMU-AML/DSWE-Python",
        "Bug Tracker": "https://github.com/TAMU-AML/DSWE-Python/issues",
    },
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
)
