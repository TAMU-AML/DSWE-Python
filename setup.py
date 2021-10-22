import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

setuptools.setup(
    name="DSWE",
    version="0.0.1",
    author="Pratyush Kumar",
    author_email="pratyush.019@gmail.com",
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
    package_dir={"": "dswe"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)
