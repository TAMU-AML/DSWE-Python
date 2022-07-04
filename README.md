# DSWE-Python
**DSWE** is a python package to supplement the Data Science for Wind Energy (DSWE) book and other state-of-the-art methods used in wind energy power modelling.

Website: https://tamu-aml.github.io/DSWE-Python/

Installation
------------

### Dependencies

DSWE requires:

- Python (>=3.8)
- NumPy (>=1.21.2)
- Pandas (>=1.3.3)
- Scikit-learn (>=1.0)
- Scipy (>=1.7.0)
- Statsmodels (>=0.13.0)
- PyTorch (>=1.0.0)
- Matplotlib (>=3.4.3)

--------------------------------------------------------------------------------

```console
pip install dswe
```

To get the latest code changes as they are merged, you can clone this repo and build from source manually.
```console
git clone -b dev https://github.com/TAMU-AML/DSWE-Python/
pip install DSWE-Python/
```

### Notes

- **AMK**: The optimal bandwidth selection algorithm i.e., the direct plug-in (DPI) approach, is not implemented yet. You need to pass bandwidth corresponding to each column.
- **BayesTreePowerCurve**: This module is built on top BartPy which is a python implementation of the Bayesian additive regressions trees (BART). The BartPy package has not been updated for a long time and simple ```pip install bartypy``` sometimes does not work. You have to explicitly clone the repo and build from source manually. You can follow the following steps to install BartPy package.

```console
git clone https://github.com/JakeColtman/bartpy
pip install bartpy/
```

--------------------------------------------------------------------------------

References
----------
- [Data science for Wind Energy book](https://aml.engr.tamu.edu/book-dswe/)
- **AMK**: [Power Curve Estimation With Multivariate Environmental Factors for Inland and Offshore Wind Farms](https://aml.engr.tamu.edu/wp-content/uploads/sites/164/2017/11/J53.pdf)
- **TempGP**: [The temporal overfitting problem with applications in wind power curve modeling](https://arxiv.org/abs/2012.01349)
- **CovMatch, FunGP**: [A Case Study of Space-time Performance Comparison of Wind Turbines on a Wind Farm](https://arxiv.org/pdf/2005.08652.pdf)
