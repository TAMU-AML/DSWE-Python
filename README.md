# DSWE-Python
**DSWE** is a python package, written by [Professor Yu Ding's](https://aml.engr.tamu.edu/) research team, to supplement the Data Science for Wind Energy ([DSWE](https://aml.engr.tamu.edu/book-dswe/)) book and other state-of-the-art methods used in wind energy applications.

Website: https://tamu-aml.github.io/DSWE-Python/

Installation
------------

### Dependencies

DSWE requires:

- Python (>=3.6)
- NumPy (>=1.21.2)
- Pandas (>=1.3.3)
- Scikit-learn (>=1.0)
- Scipy (>=1.7.0)
- Statsmodels (>=0.13.0)
- PyTorch (>=1.0.0)
- Matplotlib (>=3.4.3)

All the required packages don't need to be pre-installed. Installing DSWE would automatically install everything.

--------------------------------------------------------------------------------

```console
pip install dswe
```

To get the latest code changes as they are merged, you can clone this repo and build from source manually.

```console
git clone https://github.com/TAMU-AML/DSWE-Python/
pip install DSWE-Python/
```

### Notes

- **AMK**: The optimal bandwidth selection algorithm i.e., the direct plug-in (DPI) approach, is not implemented yet. You need to provide bandwidth corresponding to each column.
- **BayesTreePowerCurve**: This module is built on top BartPy which is a python implementation of the Bayesian additive regressions trees (BART). To use the BayesTreePowerCurve model, you need to install the BartPy manually. The BartPy package has not been updated for a long time and simple ```pip install bartypy``` sometimes does not work. 

You have to explicitly clone the repo and build from source manually. You can follow the following steps to install BartPy package.

```console
git clone https://github.com/JakeColtman/bartpy
pip install bartpy/
```

--------------------------------------------------------------------------------

References
----------
- [Data Science for Wind Energy book.](https://aml.engr.tamu.edu/book-dswe/)
- **AMK**: Lee, Ding, Genton, and Xie, 2015, “Power curve estimation with multivariate environmental factors for inland and offshore wind farms,” Journal of the American Statistical Association, Vol. 110, pp. 56-67. [[pdf]](https://aml.engr.tamu.edu/wp-content/uploads/sites/164/2017/11/J53.pdf)
- **TempGP**: Prakash, Tuo, and Ding, 2022, “The temporal overfitting problem with applications in wind power curve modeling,” Technometrics, accepted. [[preprint]](https://arxiv.org/abs/2012.01349)
- **CovMatch**: Shin, Ding, and Huang, 2018, “Covariate matching methods for testing and quantifying wind turbine upgrades,” Annals of Applied Statistics, Vol. 12(2), pp. 1271-1292. [[accepted version]](http://aml.engr.tamu.edu/wp-content/uploads/sites/164/2017/11/J64_accepted.pdf)
- **FunGP**: Prakash, Tuo, and Ding, 2022, “Gaussian process aided function comparison using noisy scattered data,” Technometrics, Vol. 64, pp. 92-102. [[preprint]](http://aml.engr.tamu.edu/wp-content/uploads/sites/164/2001/09/J78_Main.pdf)
- **ComparePCurve**: Ding, Kumar, Prakash, Kio, Liu, Liu, and Li, 2021, “A case study of space-time performance comparison of wind turbines on a wind farm,” Renewable Energy, Vol. 171, pp. 735-746. [[preprint]](https://arxiv.org/abs/2005.08652)
- **DNNPowerCurve**: The DNNPowerCurve function is partially based on Karami, Kehtarnavaz, and Rotea, 2021, "Probabilistic neural network to quantify uncertainty of wind power estimation," arXiv:2106.04656. [[preprint]](https://arxiv.org/abs/2106.04656).  Our team refined the network architecture and tuned the training parameters.
