CovMatch
=========

Covariate Matching.
The model aims to take list of two data sets and returns the after matched data sets using user specified covariates and threshold.

.. code-block:: python

  from dwse import CovMatch
  model = CovMatch(Xlist, ylist)
  matched_X = model.matched_data_X
  matched_y = model.matched_data_y

.. automodule:: dswe.covmatch
   :members: