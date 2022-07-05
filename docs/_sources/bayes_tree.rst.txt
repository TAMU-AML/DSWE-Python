BayesTreePowerCurve
====================

Bayesian Additive Regressions Trees (BART) based power curve estimatation.

.. code-block:: python

  from dwse import BayesTreePowerCurve
  model = BayesTreePowerCurve()
  model.fit(X_train, y_train)
  prediction = model.fit(X_test)

.. automodule:: dswe.bayes_tree
   :members: