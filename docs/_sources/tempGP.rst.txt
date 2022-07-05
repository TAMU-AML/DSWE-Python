TempGP
=======

The Gaussian process-based power curve that avoids temporal overfitting.

.. code-block:: python

  from dwse import TempGP
  model = TempGP()
  model.fit(X_train, y_train)
  prediction = model.predict(X_test)
  model.update(X_update, y_update)
  prediction = model.predict(X_test_new)

.. automodule:: dswe.tempGP
   :members: