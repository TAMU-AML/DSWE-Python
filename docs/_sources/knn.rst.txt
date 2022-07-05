KNNPowerCurve
==============

K-nearest neighbors (KNN) based power curve estimation.

.. code-block:: python

  from dwse import KNNPowerCurve
  model = KNNPowerCurve()
  model.fit(X_train, y_train)
  prediction = model.predict(X_test)
  model.update(X_update, y_update)
  prediction = model.predict(X_test_new)

.. automodule:: dswe.knn
   :members: