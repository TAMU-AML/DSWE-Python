DNNPowerCurve
=============

Power predictions using Deep Neural Network (DNN)

.. code-block:: python

  from dwse import DNNPowerCurve 
  model = DNNPowerCurve()
  model.train(X_train, y_train)
  predictions = model.predict(X_test)
  rmse = model.calculate_rmse(X_test, y_test)

.. automodule:: dswe.dnn
   :members: