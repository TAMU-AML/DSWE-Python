DNNPowerCurve
=============

Deep Neural Network (DNN) based power curve estimation.

.. code-block:: python

  from dwse import DNNPowerCurve 
  model = DNNPowerCurve()
  model.train(X_train, y_train)
  predictions = model.predict(X_test)
  rmse = model.calculate_rmse(X_test, y_test)

.. automodule:: dswe.dnn
   :members:

.. admonition:: Reference
  
  Karami, Kehtarnavaz, and Rotea, 2021, “Probabilistic 
  neural network to quantify uncertainty of wind power estimation,” 
  arXiv:2106.04656. 