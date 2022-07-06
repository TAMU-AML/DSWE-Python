FunGP
======

Function comparison using Gaussian Process and Hypothesis testing.

.. code-block:: python

  from dwse import FunGP
  model = FunGP(Xlist, ylist, testset)
  mu1, mu2 = model.mu1, model.mu2


.. automodule:: dswe.funGP
   :members:

.. admonition:: Reference
  
  Prakash, Tuo, and Ding, 2022, “Gaussian process 
  aided function comparison using noisy scattered data,” 
  Technometrics, Vol. 64, pp. 92-102.