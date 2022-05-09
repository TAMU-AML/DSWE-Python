ComparePCurve
==============

Power curve comparison (ComparePCurve)

.. code-block:: python

  from dwse import ComparePCurve
  model = ComparePCurve(Xlist, ylist, testcol)  # contains all the trained parameters
  diff = model.compute_weighted_difference(weights)  # weighted difference based on the provided weights

.. automodule:: dswe.comparePCurve
   :members: