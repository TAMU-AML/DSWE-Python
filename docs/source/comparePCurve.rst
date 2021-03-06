ComparePCurve
==============

Power curve comparison and uncertainity quantification.

.. code-block:: python

  from dwse import ComparePCurve
  model = ComparePCurve(Xlist, ylist, testcol)  
  diff = model.compute_weighted_difference(weights) 

.. automodule:: dswe.comparePCurve
   :members:

.. admonition:: Reference

  Ding, Kumar, Prakash, Kio, Liu, Liu, and Li, 2021, 
  “A case study of space-time performance comparison of 
  wind turbines on a wind farm,” Renewable Energy, Vol. 171, pp. 735-746.