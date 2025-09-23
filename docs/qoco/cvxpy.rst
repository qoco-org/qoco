CVXPY
=====

QOCO can be called from `CVXPY <http://www.cvxpy.org/>`_ from version 1.6.4 onwards.

After :ref:`installing QOCO <python_installation>` and defining your problem in CVXPY, QOCO can be called as follows

.. code:: python

   problem.solve(solver='QOCO', max_iters=100)


where we set the :code:`max_iters` option to :code:`100`. For a full list of settings that can be changed, refer to :ref:`settings <settings>`.

For some example problems that can be solved with CVXPY refer to `examples <https://www.cvxpy.org/examples/index.html>`_. To solve these problems with QOCO, add the keyword argument :code:`solver='QOCO'` to the :code:`problem.solve()` function.

