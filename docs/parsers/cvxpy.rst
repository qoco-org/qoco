CVXPY
=====

QOCO can be called from `CVXPY <http://www.cvxpy.org/>`_ starting in version 1.7.0. To use QOCO with CVXPY before version 1.7.0 of CVXPY is officially released, you must `build CVXPY from source <https://www.cvxpy.org/install/index.html>`_.

After :ref:`installing QOCO <python_installation>` and defining your problem problem in CVXPY, QOCO can be called as follows

.. code:: python

   problem.solve(solver='QOCO', max_iters=100)


where we set the :code:`max_iters` option to :code:`100`. For a full list of settings that can be changed, refer to :ref:`settings <settings>`.

For some example problems that can be solved with CVXPY refer to `examples <https://www.cvxpy.org/examples/index.html>`_. To solve these problems with QOCO, add the keyword argument :code:`solver='QOCO'` to the :code:`problem.solve()` function.

