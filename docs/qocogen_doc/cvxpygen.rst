CVXPYgen
========

QOCOGEN can be called from `CVXPYgen <https://github.com/cvxgrp/cvxpygen/>`_ from version 0.5.0 onwards to generate a custom solver and then to call the custom solver from Python.

After :ref:`installing QOCO <python_installation>` and defining your problem in CVXPY, QOCOGEN can be called to generate a custom solver using CVXPYgen as follows

.. code:: python

   from cvxpygen import cpg

   # problem is a CVXPY problem
   cpg.generate_code(problem, code_dir='solver_directory', solver='QOCO')


To call the custom solver through CVXPYgen, run

.. code:: python

   from solver_directory.cpg_solver import cpg_solve
   problem.register_solve('CPG', cpg_solve)
   problem.solve(method='CPG', max_iters=100)

where we set the :code:`max_iters` option to :code:`100`. For a full list of settings that can be changed, refer to :ref:`settings <settings>`.

For an example of setting up and solving a problem using QOCO through CVXPY and QOCOGEN through CVXPYgen, see :ref:`settings <lcvx_example>`.

To see how to call the custom solver directly from C, see :ref:`calling <calling>`.