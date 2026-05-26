.. _c_interface:

.. toctree::
   :maxdepth: 0
   :glob:
   :hidden:

C/C++
=====

.. _C_main_API:

Main solver API
---------------

.. doxygenfunction:: qoco_setup
.. doxygenfunction:: qoco_solve
.. doxygenfunction:: qoco_cleanup

Batch solver API
----------------

The batch API is available for CUDA builds and solves a fixed number of
problems with identical dimensions, cone structure, and CSC sparsity patterns.
Each batch item owns an independent solver instance, so the numeric values of
:code:`c`, :code:`b`, :code:`h`, and the nonzero values of :code:`P`,
:code:`A`, and :code:`G` can differ by item. :code:`qoco_batch_solve()`
attempts every item and stores each optimization status in
:code:`batch->statuses[item]`.

.. doxygenfunction:: qoco_batch_setup
.. doxygenfunction:: qoco_batch_update_vector_data
.. doxygenfunction:: qoco_batch_update_matrix_data
.. doxygenfunction:: qoco_batch_set_x0
.. doxygenfunction:: qoco_batch_solve
.. doxygenfunction:: qoco_batch_get_solution
.. doxygenfunction:: qoco_batch_cleanup

Helper functions
----------------
.. doxygenfunction:: qoco_set_csc
.. doxygenfunction:: set_default_settings
.. doxygenfunction:: qoco_update_settings
.. doxygenfunction:: qoco_update_vector_data
.. doxygenfunction:: qoco_set_x0
.. doxygenfunction:: qoco_update_matrix_data

QOCO data types
---------------
.. doxygenstruct:: QOCOSolver
   :members:
.. doxygenstruct:: QOCOBatchSolver
   :members:
.. doxygenstruct:: QOCOSettings
   :members:
.. doxygenstruct:: QOCOWorkspace
   :members:
.. doxygenstruct:: QOCOCscMatrix
   :members:
.. doxygenstruct:: QOCOSolution
   :members:

