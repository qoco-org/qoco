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

Helper functions
----------------
.. doxygenfunction:: qoco_set_csc
.. doxygenfunction:: set_default_settings
.. doxygenfunction:: qoco_update_settings
.. doxygenfunction:: update_vector_data
.. doxygenfunction:: update_matrix_data

QOCO data types
---------------
.. doxygenstruct:: QOCOSolver
   :members:
.. doxygenstruct:: QOCOSettings
   :members:
.. doxygenstruct:: QOCOWorkspace
   :members:
.. doxygenstruct:: QOCOKKT
   :members:
.. doxygenstruct:: QOCOCscMatrix
   :members:
.. doxygenstruct:: QOCOSolution
   :members:


