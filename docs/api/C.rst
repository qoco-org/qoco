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

Helper Functions
----------------
.. doxygenfunction:: qoco_set_csc
.. doxygenfunction:: set_default_settings
.. doxygenfunction:: qoco_update_settings
.. doxygenfunction:: update_vector_data
.. doxygenfunction:: update_matrix_data

Structs
-------
.. doxygenstruct:: QOCOSolver
.. doxygenstruct:: QOCOSettings
.. doxygenstruct:: QOCOWorkspace
.. doxygenstruct:: QOCOSolution
.. doxygenstruct:: QOCOKKT
.. doxygenstruct:: QOCOCscMatrix



