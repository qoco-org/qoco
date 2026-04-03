.. _api:

API
==========

.. toctree::
   :maxdepth: 1
   :hidden:

   C.rst
   python.rst
   matlab.rst
   ../cvxpy.rst
   settings.rst
   exit.rst

Direct Interfaces
-----------------

In the direct interfaces, the user constructs the problem data in standard form and passes it directly to QOCO. The links below describe how to set up and solve problems and change solver settings within each language.

+------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+
| Language                           | Maintainer                                               | Repository                                                                               |
+====================================+==========================================================+==========================================================================================+
| :ref:`C/C++ <c_interface>`         |   `Govind Chari <https://govindchari.com/>`_             | `qoco <https://github.com/qoco-org/qoco>`_                                               |
+------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+
| :ref:`Python <python_interface>`   |   `Govind Chari <https://govindchari.com/>`_             | `qoco-python <https://github.com/qoco-org/qoco-python>`_                                 |
+------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+
| :ref:`Matlab <matlab_interface>`   |   `Govind Chari <https://govindchari.com/>`_             | `qoco-matlab <https://github.com/qoco-org/qoco-matlab>`_                                 |
+------------------------------------+----------------------------------------------------------+------------------------------------------------------------------------------------------+

Modelling Interface
-------------------

:ref:`CVXPY <cvxpy_interface>` is a modelling interface, not a direct interface. The user defines a problem in CVXPY's high-level syntax, which is then parsed and reduced to QOCO's standard form automatically. CVXPY then calls the :ref:`Python interface <python_interface>` to pass the problem data to QOCO.

This section also describes the solver :ref:`settings <settings>` which can be changed, and the possible :ref:`exit flags <exit_flags>` the solver returns.
