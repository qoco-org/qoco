.. _python_installation:

Python
==============

Pip
----
QOCO is on PyPI and can be installed as follows

.. code:: bash

   pip install qoco

CUDA Backend
~~~~~~~~~~~~

To install QOCO with CUDA backend support for GPU acceleration, use:

.. code:: bash

   pip install qoco[cuda]

**Requirements**: The CUDA backend requires CUDA 13 and cudss to be installed on your system. Make sure you have these dependencies before installing the CUDA backend.


Build from Source
-----------------

The Python wrapper for QOCO can also be built from source as follows 

.. code:: bash

   git clone https://github.com/qoco-org/qoco-python
   cd qoco-python
   pip install .