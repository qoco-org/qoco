.. _python_installation:

Python
==============

Pip
----
QOCO is on PyPI and can be installed as follows. If on Windows, MSVC must be installed on your system.

.. code:: bash

   pip install qoco

CUDA Backend
~~~~~~~~~~~~

To install QOCO with GPU acceleration via the CUDA backend, install the separate ``qoco-cuda`` package using the extras specifier:

.. code:: bash

   pip install qoco[cuda]

This installs ``qoco-cuda`` alongside ``qoco`` and enables the ``algebra="cuda"`` option when constructing a solver.

**Runtime requirements**: The CUDA backend loads its GPU libraries dynamically at runtime. Before using it, ensure the following are installed on your system:

- `CUDA Toolkit 13 <https://developer.nvidia.com/cuda-downloads>`_ — provides the GPU runtime and compiler.
- `cuDSS <https://developer.nvidia.com/cudss>`_ — NVIDIA's GPU-accelerated sparse direct solver library, used internally by QOCO for KKT factorization. Download it from the NVIDIA developer website and follow the installation instructions provided there.

After installing, verify the backend is available:

.. code:: python

   import qoco
   qoco.algebra_available("cuda")  # should return True


Build from Source
-----------------

The Python wrapper for QOCO can also be built from source as follows.

.. code:: bash

   git clone https://github.com/qoco-org/qoco-python
   cd qoco-python
   pip install .

CUDA Backend from Source
~~~~~~~~~~~~~~~~~~~~~~~~

To build the CUDA backend from source, build the ``qoco-cuda`` sub-package instead:

.. code:: bash

   git clone https://github.com/qoco-org/qoco-python
   cd qoco-python/backend/cuda
   pip install .

This requires the CUDA Toolkit to be installed and the ``nvcc`` compiler to be on your ``PATH``.
