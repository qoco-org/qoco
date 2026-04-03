.. _c_installation:

C/C++
==============

Build binaries
-----------------
Building binaries from source requires installation of `cmake <https://cmake.org/download/>`_ and a C compiler such as `clang <https://clang.llvm.org/get_started.html>`_ or `gcc <https://gcc.gnu.org/install/>`_.

Run the following to get the :code:`libqoco.so` shared object file.

#. Clone the repository and change directory.
    .. code:: bash

        git clone https://github.com/qoco-org/qoco
        cd qoco

#. Make build directory and change directory.
    .. code:: bash

        mkdir build
        cd build

#. Compile sources.
    .. code:: bash

        cmake .. && make

    You should now see two binaries: :code:`libqoco.so` which is the qoco library, and :code:`qoco_demo` which solves a sample SOCP. To build unit tests, add :code:`-DENABLE_TESTING:BOOL=True` to the :code:`cmake` call. Note that when running unit tests, it is required to have :code:`cvxpy` installed.

CUDA Backend
~~~~~~~~~~~~

To build with GPU acceleration, pass :code:`-DQOCO_ALGEBRA_BACKEND=cuda` to CMake. This requires the `CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_ to be installed and :code:`nvcc` to be on your :code:`PATH`.

.. code:: bash

    cmake .. -DQOCO_ALGEBRA_BACKEND=cuda && make

**Runtime requirements**: The CUDA backend loads `cuDSS <https://developer.nvidia.com/cudss>`_ dynamically at runtime via :code:`dlopen`. cuDSS does not need to be present at build time, but it must be installed and findable on your system's library path (:code:`LD_LIBRARY_PATH` on Linux, :code:`DYLD_LIBRARY_PATH` on macOS) before running any program linked against :code:`libqoco.so`.

Use with CMake
-----------------
To use in a CMake project add the following to your :code:`CMakeLists.txt` file

.. code:: cmake

    add_subdirectory(QOCO_DIRECTORY)
    target_link_libraries(yourExecutable qoco)

Where :code:`QOCO_DIRECTORY` is the location where :code:`qoco` is cloned. To use the CUDA backend in your project, pass :code:`-DQOCO_ALGEBRA_BACKEND=cuda` when invoking CMake.
