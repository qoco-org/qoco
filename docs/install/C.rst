.. _c_installation:

C/C++
==============

Build binaries
-----------------

Run the following to get the :code:`libqoco.so` shared object file.

#. Clone the repository and change directory.
    .. code:: bash

        git clone https://github.com/govindchari/qoco
        cd qoco

#. Make build directory and change directory.
    .. code:: bash

        mkdir build
        cd build

#. Compile sources.
    .. code:: bash

        cmake .. && make

    You should now see two binaries: :code:`libqoco.so` which is the qoco library, and :code:`qoco_demo` which solves a sample SOCP.

Use with CMake
-----------------
To use in a CMake project add the following to your :code:`CMakeLists.txt` file

.. code:: bash

    add_subdirectory(QOCO_DIRECTORY)
    target_link_libraries(yourExecutable qoco)

Where :code:`QOCO_DIRECTORY` is the location where  :code:`qoco` is cloned.