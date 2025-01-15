.. _building:

Building Custom Solver
----------------------
#. Change directory into custom solver.
    .. code:: bash

        cd qoco_custom

#. Make build directory and change directory.
    .. code:: bash

        mkdir build
        cd build

#. Compile sources.
    .. code:: bash

        cmake -DENABLE_PRINTING:BOOL=TRUE .. && make
    
    You should now see two binaries: :code:`libqoco_custom.so` which is the custom solver library, and :code:`runtest` which solves a sample SOCP with the custom solver.
