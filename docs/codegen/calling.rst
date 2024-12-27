.. _calling:

Calling Custom Solver
=====================

Currently the only way to call the custom solver is using C. Remember that the :code:`qoco_custom` can only solve problems within the problem family that was specified when the solver was generated. Here we will assume that the solver was generated using the code that we generated in section :ref:`generate <generate>`. 

Below is sample code to call :code:`qoco_custom` in a file called :code:`run.c`.

.. code:: c 
  
  #include "qoco_custom.h"
  #include <stdio.h>

  int main()
  {
    Workspace work;

    // Set default settings.
    set_default_settings(&work);
    work.settings.verbose = 1;

    // Load the data P, c, A, b, G, h, nsoc, q that was specified when generating code.
    load_data(&work);

    // Solve with custom solver.
    qoco_custom_solve(&work);

    // Print optimal objective.
    printf("\nobj: %f", work.sol.obj);

    // Update A to [2 2 0 0;0 2 2 0].
    work.A[0] = 2;
    work.A[1] = 2;
    work.A[2] = 2;
    work.A[3] = 2;

    // Solve the updated problem.
    qoco_custom_solve(&work);

    // Print optimal objective.
    printf("\nobj: %f", work.sol.obj);
  }

To compile and :code:`run.c`, first follow the instruction in :ref:`building <building>` to build the custom solver, the execute the following in terminal assuming that :code:`run.c` is one directory up from :code:`qoco_custom`

.. code:: bash

  export LD_LIBRARY_PATH=./qoco_custom/build:$LD_LIBRARY_PATH
  gcc run.c -o run -Iqoco_custom -Lqoco_custom/build -lqoco_custom
  ./run
