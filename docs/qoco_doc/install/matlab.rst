.. _matlab_installation:

Matlab
==============

Build from Source
-----------------

The Matlab mex wrapper for QOCO can be built from source as follows.

#. Clone the repository
    .. code:: bash

        git clone --recursive https://github.com/qoco-org/qoco-matlab

#. Open up Matlab in the :code:`qoco-matlab` directory and run the following in the Matlab terminal
    .. code:: bash

        make_qoco

#. Test your installation by running the following in the Matlab terminal
    .. code:: bash

        run_qoco_tests

#. Add :code:`qoco-matlab` to your Matlab path by executing the following in the Matlab terminal
    .. code:: bash

        addpath('.')
        savepath
