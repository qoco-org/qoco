#!/bin/bash
# Note: cuDSS was compiled against cuBLAS 12, but system has cuBLAS 13
# The build will succeed with --allow-shlib-undefined, but runtime may fail
# To fix: either install cuBLAS 12 or get cuDSS compiled against cuBLAS 13

export CXX=/usr/bin/clang++ && export CC=/usr/bin/clang && cd build && cmake -DQOCO_BUILD_TYPE:STR=Release -DENABLE_TESTING:BOOL=True -DBUILD_QOCO_BENCHMARK_RUNNER:BOOL=False -DQOCO_ALGEBRA_BACKEND:STR=builtin .. && make -j$(nproc) && cd tests && ./markowitz_test