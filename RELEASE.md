# Release
Here are the steps to make a new qoco release

1. First ensure unit tests are passing in `CI` and run unit tests on GPU and ensure they are passing (except `cone_test`, `linalg_test`, `lcvx_bad_scaling_test`)
2. Bump version in `CMakeLists.txt`
3. If documentation changes are made, run `doc_deploy` workflow in `Actions` tab
4. Make new tag of qoco
