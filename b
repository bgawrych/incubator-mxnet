cd build
cmake -GNinja -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DUSE_LAPACK=0 -DUSE_OPENCV=1 _GLIBCXX_ASSERTIONS ..
cmake --build .
#make -j48 USE_OPENCV=0 USE_MKLDNN=1 USE_PROFILER=1 USE_LAPACK=0 USE_GPERFTOOLS=0 USE_INTEL_PATH=/opt/intel/
cd ../python
python -m pip install --user -e .
cd ..



#ci/build.py -R --docker-registry mxnetci --platform ubuntu_cpu --docker-build-retries 3 --shm-size 500m /work/runtime_functions.sh sanity_check
#cmake -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DUSE_LAPACK=0 -DUSE_OPENCV=0 _GLIBCXX_ASSERTIONS ..