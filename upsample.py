#!/usr/bin/env python
 
# This script is to benchmark some mxnet operators on cpu
# Run it as follows (change core numbers accordingly):
# export KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0
# export OMP_NUM_THREADS=28
# numactl --physcpubind=0-27 --membind=0 python mx_op_benchmark.py
# KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 OMP_NUM_THREADS=12 numactl --physcpubind=0-11 --membind=0 python upsample.py
import mxnet as mx
 
import time
from mxnet import profiler

profiler.set_config(profile_all=True,
                    aggregate_stats=True,
                    continuous_dump=True,
                    filename='profile_output.json')
                    
def perf_upsampling():
    shapes = [(32, 3, 256, 256)]
    scales = [2, 3, 4]
 
    warmup = 20
    runs = 1000
 
    for shape in shapes:
        for scale in scales:
            a = mx.random.uniform(shape=shape)
 
            tic = 0
            for i in range(runs + warmup):
                if i == warmup:
                    tic = time.time()
                _ = mx.nd.UpSampling(a, scale=scale, sample_type='nearest')
                mx.nd.waitall()
 
            toc = time.time()
            print('upsampling benchmark, shape={}, scale={} time={} ms/iter'.format(shape, scale, (toc-tic)*1000/(runs)))

if __name__ == '__main__':
    perf_upsampling()



# [11:24:23] ../src/storage/storage.cc:198: Using Pooled (Naive) StorageManager for CPU
# upsampling benchmark, shape=(3, 3, 128, 64), scale=2 time=0.4628729820251465 ms/iter
# upsampling benchmark, shape=(3, 3, 128, 64), scale=3 time=0.9577136039733887 ms/iter
# upsampling benchmark, shape=(3, 3, 128, 64), scale=4 time=1.502694845199585 ms/iter
# upsampling benchmark, shape=(3, 3, 128, 128), scale=2 time=0.7934966087341309 ms/iter
# upsampling benchmark, shape=(3, 3, 128, 128), scale=3 time=1.6678402423858643 ms/iter
# upsampling benchmark, shape=(3, 3, 128, 128), scale=4 time=2.7772817611694336 ms/iter
# upsampling benchmark, shape=(5, 5, 128, 128), scale=2 time=1.9956700801849365 ms/iter
# upsampling benchmark, shape=(5, 5, 128, 128), scale=3 time=4.371188163757324 ms/iter
# upsampling benchmark, shape=(5, 5, 128, 128), scale=4 time=7.666296482086182 ms/iter
# upsampling benchmark, shape=(5, 5, 128, 128), scale=2 time=2.0013015270233154 ms/iter
# upsampling benchmark, shape=(5, 5, 128, 128), scale=3 time=4.36636209487915 ms/iter
# upsampling benchmark, shape=(5, 5, 128, 128), scale=4 time=7.66067910194397 ms/iter