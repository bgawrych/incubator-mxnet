#!/usr/bin/env python
 
# This script is to benchmark some mxnet operators on cpu
# Run it as follows (change core numbers accordingly):
# KMP_AFFINITY=granularity=fine,noduplicates,compact,1,0 OMP_NUM_THREADS=12 numactl --physcpubind=0-11 --membind=0 python mxnet_perf.py
# It will run with oneDNN if built with oneDNN
# To run by native MXNet implementation:
# export MXNET_MKLDNN_ENABLED=0

import mxnet as mx
 
import time
from mxnet import profiler
import sys

profiler.set_config(profile_all=True,
                    aggregate_stats=True,
                    continuous_dump=True,
                    filename='profile_output.json')
                    
def perf_upsampling_nearest():
    shapes = [(3, 3, 128, 64), (3, 3, 128, 128), (5, 5, 128, 128), (5, 5, 256, 256), (32, 3, 256, 256), (1,256, 136, 100)]
    scales = [2, 3, 4]
    warmup = 20
    runs = 1000
    results = []
    profile = len(sys.argv) > 1
    for shape in shapes:
        for scale in scales:
            a = mx.random.uniform(shape=shape)
            tic = 0
            for i in range(runs + warmup):
                if i == warmup:
                    if profile:
                        profiler.set_state(state='run')
                    tic = time.time()
                y = mx.nd.UpSampling(a, scale=scale, sample_type='nearest')
                mx.nd.waitall()
 
            toc = time.time()
            if(profile):
                profiler.set_state(state='stop')
                print('upsampling benchmark, shape={}, scale={} time={} ms/iter'.format(shape, scale, (toc-tic)*1000/(runs)))
                print(profiler.dumps(reset=True))
            results.append((shape, scale, (toc-tic)*1000/(runs)))

    # if MKLDNN_VERBOSE is enabled printing now will avoid flooding result with logs
    for shape, scale, duration in results:
        print('upsampling benchmark, shape={}, scale={} time={} ms/iter'.format(shape, scale, duration))


if __name__ == '__main__':
    perf_upsampling_nearest()