# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import sys
import mxnet as mx
from mxnet import nd, gluon
import numpy as np
from random import randint
from mxnet.test_utils import almost_equal, assert_almost_equal
from numpy.testing import assert_allclose
import warnings
import collections
import ctypes
import itertools
from itertools import product
from functools import partial
import mxnet.contrib.amp as amp
from mxnet.test_utils import set_default_context, download_model, same_symbol_structure, assert_almost_equal_with_err, rand_shape_nd
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon import SymbolBlock, nn, rnn
from mxnet.contrib.amp import amp
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.insert(0, os.path.join(curr_path, '../unittest'))
import pytest
from common import assert_raises_cudnn_not_satisfied, with_seed, retry


bfloat16 = np.dtype([('bfloat16', np.uint16)])



def check_rnn_states(fused_states, stack_states, num_layers, bidirectional=False, is_lstm=True):
    directions = 2 if bidirectional else 1
    assert len(stack_states) / len(fused_states) == num_layers * directions

    fused_states = [state.asnumpy() for state in fused_states]
    stack_states = [np.expand_dims(state.asnumpy(), axis=0) for state in stack_states]
    if is_lstm:
        stack_states_h = stack_states[0::2]
        stack_states_c = stack_states[1::2]
        stack_states = [np.concatenate(stack_states_h, axis=0), np.concatenate(stack_states_c, axis=0)]
    else:
        stack_states = [np.concatenate(stack_states, axis=0)]

    for f, s in zip(fused_states, stack_states):
        assert f.shape == s.shape
        assert_almost_equal(f, s, atol=1e-4, rtol=1e-4)


def create_op_by_mode(mode):
    if mode == 'lstm':
        fused_op = gluon.rnn.LSTM
        stack_op = gluon.rnn.LSTMCell
        recurrent_block_prefix = 'lstm0_'
    elif mode == 'gru':
        fused_op = gluon.rnn.GRU
        stack_op = gluon.rnn.GRUCell
        recurrent_block_prefix = 'gru0_'
    elif mode == 'rnn_relu':
        fused_op = partial(gluon.rnn.RNN, activation='relu')
        stack_op = partial(gluon.rnn.RNNCell, activation='relu')
        recurrent_block_prefix = 'rnn0_'
    elif mode == 'rnn_tanh':
        fused_op = partial(gluon.rnn.RNN, activation='tanh')
        stack_op = partial(gluon.rnn.RNNCell, activation='tanh')
        recurrent_block_prefix = 'rnn0_'

    return fused_op, stack_op, recurrent_block_prefix
    
def check_rnn_consistency(fused_layer, stack_layer, loss, input_size, hidden_size, bidirectional=False, rtol=1e-2, atol=1e-4):
    data_shape = (1, 5, input_size)
    data_range = (0.0, 10.0)
    rand_input_fp32 = mx.nd.random.uniform(low=data_range[0], high=data_range[1], shape=data_shape)
    rand_input_bf16 = mx.nd.amp_cast(rand_input_fp32, dtype=bfloat16)

    fused_begin_state = fused_layer.begin_state(1)
    stack_states = stack_layer.begin_state(1)
    fused_layer.infer_shape(rand_input_fp32, fused_begin_state)
    fused_layer_params = fused_layer.collect_params()
    stack_layer_params = stack_layer.collect_params()
    print("\n")
    for name, value in fused_layer_params.items():
        if 'weight' in name:
            w = mx.nd.zeros(shape=value.shape)
            value.set_data(w.copy())
            stack_layer_params[name].set_data((mx.nd.amp_cast(w.copy(), dtype=bfloat16)))
        else:
            w = mx.nd.random.normal(shape=value.shape)
            value.set_data(w.copy())
            stack_layer_params[name].set_data(w.copy())
        print(name, value)
        print(stack_layer_params[name])

    #return
    fx = rand_input_fp32.copy()
    sx = rand_input_bf16.copy()
    y = nd.random.uniform(shape=(1, 5, hidden_size * 2 if bidirectional else hidden_size))

    fx.attach_grad()
    with mx.autograd.record():
        fused_out, fused_states = fused_layer(fx, fused_begin_state)
        l = loss(fused_out, y).mean()
    #l.backward()
    #fused_grads = dict([(name, p.grad()) for name, p in fused_layer.collect_params().items()])
    #fused_input_grad = fx.grad.asnumpy()

    sx.attach_grad()
    with mx.autograd.record():
        stack_out, stack_states = stack_layer(sx, stack_states) #stack_layer.unroll(5, sx, begin_state=stack_states, merge_outputs=True)
        #stack_out.wait_to_read()
        output_bf16_2_fp32 = mx.nd.amp_cast(stack_out, dtype="float32")
        #l = loss(output_bf16_2_fp32, y).mean()
    #l.backward()
    #stack_grads = dict([(name, p.grad()) for name, p in stack_layer.collect_params().items()])
    #stack_input_grad = sx.grad.asnumpy()
    #return
    assert_allclose(fused_out.asnumpy(), output_bf16_2_fp32.asnumpy(), rtol=rtol, atol=atol)
    return
    assert_allclose(fused_input_grad, stack_input_grad, rtol=rtol, atol=atol)
    for key, value in fused_grads.items():
        assert_allclose(value.asnumpy(), stack_grads[key].asnumpy(), rtol=rtol, atol=atol)

    num_layers = fused_begin_state[0].shape[0] // (2 if bidirectional else 1)
    check_rnn_states(fused_states, stack_states, num_layers, bidirectional, len(fused_begin_state) == 2)




def check_rnn_unidir_layer_gradients(mode, input_size, hidden_size, num_layers, loss):
    fused_op, stack_op, recurrent_block_prefix = create_op_by_mode(mode)

    fp32_rnn = fused_op(hidden_size, num_layers=num_layers, layout='NTC', bidirectional=False, prefix=recurrent_block_prefix)
    fp32_rnn.initialize()

    bf16_rnn = fused_op(hidden_size, num_layers=num_layers, layout='NTC', bidirectional=False, prefix=recurrent_block_prefix, dtype=bfloat16)
    bf16_rnn.initialize()
    check_rnn_consistency(fp32_rnn, bf16_rnn, loss, input_size, hidden_size)


# def check_rnn_bidir_layer_gradients(mode, input_size, hidden_size, num_layers, loss):
#     fused_op, stack_op, recurrent_block_prefix = create_op_by_mode(mode)

#     fused_layer = fused_op(hidden_size, num_layers=num_layers, layout='NTC', bidirectional=True, prefix=recurrent_block_prefix, dtype='bfloat16')
#     fused_layer.initialize()

#     stack_layer = mx.gluon.rnn.HybridSequentialRNNCell(prefix=recurrent_block_prefix)
#     with stack_layer.name_scope():
#         for n in range(num_layers):
#             stack_layer.add(gluon.rnn.BidirectionalCell(stack_op(hidden_size, prefix=f'l{n}_'),
#                                                 stack_op(hidden_size, prefix=f'r{n}_')))
#         stack_layer.initialize()
#     check_rnn_consistency(fused_layer, stack_layer, loss, input_size, hidden_size, bidirectional=True)




@with_seed()
@assert_raises_cudnn_not_satisfied(min_version='5.1.10')
def test_fused_gru_layer():
    input_sizes = [8]
    hidden_sizes = [8]
    num_layers = [1]
    dtypes = [bfloat16]
    for input_size, hidden_size, num_layers, dtype in product(input_sizes, hidden_sizes, num_layers, dtypes):
        loss = mx.gluon.loss.L2Loss()
        check_rnn_unidir_layer_gradients('gru', input_size, hidden_size, num_layers, loss)
        #check_rnn_bidir_layer_gradients('gru', input_size, hidden_size, num_layers, loss, dtype)
