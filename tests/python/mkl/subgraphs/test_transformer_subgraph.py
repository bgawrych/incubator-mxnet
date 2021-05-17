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

import copy
import mxnet as mx
import pytest
from subgraph_common import check_fusion, check_neg_fusion, check_quantize
from subgraph_common import CustomNormalInit, DATA_SHAPE, RELU6, TailNegBlock
from subgraph_common import DATA_SHAPE, SG_PASS_NAME, QUANTIZE_SG_PASS_NAME
from mxnet.contrib import quantization
from mxnet.gluon import nn
from mxnet.test_utils import assert_almost_equal, assert_almost_equal_with_err
from mxnet.util import use_np
from mxnet import np, npx
import math

@use_np
def test_self_attention():

  class MultiHeadAttention(nn.HybridBlock):
    def __init__(self, units, num_heads, dtype='float32', **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self._units = units
        self._num_heads = num_heads
        self._fc = nn.Dense(in_units=self._units, units=3*self._units, flatten=False, dtype=dtype)
        self._scale = math.sqrt(self._units // self._num_heads)

    def hybrid_forward(self, F, x, mask):
        out = self._fc(x)
        query, key, value = F.np.split(out, 3, axis=-1)
        query = F.npx.reshape(query, (-2, -2, self._num_heads, -1))
        key = F.npx.reshape(key, (-2, -2, self._num_heads, -1))
        value = F.npx.reshape(value, (-2, -2, self._num_heads, -1))
        scores = F.npx.batch_dot(F.np.swapaxes(query, 1, 2), F.np.swapaxes(key, 1, 2),
                               transpose_b=True)
        mask = F.np.expand_dims(mask, axis=1).astype(np.bool)
        attn_weights = F.npx.masked_softmax(scores, mask=mask.astype(np.bool),
                                            axis=-1, temperature=self._scale)
        attn_weights = F.npx.dropout(attn_weights, p=0.1)
        context_vec = F.npx.batch_dot(attn_weights,
                                     F.np.swapaxes(value, 1, 2)).transpose((0, 2, 1, 3))
        context_vec = F.npx.reshape(context_vec, (-2, -2, -1))
        return context_vec, [scores, attn_weights]

  batch_size = 24
  seq_length = 384
  units = 768
  num_heads = 4
  net = MultiHeadAttention(units, num_heads)
  in_data = mx.np.random.normal(size=[batch_size, seq_length, units ], dtype='float32')
  mask = mx.np.random.normal(size=[batch_size, seq_length, seq_length ], dtype='float32')

  net.initialize()
  print(in_data.shape)
  net.hybridize()
  ref_ctx_vec, ref_out = net(in_data, mask)
  net.optimize_for(in_data, mask, backend="MKLDNN")
  ctx_vec, out = net(in_data, mask)

  assert_almost_equal(ctx_vec.asnumpy(), ref_ctx_vec.asnumpy())
  for i in range(len(out)):
    assert_almost_equal(out[i].asnumpy(), ref_out[i].asnumpy())
  net.export("CHECK")
  print(out)