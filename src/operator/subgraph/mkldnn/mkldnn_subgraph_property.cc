/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#if MXNET_USE_MKLDNN == 1

#include "mkldnn_conv_property.h"
#include "mkldnn_fc_property.h"
#include "mkldnn_post_quantize_property.h"
#include "mkldnn_fc_post_quantize_property.h"
#include "mkldnn_elemwisemul_post_quantize_property.h"
#include "mkldnn_post_quantize_align_scale_property.h"
#include "mkldnn_transformer_property.h"
#include "mkldnn_transformer_post_quantize_property.h"
#include "mkldnn_asym_quant_fc_property.h"
#include "mkldnn_interleaved_u8_fc_property.h"
#include "mkldnn_fc_u8_fc_property.h"

namespace mxnet {
namespace op {

MXNET_REGISTER_SUBGRAPH_BACKEND(MKLDNN_BERT)
.set_attr("enable", MKLDNNEnvSet())
.set_attr("context", Context::CPU());

MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN_BERT, SgMKLDNNTransformerProperty);

MXNET_REGISTER_SUBGRAPH_BACKEND(MKLDNN_BERT_QUANTIZE)
.set_attr("enable", MKLDNNEnvSet())
.set_attr("context", Context::CPU());

MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN_BERT_QUANTIZE, SgMKLDNNTransformerPostQuantizeProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN_BERT_QUANTIZE, SgMKLDNNAsymQuantFCProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN_BERT_QUANTIZE, SgMKLDNNInterleavedu8FCProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN_BERT_QUANTIZE, SgMKLDNNFCu8FCProperty);

MXNET_REGISTER_SUBGRAPH_BACKEND(MKLDNN)
.set_attr("enable", MKLDNNEnvSet())
.set_attr("context", Context::CPU());

MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN, SgMKLDNNConvProperty);

#endif  // MXNET_USE_MKLDNN == 1
#if MXNET_USE_MKLDNN == 1
MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN, SgMKLDNNFCProperty);
#endif  // MXNET_USE_MKLDNN == 1
#if MXNET_USE_MKLDNN == 1
MXNET_REGISTER_SUBGRAPH_BACKEND(MKLDNN_QUANTIZE)
.set_attr("context", Context::CPU());

MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN_QUANTIZE, SgMKLDNNConvProperty)
.set_attr("quantize", true);

#endif  // MXNET_USE_MKLDNN == 1
#if MXNET_USE_MKLDNN == 1

MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN_QUANTIZE, SgMKLDNNFCProperty)
.set_attr("quantize", true);
#endif  // MXNET_USE_MKLDNN == 1
#if MXNET_USE_MKLDNN == 1
MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN_QUANTIZE, SgMKLDNNPostQuantizeProperty);
#endif  // MXNET_USE_MKLDNN == 1

#if MXNET_USE_MKLDNN == 1
MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN_QUANTIZE, SgMKLDNNFCPostQuantizeProperty);
MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN_QUANTIZE, ElemwiseMulPostQuantizeProperty);

MXNET_REGISTER_SUBGRAPH_PROPERTY(MKLDNN_QUANTIZE, SgMKLDNNPostQuantizeAlignScaleProperty);
#endif  // MXNET_USE_MKLDNN == 1
#if MXNET_USE_MKLDNN == 1
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
