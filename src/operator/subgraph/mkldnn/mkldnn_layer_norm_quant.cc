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

#include "../../nn/layer_norm-inl.h"
#include "mkldnn_layer_norm_quant-inl.h"
#include "../../quantization/quantization_utils.h"

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(LayerNormQuantParam);

template <typename OutType>
void LayerNormQuantCPUKernel(size_t width,
                             size_t instances,
                             float eps,
                             const float *data,
                             const float *gamma,
                             const float *beta,
                             OutType *out,
                             float scale = 1.0f) {
  // Parallelize over independent instances to normalize.
  // MSVC says index variable in OpenMP 'for' statement must have signed integral type.
  const mshadow::index_t signed_instances = static_cast<mshadow::index_t>(instances);
#pragma omp parallel for
  for (nnvm::dim_t j = 0; j < signed_instances; ++j) {
    const float *from = data + j * width;

    // Sum the values to compute mean.
    float sum = 0.f;
#pragma omp simd reduction(+ : sum)
    for (size_t i = 0; i < width; ++i) {
      sum += from[i];
    }
    float mean_value = sum / width;

    // Sum squares from mean to compute stddev.
    float squares = 0.f;
#pragma omp simd reduction(+ : squares)
    for (size_t i = 0; i < width; ++i) {
      float off = from[i] - mean_value;
      squares += off * off;
    }
    float sigma = std::sqrt(squares / width + eps);

    // Write normalized values.
    OutType *to = out + j * width;
#pragma omp simd
    for (size_t i = 0; i < width; ++i) {
      to[i] = static_cast<OutType>(((from[i] - mean_value) * gamma[i] / sigma + beta[i]) * scale + 0.5); // Rounding
    }
  }
}

bool LayerNormQuantCPU(const nnvm::NodeAttrs& attrs,
                  const OpContext& ctx, const std::vector<TBlob>& inputs,
                  const std::vector<OpReqType>& req,
                  const std::vector<TBlob>& outputs) {
  const LayerNormQuantParam& param = nnvm::get<LayerNormQuantParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 3U);
  CHECK_EQ(outputs.size(), 3U);

  switch (req[layernorm::kOut]) {
    case kNullOp:
      return true;
    case kWriteTo:
      break;
    case kWriteInplace:
      break;
    default:
      // Should only be kAddTo, which isn't supported by the others implementation either.
      return false;
  }
  // Axis must be the last one.
  int axis = GetRealAxis(param.axis, inputs[layernorm::kData].ndim());
  if (axis != inputs[layernorm::kData].ndim() - 1) {
    return false;
  }

  const float scale = GetQuantizeScale(outputs[layernorm::kOut].type_flag_, param.min_calib_range, param.max_calib_range);
  const size_t moments_size = inputs[layernorm::kData].Size() / inputs[layernorm::kData].shape_[axis];

  MSHADOW_TYPE_SWITCH(outputs[layernorm::kOut].type_flag_, OutType, {
    LayerNormQuantCPUKernel<OutType>(
      inputs[layernorm::kData].shape_[axis],
      moments_size,
      param.eps,
      inputs[layernorm::kData].dptr<float>(),
      inputs[layernorm::kGamma].dptr<float>(),
      inputs[layernorm::kBeta].dptr<float>(),
      outputs[layernorm::kOut].dptr<OutType>(),
      scale);
  });

  return true;
}

void LayerNormQuantCompute(const nnvm::NodeAttrs& attrs,
                           const OpContext& ctx, const std::vector<TBlob>& inputs,
                           const std::vector<OpReqType>& req,
                           const std::vector<TBlob>& outputs) {
  LayerNormQuantCPU(attrs, ctx, inputs, req, outputs);
}

static bool LayerNormQuantShape(const nnvm::NodeAttrs& attrs,
                           mxnet::ShapeVector *in_shape,
                           mxnet::ShapeVector *out_shape) {
  using namespace mshadow;
  CHECK_EQ(in_shape->size(), 3U) << "Input:[data, gamma, beta]";
  const mxnet::TShape &dshape = in_shape->at(layernorm::kData);
  if (!mxnet::ndim_is_known(dshape)) {
    return false;
  }
  SHAPE_ASSIGN_CHECK(*out_shape, 0, dshape);
  SHAPE_ASSIGN_CHECK(*out_shape, 1, mxnet::TShape({1}));
  SHAPE_ASSIGN_CHECK(*out_shape, 2, mxnet::TShape({1}));
  return true;
}

static bool LayerNormQuantInferType(const nnvm::NodeAttrs &attrs,
                                                      std::vector<int> *in_types,
                                                      std::vector<int> *out_types) {
  const auto &param = nnvm::get<LayerNormQuantParam>(attrs.parsed);

  if (param.out_type == QuantizeOutType::kAuto) {
    if (param.min_calib_range >= 0.0) {
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kUint8);
    } else {
      TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);
    }
  } else if (param.out_type == QuantizeOutType::kInt8) {
    TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kInt8);
  } else if (param.out_type == QuantizeOutType::kUint8) {
    TYPE_ASSIGN_CHECK(*out_types, 0, mshadow::kUint8);
  } else {
    LOG(FATAL) << "Unsupported out_type in params: " <<param.out_type;
  }

  TYPE_ASSIGN_CHECK(*out_types, 1, mshadow::kFloat32);
  TYPE_ASSIGN_CHECK(*out_types, 2, mshadow::kFloat32);
  return true;
}

NNVM_REGISTER_OP(_sg_mkldnn_layer_norm_quant)
.describe(R"code(_sg_mkldnn_layer_norm_quant)code" ADD_FILELINE)
.set_num_inputs(3)
.set_num_outputs(3)
.set_attr_parser(ParamParser<LayerNormQuantParam>)
.set_attr<nnvm::FListInputNames>("FListInputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"data", "gamma", "beta"};
})
.set_attr<nnvm::FListOutputNames>("FListOutputNames",
    [](const NodeAttrs& attrs) {
  return std::vector<std::string>{"output", "data_min", "data_max"};
})
.set_attr<mxnet::FInferShape>("FInferShape", LayerNormQuantShape)
.set_attr<nnvm::FInferType>("FInferType", LayerNormQuantInferType)
.set_attr<FCompute>("FCompute<cpu>", LayerNormQuantCompute)
.set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
  return std::vector<std::pair<int, int> >{{0, 0}};
})
.set_attr<FResourceRequest>("FResourceRequest", [](const NodeAttrs& n) {
  return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
})
.set_attr<THasDeterministicOutput>("THasDeterministicOutput", true)
.add_argument("data", "NDArray-or-Symbol", "Input data to layer normalization")
.add_argument("gamma", "NDArray-or-Symbol", "gamma array")
.add_argument("beta", "NDArray-or-Symbol", "beta array")
.add_arguments(LayerNormQuantParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet

#endif
