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

#ifndef MXNET_OPERATOR_CONTRIB_MKLDNN_LAYER_NORM_QUANT_INL_H_
#define MXNET_OPERATOR_CONTRIB_MKLDNN_LAYER_NORM_QUANT_INL_H_

#include "../../mxnet_op.h"
#include "../../quantization/quantization_utils.h"

namespace mxnet {
namespace op {

struct LayerNormQuantParam : public dmlc::Parameter<LayerNormQuantParam> {
  int axis;
  float eps;
  bool output_mean_var;
  // Quant params
  int out_type;
  float min_calib_range;
  float max_calib_range;
  DMLC_DECLARE_PARAMETER(LayerNormQuantParam) {
    DMLC_DECLARE_FIELD(axis).set_default(-1)
      .describe("The axis to perform layer normalization. "
                "Usually, this should be be axis of the channel dimension. "
                "Negative values means indexing from right to left.");
    DMLC_DECLARE_FIELD(eps).set_default(1e-5f)
      .describe("An `epsilon` parameter to prevent division by 0.");
    DMLC_DECLARE_FIELD(output_mean_var).set_default(false)
      .describe("Output the mean and std calculated along the given axis.");
    DMLC_DECLARE_FIELD(out_type)
      .add_enum("auto", QuantizeOutType::kAuto)
      .add_enum("int8", QuantizeOutType::kInt8)
      .add_enum("uint8", QuantizeOutType::kUint8)
      .set_default(QuantizeOutType::kInt8)
      .describe("Output data type. `auto` can be specified to automatically determine output type "
                "according to min_calib_range.");
    DMLC_DECLARE_FIELD(min_calib_range)
      .set_default(0.0f)
      .describe("The minimum scalar value in the form of float32. If present, it will be used to "
                "quantize the fp32 data into int8 or uint8.");
    DMLC_DECLARE_FIELD(max_calib_range)
      .set_default(0.0f)
      .describe("The maximum scalar value in the form of float32. If present, it will be used to "
                "quantize the fp32 data into int8 or uint8.");
  }
};

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_MKLDNN_LAYER_NORM_QUANT_INL_H_
