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

/*!
 * \file mkldnn_upsampling.cc
 * \brief
*/

#if MXNET_USE_MKLDNN == 1

#include <chrono>
#include <utility>
#include <mkldnn.hpp>
#include "../upsampling-inl.h"
#include "./mkldnn_base-inl.h"
namespace mxnet {
namespace op {

static mkldnn::resampling_forward::primitive_desc GetUpSamplingFwdPd(
                                  bool is_train,
                                  const mkldnn::memory &input_mem,
                                  const mkldnn::memory &output_mem) {
  mkldnn::memory::desc input_md  = input_mem.get_desc();
  mkldnn::memory::desc output_md = output_mem.get_desc();
  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto prop = is_train ? mkldnn::prop_kind::forward_training
                       : mkldnn::prop_kind::forward_inference;
  auto desc = mkldnn::resampling_forward::desc(prop,
                                               mkldnn::algorithm::resampling_nearest,
                                               input_md,
                                               output_md);
  return mkldnn::resampling_forward::primitive_desc(desc, cpu_engine);
}


bool SupportMKLDNNUpSampling(const UpSamplingParam &param,
                                    const NDArray& input,
                                    const NDArray& output) {
  const auto input_dtype = input.dtype();
  const int out_dtype = output.dtype();

  if(input.storage_type() != kDefaultStorage)
    return false;

  if(!(input_dtype == mshadow::kFloat32 || input_dtype == mshadow::kBfloat16) ||
     out_dtype != input_dtype)
  {
    return false;
  }
  
  // if (param.sample_type == up_enum::kBilinear) {
  //   CHECK_EQ(input.size(), 2U);
  //   auto kernel_dtype = input.dtype();
  //   if(!(kernel_dtype == mshadow::kFloat32 || kernel_dtype == mshadow::kBfloat16))
  //     return false;
  // }
  return true;
}


class MKLDNNUpSamplingFwd {
 public:
  mkldnn::resampling_forward::primitive_desc pd;

  MKLDNNUpSamplingFwd(const bool is_train,
                      const mkldnn::memory &input,
                      const mkldnn::memory &output)
      : pd(GetUpSamplingFwdPd(is_train, input, output)) {
    fwd_ = std::make_shared<mkldnn::resampling_forward>(pd);
  }

  const mkldnn::resampling_forward &GetFwd() const {
    return *fwd_;
  }

 private:
  std::shared_ptr<mkldnn::resampling_forward> fwd_;
};


typedef ParamOpSign<UpSamplingParam> MKLDNNUpSamplingSignature;

static MKLDNNUpSamplingFwd &GetUpSamplingFwd(const UpSamplingParam &param,
                                             const bool is_train,
                                             const NDArray &input,
                                             const NDArray &output) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<MKLDNNUpSamplingSignature,
                                         MKLDNNUpSamplingFwd,
                                         OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<MKLDNNUpSamplingSignature,
                                            MKLDNNUpSamplingFwd,
                                            OpHash> fwds;
#endif

  MKLDNNUpSamplingSignature key(param);
  key.AddSign(input);
  key.AddSign(output);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNUpSamplingFwd fwd(is_train, *(input.GetMKLDNNData()), *(output.GetMKLDNNData()));
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNLUpSamplingForward(const nnvm::NodeAttrs& attrs,
                             const OpContext &ctx,
                             const NDArray &in_data,
                             const OpReqType &req,
                             const NDArray &out_data) {
  if (req == kNullOp) return;
  CHECK_NE(req, kAddTo);

  const UpSamplingParam& param = nnvm::get<UpSamplingParam>(attrs.parsed);
  auto fwd = GetUpSamplingFwd(param, ctx.is_train, in_data, out_data);

  auto in_mem = in_data.GetMKLDNNData();
  auto out_mem = out_data.GetMKLDNNData(fwd.pd.dst_desc());
  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrimArgs(fwd.GetFwd(), {{MKLDNN_ARG_SRC, *in_mem}, {MKLDNN_ARG_DST, *out_mem}});
  stream->Submit();
}
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_USE_MKLDNN == 1
