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
 * \file mkldnn_sum.cc
 * \brief
 * \author Da Zheng
*/
#include <iostream>

#include "../../operator_common.h"
#include "./mkldnn_ops-inl.h"
#include "./mkldnn_base-inl.h"

namespace mxnet {
namespace op {


#if MXNET_USE_MKLDNN == 1
static mkldnn::eltwise_forward::primitive_desc GetSigmoidFwdPd(bool is_train,
       const mkldnn::memory::desc &input_md) {

  auto cpu_engine = CpuEngine::Get()->get_engine();
  auto prop = is_train ? mkldnn::prop_kind::forward_training
                       : mkldnn::prop_kind::forward_scoring;
  auto desc = mkldnn::eltwise_forward::desc(prop, mkldnn::algorithm::eltwise_logistic,
                                            input_md, 0.f, 0.f);
  return mkldnn::eltwise_forward::primitive_desc(desc, cpu_engine);
}

class MKLDNNSigmoidFwd {
 public:
  mkldnn::eltwise_forward::primitive_desc fwd_pd;

  MKLDNNSigmoidFwd(const bool is_train, const mkldnn::memory::desc &data_md)
      : fwd_pd(GetSigmoidFwdPd(is_train, data_md)) {
    fwd_ = std::make_shared<mkldnn::eltwise_forward>(fwd_pd);
  }

  const mkldnn::eltwise_forward &GetFwd() const { return *fwd_; }

 private:
  std::shared_ptr<mkldnn::eltwise_forward> fwd_;
};

static MKLDNNSigmoidFwd &GetSigmoidForward( const bool is_train,
                                            const NDArray &in_data,
                                            const mkldnn::memory::desc &data_md) {
#if DMLC_CXX11_THREAD_LOCAL
  static thread_local std::unordered_map<OpSignature, MKLDNNSigmoidFwd, OpHash> fwds;
#else
  static MX_THREAD_LOCAL std::unordered_map<OpSignature, MKLDNNSigmoidFwd, OpHash> fwds;
#endif
  OpSignature key;
  key.AddSign(in_data);
  key.AddSign(is_train);

  auto it = fwds.find(key);
  if (it == fwds.end()) {
    MKLDNNSigmoidFwd fwd(is_train, data_md);
    it = AddToCache(&fwds, key, fwd);
  }
  return it->second;
}

void MKLDNNSigmoidForward(const nnvm::NodeAttrs& attrs, const OpContext &ctx,
                          const NDArray &input,
                          const OpReqType &req,
                          const NDArray &output) {
  //TmpMemMgr::Get()->Init(ctx.requested[0]);

  //   const NDArray &out_data = outputs[0];
  // std::vector<mkldnn::memory::desc> data_md;
  // std::vector<const mkldnn::memory *> data_mem;
  // std::vector<float> scales(num_inputs, 1);

  // data_md.reserve(num_inputs);
  // data_mem.reserve(num_inputs);

  // for (int i = 0; i < num_inputs; ++i) {
  //   const mkldnn::memory *in_mem = inputs[i].GetMKLDNNData();
  //   mkldnn::memory::desc tmp_md = in_mem->get_desc();
  //   data_md.push_back(tmp_md);
  //   data_mem.push_back(in_mem);
  // }


  const mkldnn::memory *input_mem = input.GetMKLDNNData();;
  mkldnn::memory::desc input_md = input_mem->get_desc();

  auto fwd = GetSigmoidForward(ctx.is_train, input, input_md);
  // mxnet::mkldnn_output_t out_mem = CreateMKLDNNMem(output,
  //                                                  fwd.fwd_pd.dst_desc(),
  //                                                  req[0],
  //                                                  &input);
  auto out_mem = output.GetMKLDNNData(fwd.fwd_pd.dst_desc());
  mkldnn_args_map_t net_args;
  net_args.insert({MKLDNN_ARG_DST, *out_mem});
  net_args.insert({MKLDNN_ARG_SRC, *input_mem});
  
  MKLDNNStream *stream = MKLDNNStream::Get();
  stream->RegisterPrimArgs(fwd.GetFwd(), net_args);
  
  //CommitOutput(output, out_mem);
  stream->Submit();
}
#endif

}  // namespace op
}  // namespace mxnet
