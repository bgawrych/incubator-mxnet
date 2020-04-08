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
 * \file np_init_op.cc
 * \brief Implementation of the API of functions in src/operator/numpy/np_init_op.cc
 */
#include <dmlc/optional.h>
#include <mxnet/api_registry.h>
#include <mxnet/runtime/packed_func.h>
#include "../utils.h"
#include "../../../operator/tensor/init_op.h"
#include "../../../operator/numpy/np_init_op.h"

namespace mxnet {

MXNET_REGISTER_API("_npi.zeros")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_zeros");
  nnvm::NodeAttrs attrs;
  op::InitOpParam param;
  if (args[0].type_code() == kDLInt) {
    param.shape = TShape(1, args[0].operator int64_t());
  } else {
    param.shape = TShape(args[0].operator ObjectRef());
  }
  if (args[1].type_code() == kNull) {
    param.dtype = mshadow::kFloat32;
  } else {
    param.dtype = String2MXNetTypeWithBool(args[1].operator std::string());
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::InitOpParam>(&attrs);
  if (args[2].type_code() != kNull) {
    attrs.dict["ctx"] = args[2].operator std::string();
  }
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, 0, nullptr, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.full_like")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_full_like");
  nnvm::NodeAttrs attrs;
  op::FullLikeOpParam param;
  param.fill_value = args[1].operator double();
  if (args[2].type_code() == kNull) {
    param.dtype = dmlc::nullopt;
  } else {
    param.dtype = String2MXNetTypeWithBool(args[2].operator std::string());
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  if (args[3].type_code() != kNull) {
    attrs.dict["ctx"] = args[3].operator std::string();
  }
  SetAttrDict<op::FullLikeOpParam>(&attrs);
  NDArray* out = args[4].operator mxnet::NDArray*();
  NDArray** outputs = out == nullptr ? nullptr : &out;
  int num_outputs = out != nullptr;
  NDArray* inputs[] = {args[0].operator mxnet::NDArray*()};
  int num_inputs = 1;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, outputs);
  if (out) {
    *ret = PythonArg(4);
  } else {
    *ret = ndoutputs[0];
  }
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.indices")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_indices");
  nnvm::NodeAttrs attrs;
  op::IndicesOpParam param;
  // param.dimensions
  if (args[0].type_code() == kDLInt) {
    param.dimensions = TShape(1, args[0].operator int64_t());
  } else {
    param.dimensions = TShape(args[0].operator ObjectRef());
  }
  // param.dtype
  if (args[1].type_code() == kNull) {
    param.dtype = mshadow::kInt32;
  } else {
    param.dtype = String2MXNetTypeWithBool(args[1].operator std::string());
  }
  attrs.parsed = std::move(param);
  attrs.op = op;
  SetAttrDict<op::IndicesOpParam>(&attrs);
  // param.ctx
  if (args[2].type_code() != kNull) {
    attrs.dict["ctx"] = args[2].operator std::string();
  }
  int num_inputs = 0;
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, nullptr, &num_outputs, nullptr);
  *ret = ndoutputs[0];
});

MXNET_REGISTER_API("_npi.atleast_1d")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_atleast_1d");
  nnvm::NodeAttrs attrs;
  op::AtleastNDParam param;
  int args_size = args.size();
  param.num_args = args_size;
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::AtleastNDParam>(&attrs);
  int num_inputs = args_size;
  std::vector<NDArray*> inputs_vec(args_size, nullptr);
  for (int i = 0; i < args_size; ++i) {
    inputs_vec[i] = args[i].operator mxnet::NDArray*();
  }
  NDArray** inputs = inputs_vec.data();
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  std::vector<NDArrayHandle> ndarray_handles;
  ndarray_handles.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    ndarray_handles.emplace_back(ndoutputs[i]);
  }
  *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
});

MXNET_REGISTER_API("_npi.atleast_2d")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_atleast_2d");
  nnvm::NodeAttrs attrs;
  op::AtleastNDParam param;
  int args_size = args.size();
  param.num_args = args_size;
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::AtleastNDParam>(&attrs);
  int num_inputs = args_size;
  std::vector<NDArray*> inputs_vec(args_size, nullptr);
  for (int i = 0; i < args_size; ++i) {
    inputs_vec[i] = args[i].operator mxnet::NDArray*();
  }
  NDArray** inputs = inputs_vec.data();
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  std::vector<NDArrayHandle> ndarray_handles;
  ndarray_handles.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    ndarray_handles.emplace_back(ndoutputs[i]);
  }
  *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
});

MXNET_REGISTER_API("_npi.atleast_3d")
.set_body([](runtime::MXNetArgs args, runtime::MXNetRetValue* ret) {
  using namespace runtime;
  const nnvm::Op* op = Op::Get("_npi_atleast_3d");
  nnvm::NodeAttrs attrs;
  op::AtleastNDParam param;
  int args_size = args.size();
  param.num_args = args_size;
  attrs.parsed = param;
  attrs.op = op;
  SetAttrDict<op::AtleastNDParam>(&attrs);
  int num_inputs = args_size;
  std::vector<NDArray*> inputs_vec(args_size, nullptr);
  for (int i = 0; i < args_size; ++i) {
    inputs_vec[i] = args[i].operator mxnet::NDArray*();
  }
  NDArray** inputs = inputs_vec.data();
  int num_outputs = 0;
  auto ndoutputs = Invoke(op, &attrs, num_inputs, inputs, &num_outputs, nullptr);
  std::vector<NDArrayHandle> ndarray_handles;
  ndarray_handles.reserve(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    ndarray_handles.emplace_back(ndoutputs[i]);
  }
  *ret = ADT(0, ndarray_handles.begin(), ndarray_handles.end());
});

}  // namespace mxnet
