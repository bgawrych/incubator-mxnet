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


#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_LAYER_NORM_QUANT_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_LAYER_NORM_QUANT_PROPERTY_H_
#if MXNET_USE_MKLDNN == 1

#include <string>
#include <vector>
#include "../common.h"
#include "mkldnn_subgraph_base-inl.h"
#include "../../nn/layer_norm-inl.h"
#include "mkldnn_layer_norm_quant-inl.h"

namespace mxnet {
namespace op {

class SgMKLDNNLayerNormQuantSelector : public SubgraphSelector {
 public:
  explicit SgMKLDNNLayerNormQuantSelector() {}

  bool Select(const nnvm::Node &n, const std::shared_ptr<NodeAttr>& node_attr) override {
    return n.op() && n.op() == Op::Get("LayerNorm");
  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return false;
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    if (n.op() && n.op() == Op::Get("LayerNorm")) {
      if (new_node.op() && new_node.op() == Op::Get("_contrib_quantize_v2")) {
        auto quantize_param = nnvm::get<QuantizeV2Param>(new_node.attrs.parsed);
        if (quantize_param.min_calib_range.has_value() && quantize_param.max_calib_range.has_value())
          return true;
        else
          dont_fuse_ = true;
      } else {
        dont_fuse_ = true;
      }
    }
    return false;
  }

  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
    if (dont_fuse_)
      return std::vector<nnvm::Node *>();
    else
      return candidates;
  }

 private:
  bool dont_fuse_ = false;
};

class SgMKLDNNLayerNormQuantProperty : public SubgraphProperty {
 public:
  SgMKLDNNLayerNormQuantProperty() {}

  static SubgraphPropertyPtr Create() {
    static const std::string &name = "MKLDNN LayerNormQuant optimization pass";
    auto property = std::make_shared<SgMKLDNNLayerNormQuantProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_MKLDNN_LAYER_NORM_QUANT_OPT", 0)) {
      property->SetAttr<bool>("disable", true);
    }
    return property;
  }

  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
    nnvm::ObjectPtr n = nnvm::Node::Create();
    // This op has single output, remove duplicated.
    auto last_node = sym.outputs[0].node;
    nnvm::Symbol new_sym;
    new_sym.outputs.emplace_back(last_node);
    std::ostringstream node_name;
    node_name << "_sg_mkldnn";
  
    LayerNormQuantParam new_param;
    DFSVisit(new_sym.outputs, [&](const nnvm::ObjectPtr &node) {
      if (node->op() && node->op()->name == "LayerNorm") {
        auto ln_param = nnvm::get<LayerNormParam>(node->attrs.parsed);
        new_param.axis = ln_param.axis;
        new_param.eps = ln_param.eps;
        new_param.output_mean_var = ln_param.output_mean_var;
      } else if (node->op() && node->op()->name == "_contrib_quantize_v2") {
        auto quant_param = nnvm::get<QuantizeV2Param>(node->attrs.parsed);
        new_param.out_type = quant_param.out_type;
        new_param.min_calib_range = quant_param.min_calib_range.value();
        new_param.max_calib_range = quant_param.max_calib_range.value();
      }
    });
    node_name << "_sg_mkldnn_layer_norm_quant_" << std::to_string(subgraph_id);


    n->attrs.name = node_name.str();
    n->attrs.op = Op::Get("_sg_mkldnn_layer_norm_quant");
    CHECK(n->attrs.op);
    n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));
    n->attrs.parsed = new_param;
    return n;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    auto selector = std::make_shared<SgMKLDNNLayerNormQuantSelector>();
    return selector;
  }

  void ConnectSubgraphOutputs(
      const nnvm::ObjectPtr n,
      std::vector<nnvm::NodeEntry *> *output_entries) const override {
    // Connect all extern output entries to output[0]
    for (size_t i = 0; i < output_entries->size(); ++i) {
      auto entry_ptr = output_entries->at(i);
      *entry_ptr = nnvm::NodeEntry{n, entry_ptr->index, 0};
    }
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_LAYER_NORM_QUANT_PROPERTY_H_
