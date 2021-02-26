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
 * Copyright (c) 2019 by Contributors
 * \file mkldnn_fc_property.cc
 * \brief Partition gragph property for FullyConnected operator
 * \author Ciyong Chen // TODO(anko)
*/

// TODO(anko) integtate it with mkldnn_fc_property.cc code or made separate MKLDNN_QUANTIZE pass (before/after?)

#ifndef MKLDNN_FC_POST_QUANTIZE_2_PROPERTY
#define MKLDNN_FC_POST_QUANTIZE_2_PROPERTY
#if MXNET_USE_MKLDNN == 1

#include <string>
#include <vector>
#include "../common.h"
#include "../../tensor/matrix_op-inl.h"
#include "mkldnn_subgraph_base-inl.h"
#include "mkldnn_fc-inl.h"

namespace mxnet {
namespace op {

class SgMKLDNNFCPostQuantize2Selector : public SubgraphSelector {
 public:
  /*! \brief pattern match status */
  enum SelectStatus {
    kFail = 0,
    kStart,
    kSuccess,
  };

 private:
  bool disable_fc_eltwise_;
  bool quantized_;
  SelectStatus status_;
  std::vector<const nnvm::Node *> matched_list_;

 public:
  explicit SgMKLDNNFCPostQuantize2Selector(const bool dis_fc_eltwise, bool quantized) :
      disable_fc_eltwise_(dis_fc_eltwise),
      quantized_(quantized) {
      //LOG(INFO) << " SgMKLDNNFCPostQuantize2Selector quantized="  << quantized_ ;
      }

  bool Select(const nnvm::Node &n, const std::shared_ptr<NodeAttr>& node_attr) override {
    if (n.op() == Op::Get("_sg_mkldnn_fully_connected") && SupportMKLDNNAttr(node_attr)) {
      auto const &fc_param = nnvm::get<MKLDNNFCFullParam>(n.attrs.parsed);
      if (fc_param.mkldnn_param.enable_float_output) { //} && fc_param.mkldnn_param.quantized) {
        status_ = disable_fc_eltwise_ ? kSuccess : kStart;
        matched_list_.clear();
        matched_list_.push_back(&n);
        return true;
      }
    }
    return false;
  }

  bool SelectInput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    return false;
  }

  bool SelectOutput(const nnvm::Node &n, const nnvm::Node &new_node) override {
    if (status_ == kFail || status_ == kSuccess || new_node.is_variable())
      return false;
    // If n isn't the last matched node, then we encoutered a internal
    // branch, we should pop out the node behind n and stop fusion.
    if (matched_list_.back() != &n) {
      if (std::find(matched_list_.begin(), matched_list_.end(), &n) !=
        matched_list_.end()) {
        while (matched_list_.back() != &n) {
          matched_list_.pop_back();
        }
      }
      status_ = kSuccess;
      return false;
    }

    switch (status_) {
      case kStart:
        // Currently, For INT8 FC fusion, only supports relu/bounded_relu(clip)/abs.
        if (new_node.op() == Op::Get("Activation")) {
          const ActivationParam &param = nnvm::get<ActivationParam>(new_node.attrs.parsed);
          if ((quantized_ && SupportQuantizedMKLDNNAct(param)) ||
              (!quantized_ && SupportMKLDNNAct(param))) {
            matched_list_.push_back(&new_node);
            status_ = kSuccess;
            return true;
          }
        }
        if (!quantized_ && (new_node.op() == Op::Get("square") ||
            new_node.op() == Op::Get("sqrt") ||
            new_node.op() == Op::Get("exp"))) {
          matched_list_.push_back(&new_node);
          status_ = kSuccess;
          return true;
        }
        if (new_node.op() == Op::Get("abs")) {
          matched_list_.push_back(&new_node);
          status_ = kSuccess;
          return true;
        }
        if (new_node.op() == Op::Get("clip")) {
          const ClipParam &param = nnvm::get<ClipParam>(new_node.attrs.parsed);
          if (param.a_min == 0.f) {
            matched_list_.push_back(&new_node);
            status_ = kSuccess;
            return true;
          }
          status_ = kSuccess;
          return false;
        }
        if (new_node.op()->name == "elemwise_add") {
          matched_list_.push_back(&new_node);
          status_ = kSuccess;
          return true;
        }
      default:
        status_ = kSuccess;
        return false;
    }
  }

  std::vector<nnvm::Node *> Filter(
      const std::vector<nnvm::Node *> &candidates) override {
    if (status_ == kFail) {
      return std::vector<nnvm::Node *>(0);
    } else {
      std::vector<nnvm::Node *> ret;
      for (auto i : matched_list_) {
        auto non_const_i = const_cast<nnvm::Node *>(i);
        if (std::find(candidates.begin(), candidates.end(), non_const_i) !=
            candidates.end()) {
          ret.push_back(non_const_i);
        }
      }
      return candidates;
    }
  }

  void Reset() override {
    CHECK_GE(matched_list_.size(), 1);
    auto new_selector = SgMKLDNNFCPostQuantize2Selector(disable_fc_eltwise_, quantized_);
    new_selector.Select(*matched_list_[0], nullptr);
    *this = new_selector;
  }
};

class SgMKLDNNFC_PostQuantize_2_Property : public SubgraphProperty {
 public:
  SgMKLDNNFC_PostQuantize_2_Property() {
    disable_fc_eltwise_ = dmlc::GetEnv("MXNET_DISABLE_MKLDNN_FUSE_FC_ELTWISE", false);
  }

  static SubgraphPropertyPtr Create() {

    static const std::string &name = "MKLDNN FullyConnected post quantization second pass"; // TODO(anko)
    auto property = std::make_shared<SgMKLDNNFC_PostQuantize_2_Property>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_MKLDNN_FC_OPT", 0)) { //TODO(anko)
      property->SetAttr<bool>("disable", true);
    }
    return property;
  }


  nnvm::ObjectPtr CreateSubgraphNode(const nnvm::Symbol &sym,
                                   const int subgraph_id = 0) const override {
    nnvm::ObjectPtr fc_node = nullptr;
    nnvm::ObjectPtr ew_add_node = nullptr;

    DFSVisit(sym.outputs, [&](const nnvm::ObjectPtr &node) {
      if (node->is_variable()) return;
      auto &sub_name = node->op()->name;
      if (sub_name == "_sg_mkldnn_fully_connected") {
        fc_node = node;
      } else if (sub_name == "elemwise_add") {
        ew_add_node = node;
      }
    });

    CHECK_NOTNULL(fc_node);
    if (ew_add_node != nullptr) {
      // TODO(anko)    change node name ?
      CHECK_NOTNULL(fc_node->attrs.subgraphs[0]);
      auto fc_orginal = fc_node->attrs.subgraphs[0]->outputs[0].node;
      if (fc_orginal->op() == Op::Get("FullyConnected")) {
        nnvm::Symbol new_sym;
        nnvm::NodeEntry &ew_input_with_fc = (ew_add_node->inputs[1].node == fc_node) ?
                                        ew_add_node->inputs[1] :
                                        ew_add_node->inputs[0];
        ew_input_with_fc.node = fc_orginal;
        //ew_input_with_fc.index = 0; // should be already set to 0
        //ew_input_with_fc.version = 0;
        new_sym.outputs.emplace_back(ew_add_node);
        fc_node->attrs.subgraphs.clear();
        fc_node->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));
        fc_node->attrs.dict["with_sum"] = "True";
        fc_node->op()->attr_parser(&(fc_node->attrs));
      }
    }
    return fc_node;
  }

  SubgraphSelectorPtr CreateSubgraphSelector() const override {
    bool quantized = HasAttr("quantize") ? GetAttr<bool>("quantize") : false;
    auto selector =
      std::make_shared<SgMKLDNNFCPostQuantize2Selector>(disable_fc_eltwise_, quantized);
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

  void ConnectSubgraphInputs(
      const nnvm::ObjectPtr n, std::vector<nnvm::NodeEntry *> *input_entries,
      std::vector<nnvm::NodeEntry> *orig_input_entries) const override {
    auto sym = n->attrs.subgraphs[0];
    auto const &fc_param = nnvm::get<MKLDNNFCFullParam>( n->attrs.parsed );
    std::unordered_set<const nnvm::Node *> node_sets;
    DFSVisit(sym->outputs, [&](const nnvm::ObjectPtr &node) {
        if (node->is_variable()) return;
        node_sets.insert(node.get());
        if (node->op()->name == "elemwise_add") {
          const size_t base_inputs = fc_param.default_param.no_bias ? 3 : 4;

          // Make sure n is the left operand of sum, if not,
          // switch sum operands sequence to ensure that
          // the extra sum operand stays in the last of inputs.
          if (node_sets.count(node->inputs[1].node.get())) {
            std::swap( node->inputs[0],  node->inputs[1]);
            // std::swap(input_entries[0][0],  input_entries[0][1]);
            // std::swap(orig_input_entries[0][0],  orig_input_entries[0][1]);

            std::rotate(input_entries->begin(),
                        input_entries->begin() + 1,
                        input_entries->begin() + base_inputs );
            std::rotate(orig_input_entries->begin(),
                        orig_input_entries->begin() + 1,
                        orig_input_entries->begin() + base_inputs);
          } else {
            std::rotate(input_entries->begin() + base_inputs - 1 ,
                        input_entries->end() - 1,
                        input_entries->end());
            std::rotate(orig_input_entries->begin() + base_inputs - 1,
                        orig_input_entries->end() - 1 ,
                        orig_input_entries->end());
          }
        }
      });
    n->inputs = *orig_input_entries;
  }

 private:
  bool disable_fc_eltwise_;
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_MKLDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_FC_PROPERTY_H_
