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


#ifndef MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_TRANSFORMER_VALATT_PROPERTY_H_
#define MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_TRANSFORMER_VALATT_PROPERTY_H_
#if MXNET_USE_ONEDNN == 1

#include <map>
#include <string>
#include <vector>
#include "../common.h"
#include "../../tensor/matrix_op-inl.h"
#include "../../numpy/np_matrix_op-inl.h"
#include "../../swapaxis-inl.h"
#include "../../contrib/transformer-inl.h"
#include "mkldnn_common.h"
#include "mkldnn_transformer-inl.h"
#include "mkldnn_subgraph_base-inl.h"
/*
                 Split
                   |
               _npx_reshape
                   |
   custom_op    SwapAxis
        \        /
         batch_dot
            |
        transpose
           |
        reshape
*/

namespace mxnet {
namespace op {

#define SELFATT_QK     "_contrib_interleaved_matmul_selfatt_qk"
#define SELFATT_VALATT "_contrib_interleaved_matmul_selfatt_valatt"


bool CheckReshapeConditions(const BiDirectedNode& bi_node) {
  const nnvm::Node* rawnode = bi_node.node;
  return CheckReshapeConditions(*rawnode, 2);
}

bool CheckSwapAxisConditions(const BiDirectedNode& bi_node) {
  const nnvm::Node* rawnode = bi_node.node;
  return CheckSwapAxisConditions(*rawnode);
}

bool CheckSplitConditions(const BiDirectedNode& bi_node) {
  const nnvm::Node* rawnode = bi_node.node;
  auto const &split_params = nnvm::get<SplitParam>(rawnode->attrs.parsed);

  if (split_params.axis != -1 || split_params.sections != 3
     || split_params.indices.ndim() != 0 || split_params.squeeze_axis != 0) {
      return false;
  }

  LOG(INFO) << bi_node.outputs.size();
  if (bi_node.outputs.size() != 1) {
      return false;
  }
  return true;
}

class SgMKLDNNTransformerValAttSelector : public SubgraphSelectorV2 {
  enum InStatus {
    kFail = 0,
    kStart,
    kSecondStart,
    kIgnoreSecond,
    kSwapAx,
    kReshape,
    kSuccess
  };
  /*                 (custom_op)
             /---> kSecondStart ---\
  kStart -->                         > kSwapAx --> kReshape --> kSuccess
            \---> kIgnoreSecond ---/
            (SwapAx recognized tmp
            state to drop second input)

  Each status except kStart is connected with kFail
*/

  enum OutStatus {
    oFail = 0,
    oStart,
    oTranspose,
    oReshape,
    oSuccess
  };


 private:
  InStatus in_status_;
  OutStatus out_status_;
  std::vector<const BiDirectedNode *> matched_list_;

 public:
  bool Select(const BiDirectedNode& seed_node, const std::shared_ptr<NodeAttr>& node_attr) override {
    if (seed_node.node->op() == Op::Get("batch_dot")) {
      in_status_ = InStatus::kStart;
      out_status_ = OutStatus::oStart;
      matched_list_.clear();
      matched_list_.push_back(&seed_node);
      return true;
    }
    return false;
  }

  bool SelectInput(const BiDirectedNode &n, const BiDirectedNode &input_node) override {
    if (in_status_ == InStatus::kFail || in_status_ == InStatus::kSuccess || input_node.node->is_variable())
      return false;

    switch (in_status_) {
      case InStatus::kStart:
        if (input_node.node->op() == Op::Get("SwapAxis")) {
          in_status_ = InStatus::kIgnoreSecond;
          matched_list_.push_back(&input_node);
          return true;
        } else {
          in_status_ = InStatus::kSecondStart;
          return false;
        }
        break;
      case InStatus::kSecondStart:
        if (input_node.node->op() == Op::Get("SwapAxis")) {
          if (CheckSwapAxisConditions(input_node)) {
            in_status_ = InStatus::kSwapAx;
            matched_list_.push_back(&input_node);
            return true;
          } else {
            return false;
          }
        }
        break;
      case InStatus::kSwapAx:
        if (input_node.node->op() == Op::Get("_npx_reshape")) {
          if (CheckReshapeConditions(input_node)) {
            in_status_ = InStatus::kReshape;
            matched_list_.push_back(&input_node);
            return true;
          } else {
            return false;
          }
        }
        break;
      case InStatus::kReshape:
        if (input_node.node->op() == Op::Get("_split_v2")) {
          if (CheckSplitConditions(input_node)) {
            in_status_ = InStatus::kSuccess;
            matched_list_.push_back(&input_node);
            return true;
          }
        }
        break;
      case kIgnoreSecond:
          LOG(INFO) << "IGNORESECOND! " << input_node.node->op()->name;
        // BFS algorithm - we need to exclude single input of batch_dot (custom_op)
        in_status_ = InStatus::kSwapAx;
        return false;
      default:
        in_status_ = InStatus::kFail;
        return false;
    }
      return false;
  }

  bool SelectOutput(const BiDirectedNode &n, const BiDirectedNode &output_node) override {
    if (out_status_ == OutStatus::oFail || out_status_ == OutStatus::oSuccess || output_node.node->is_variable())
      return false;

    switch (out_status_) {
      case OutStatus::oStart:
        if (output_node.node->op() == Op::Get("_npi_transpose")) {
          auto const &transpose_params = nnvm::get<NumpyTransposeParam>(output_node.node->attrs.parsed);
          auto axes = transpose_params.axes;
          // LOG(INFO) << axes.ndim() << " "
          //           << axes[0] << " "
          //           << axes[1] << " "
          //           << axes[2] << " "
          //           << axes[3] << " ";
          if (axes.ndim() == 4 && axes[0] == 0 && axes[1] == 2 && axes[2] == 1 && axes[3] == 3) {
            out_status_ = OutStatus::oTranspose;
            matched_list_.push_back(&output_node);
            return true;
          }
        }
      case OutStatus::oTranspose:
        if (out_status_ == OutStatus::oTranspose && output_node.node->op() == Op::Get("_npx_reshape")) {
          auto const &reshape_param = nnvm::get<NumpyXReshapeParam>(output_node.node->attrs.parsed);
          auto newshape = reshape_param.newshape;
          // LOG(INFO) << newshape.ndim() << " "
          //           << newshape[0] << " "
          //           << newshape[1] << " "
          //           << newshape[2] << " ";
          if(newshape.ndim() == 3 && (newshape[0] == newshape[1] && newshape[0] == -2) && newshape[2] == -1) {
            out_status_ = OutStatus::oSuccess;
            matched_list_.push_back(&output_node);
            return true;
          }
        }
      default:
        // LOG(INFO) << output_node.node->attrs.name;
        out_status_ = OutStatus::oFail;
        return false;
    }
    
    return false;
  }
  
  std::vector<BiDirectedNode *> Filter(const std::vector<BiDirectedNode*>& candidates) override {
    LOG(INFO) << in_status_;
    if (in_status_ == InStatus::kFail || in_status_ != InStatus::kSuccess ||
        out_status_ == OutStatus::oFail || out_status_ != OutStatus::oSuccess) {
      return std::vector<BiDirectedNode *>(0);
    } else {
      std::vector<BiDirectedNode *> ret;
      for (auto i : matched_list_) {
        auto non_const_i = const_cast<BiDirectedNode *>(i);
        if (std::find(candidates.begin(), candidates.end(), non_const_i) !=
            candidates.end()) {
          ret.push_back(non_const_i);
        }
      }
      return ret;
    }
  }

  void Reset() override {
    CHECK_GE(matched_list_.size(), 1);
    auto new_selector = SgMKLDNNTransformerValAttSelector();
    new_selector.Select(*matched_list_[0], nullptr);
    *this = new_selector;
  }

};

class SgMKLDNNTransformerValAttProperty : public SubgraphProperty {
 public:
  SgMKLDNNTransformerValAttProperty() {}

  static SubgraphPropertyPtr Create() {
    static const std::string &name = "MKLDNN Transformer optimization pass";
    auto property = std::make_shared<SgMKLDNNTransformerValAttProperty>();
    property->SetAttr<std::string>("property_name", name);
    property->SetAttr<bool>("inference_only", true);
    if (dmlc::GetEnv("MXNET_DISABLE_MKLDNN_TRANSFORMER_OPT", 0)) {
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
    std::string op_name;
    // LOG(INFO) << "VALLATT SUBGRAPH";
    DFSVisit(new_sym.outputs, [&](const nnvm::ObjectPtr &node) {
      // if(node->op()) {
      //   LOG(INFO) << node->op()->name << " " << node->attrs.name;
      // } else {
      //     LOG(INFO) << "VAR " << node->attrs.name;
      // }
      if((node->op() == Op::Get("_npx_reshape"))) {
        auto const &reshape_param = nnvm::get<NumpyXReshapeParam>(node->attrs.parsed);
        if(reshape_param.newshape.ndim() == 4)
          n->attrs.dict["heads"] = std::to_string(reshape_param.newshape[2]);
      }
    });
    node_name << "sg_mkldnn_selfatt_valatt_" << subgraph_id;
    n->attrs.name = node_name.str();
    n->attrs.op = Op::Get("sg_mkldnn_selfatt_valatt");
    CHECK(n->attrs.op);
    //n->attrs.subgraphs.emplace_back(std::make_shared<nnvm::Symbol>(new_sym));
    n->op()->attr_parser(&(n->attrs));
    return n;
  }

  SubgraphSelectorV2Ptr CreateSubgraphSelectorV2() const override {
    auto selector = std::make_shared<SgMKLDNNTransformerValAttSelector>();
    return selector;
  }

  // void ConnectSubgraphOutputs(
  //     const nnvm::ObjectPtr n,
  //     std::vector<nnvm::NodeEntry *> *output_entries) const override {
  //   // Connect all extern output entries to output[0]
  //   for (size_t i = 0; i < output_entries->size(); ++i) {
  //     auto entry_ptr = output_entries->at(i);
  //     *entry_ptr = nnvm::NodeEntry{n, entry_ptr->index, 0};
  //   }
  // }

  // virtual void ConnectSubgraphInputs(const nnvm::ObjectPtr subgraph_node,
  //                                    std::vector<nnvm::NodeEntry*>* input_entries,
  //                                    std::vector<nnvm::NodeEntry>* orig_input_entries) const {
  //   for (size_t i = 0; i < orig_input_entries->size(); ++i) {
  //     auto entry_ptr = orig_input_entries->at(i);
  //     LOG(INFO) << entry_ptr.node->attrs.name << " " << entry_ptr.node->op()->name;
  //   }
  //   subgraph_node->inputs = *orig_input_entries;
  // }

  // virtual void ConnectSubgraphInputs(const nnvm::ObjectPtr subgraph_node,
  //                                    std::vector<nnvm::NodeEntry*>* input_entries,
  //                                    std::vector<nnvm::NodeEntry>* orig_input_entries) const {
  //   LOG(INFO) << "ConnectSubgraphInputs";
  //   for (size_t i = 0; i < orig_input_entries->size(); ++i) {
  //     auto entry_ptr = orig_input_entries->at(i);
  //     if((entry_ptr.node->op() == nullptr)) {
  //       LOG(INFO) << entry_ptr.node->attrs.name << "null";
  //     } else {
  //       LOG(INFO) << entry_ptr.node->attrs.name << entry_ptr.node->op()->name;
  //     }
  //     //*entry_ptr = nnvm::NodeEntry{n, entry_ptr->index, 0};
  //   }
  //   subgraph_node->inputs.resize(1);
  //   subgraph_node->inputs[0] = orig_input_entries->at(0).node->inputs[0];
  //   // *((*input_entries)[0]) = orig_input_entries->at(0).node->inputs[0];
  //   // input_entries->pop_back();
  //   // (*orig_input_entries)[0] = orig_input_entries->at(0).node->inputs[0];
  //   // orig_input_entries->pop_back();
  // }
};

}  // namespace op
}  // namespace mxnet

#endif  // if MXNET_USE_ONEDNN == 1
#endif  // MXNET_OPERATOR_SUBGRAPH_MKLDNN_MKLDNN_TRANSFORMER_VALATT_PROPERTY_H_
