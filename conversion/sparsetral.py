from __future__ import annotations

from .base import ModelBase, gguf

from .llama import LlamaModel

from typing import TYPE_CHECKING

import re 
import torch

if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register(
    "modeling_sparsetral.MistralForCausalLM")
class SparsetralModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.LLAMA
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "topk" in self.hparams:
            self.hparams["num_experts_per_tok"] = self.hparams["topk"]

        self.hparams["scoring_func"] = "softmax"

    _moe_adapter_experts = None
    
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        if ".mlp.moe_adapter.experts." not in name:
            yield from super().modify_tensors(data_torch, name, bid)
            return
        
        assert bid is not None

        # match the pattern with the tensor names to extract layer_id and expert_id
        match = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.moe_adapter\.experts\.(\d+)\.(adapter_up|adapter_down)\.weight", name)

        if match is None:
            raise ValueError(f"Cannot map tensor {name!r}")
        
        layer_id = int(match.group(1))

        expert_id = int(match.group(2))

        proj_type = match.group(3)

        if layer_id != bid:
            raise ValueError(f"Tensor has layer_id {layer_id} but converter expected bid {bid}")
        
        n_experts = self.hparams["num_experts"]

        if expert_id >= n_experts:
            raise ValueError(f"Invalid expert_id. The expert_id {expert_id} is larger than the total number of experts {n_experts}")
        
        # temporary collector that makes all experts with the same projection type stacked together
        
        if self._moe_adapter_experts is None:
            # create one dict per layer
            self._moe_adapter_experts = [{} for _ in range(self.block_count)]

        # for the current layer bid, remember tensor named `name`
        self._moe_adapter_experts[bid][name] = data_torch

        expected = [
            f"model.layers.{bid}.mlp.moe_adapter.experts.{xid}.{proj_type}.weight"
            for xid in range(n_experts)
        ]

        # if any expected tensor has not arrived, wait for more tensors
        if not all(ename in self._moe_adapter_experts[bid] for ename in expected):
            return
        
        datas = [self._moe_adapter_experts[bid][ename] for ename in expected]

        merged_tensor = torch.stack(datas, dim=0)

        for ename in expected:
            del self._moe_adapter_experts[bid][ename]
        
        merged_name = f"model.layers.{bid}.mlp.moe_adapter.experts.{proj_type}.weight"

        yield from super().modify_tensors(merged_tensor, merged_name, bid)
    
    # check whether any adapter expert tensors entered the buffer but never came out as merged tensors
    def prepare_tensors(self):
        super().prepare_tensors()

        if self._moe_adapter_experts is not None:
            experts = []
            for layer_buffer in self._moe_adapter_experts:
                # add all remaining tensor names from the layer into the list
                experts.extend(layer_buffer.keys())
            
            # if any tensor remains, something was not merged
            if len(experts) > 0:
                raise ValueError(f"Unprocessed Sparsetral adapter experts: {experts}")

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        self.gguf_writer.add_adapter_feed_forward_length(self.hparams["adapter_dim"])