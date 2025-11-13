# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union

import torch
import logging
from megatron.core import parallel_state, tensor_parallel, utils
from megatron.core.debug_utils import debug_log, is_debug_enabled, debug_assert
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig

try:
    import transformer_engine as te  # pylint: disable=unused-import

    from megatron.core.extensions.transformer_engine import te_checkpoint

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

logger = logging.getLogger(__name__)


@dataclass
class MoESubmodules:
    """MoE Layer Submodule spec"""

    experts: Union[ModuleSpec, type] = None
    shared_experts: Union[ModuleSpec, type] = None


class BaseMoELayer(MegatronModule, ABC):
    """Base class for a mixture of experts layer.

    Args:
        config (TransformerConfig): Configuration object for the transformer model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super(BaseMoELayer, self).__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.ep_group = pg_collection.ep
        # use pg_collection.expt_tp_group as tensor parallel group in this module.
        self.attn_tp_group = pg_collection.tp
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)
        debug_assert(ep_size > 0, "Expected non-negative expert parallel size")

        debug_assert(self.config.num_moe_experts % ep_size == 0)
        self.num_local_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_experts

        self.use_shared_expert = self.config.moe_shared_expert_intermediate_size is not None
        self.shared_expert_overlap = self.config.moe_shared_expert_overlap

        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
        debug_assert(all(map(lambda x: x < self.config.num_moe_experts, self.local_expert_indices)))
        self.router: TopKRouter = None
        self.experts = None
        self.shared_experts = None
        self.token_dispatcher: Optional[MoETokenDispatcher] = None
        self.layer_number = layer_number

    @abstractmethod
    def forward(self, hidden_states):
        """Forward method for the MoE layer."""
        pass

    def set_layer_number(self, layer_number: int):
        """Set the layer number for the MoE layer."""
        self.layer_number = layer_number
        self.router.set_layer_number(layer_number)


class MoELayer(BaseMoELayer):
    """Mixture of Experts layer.

    This layer implements a Mixture of Experts model, where each token is routed to a
    subset of experts. This implementation supports different token dispatching
    strategies such as All-to-All and All-Gather.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: Optional[MoESubmodules] = None,
        layer_number: Optional[int] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        self.submodules = submodules
        # TODO(Hepteract): delete the usage of the global parallel_state.
        # Initialize process groups with the global parallel_state.
        if pg_collection is None:
            pg_collection = get_default_pg_collection()
        super(MoELayer, self).__init__(
            config=config, layer_number=layer_number, pg_collection=pg_collection
        )
        self.moe_layer_recompute = (
            config.recompute_granularity == 'selective' and "moe" in config.recompute_modules
        )
        self.shared_experts_recompute = (
            config.recompute_granularity == 'selective'
            and "shared_experts" in config.recompute_modules
        )

        # Initialize router
        self.router = TopKRouter(config=self.config, pg_collection=pg_collection)

        # Initialize token dispatcher
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_experts,
                self.local_expert_indices,
                config=self.config,
                pg_collection=pg_collection,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {config.moe_token_dispatcher_type}"
            )

        # Initialize experts
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
            pg_collection=pg_collection,
        )

        # Initialize shared experts
        if self.use_shared_expert:
            self.shared_experts = build_module(
                self.submodules.shared_experts, config=self.config, pg_collection=pg_collection
            )
            if self.shared_expert_overlap:
                self.token_dispatcher.set_shared_experts(self.shared_experts)

    def router_and_preprocess(self, hidden_states: torch.Tensor):
        """Compute and preprocess token routing for dispatch.

        This method uses the router to determine which experts to send each token to,
        producing routing probabilities and a mapping. It then preprocesses the
        hidden states and probabilities for the token dispatcher. The original
        hidden states are returned as a residual connection.
        """
        residual = hidden_states
        if hidden_states is not None:
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 178, hidden_states: {hidden_states.shape}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 178, hidden_states has nan: {torch.isnan(hidden_states).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 178, hidden_states has inf: {torch.isinf(hidden_states).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 178, hidden_states has -inf: {torch.isneginf(hidden_states).any()}")
            if is_debug_enabled():

                if torch.isnan(hidden_states).any():

                    raise ValueError(f"hidden_states contains NaN values at line 178. Shape: {hidden_states.shape}")
                if torch.isinf(hidden_states).any():
                    raise ValueError(f"hidden_states contains inf values at line 178. Shape: {hidden_states.shape}")
                if torch.isneginf(hidden_states).any():
                    raise ValueError(f"hidden_states contains -inf values at line 178. Shape: {hidden_states.shape}")
        else:
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 178, hidden_states is None")
        probs, routing_map = self.router(hidden_states)
        if probs is not None:
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 182, probs: {probs.shape}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 182, probs has nan: {torch.isnan(probs).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 182, probs has inf: {torch.isinf(probs).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 182, probs has -inf: {torch.isneginf(probs).any()}")
            if is_debug_enabled():

                if torch.isnan(probs).any():

                    raise ValueError(f"probs contains NaN values at line 182. Shape: {probs.shape}")
        hidden_states, probs = self.token_dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        if hidden_states is not None:
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 186, hidden_states: {hidden_states.shape}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 186, hidden_states has nan: {torch.isnan(hidden_states).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 186, hidden_states has inf: {torch.isinf(hidden_states).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 186, hidden_states has -inf: {torch.isneginf(hidden_states).any()}")
            if is_debug_enabled():

                if torch.isnan(hidden_states).any():

                    raise ValueError(f"hidden_states contains NaN values at line 186. Shape: {hidden_states.shape}")
                if torch.isinf(hidden_states).any():
                    raise ValueError(f"hidden_states contains inf values at line 186. Shape: {hidden_states.shape}")
                if torch.isneginf(hidden_states).any():
                    raise ValueError(f"hidden_states contains -inf values at line 186. Shape: {hidden_states.shape}")
        if probs is not None:
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 186, probs: {probs.shape}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 186, probs has nan: {torch.isnan(probs).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 186, probs has inf: {torch.isinf(probs).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 186, probs has -inf: {torch.isneginf(probs).any()}")
            if is_debug_enabled():

                if torch.isnan(probs).any():

                    raise ValueError(f"probs contains NaN values at line 186. Shape: {probs.shape}")
                if torch.isinf(probs).any():
                    raise ValueError(f"probs contains inf values at line 186. Shape: {probs.shape}")
                if torch.isneginf(probs).any():
                    raise ValueError(f"probs contains -inf values at line 186. Shape: {probs.shape}")
        else:
            if is_debug_enabled():
                debug_log(logger, logging.INFO, f"in moe layer line 186, probs is None")
        return hidden_states, probs, residual

    def dispatch(self, hidden_states: torch.Tensor, probs: torch.Tensor):
        """Dispatches tokens to assigned expert ranks via communication.
        This method performs the actual communication (e.g., All-to-All) to distribute
        tokens and their associated probabilities to the devices hosting their assigned
        experts.
        """
        return self.token_dispatcher.token_dispatch(hidden_states, probs)

    def shared_experts_compute(self, hidden_states: torch.Tensor):
        """Computes the output of the shared experts.

        If a shared expert is configured and not overlapped with communication,
        it is computed here.
        """
        shared_expert_output = None
        if self.use_shared_expert and not self.shared_expert_overlap:
            # Compute the shared expert separately when not overlapped with communication.
            if self.shared_experts_recompute:
                if self.config.fp8:
                    shared_expert_output = te_checkpoint(
                        self.shared_experts,
                        False,
                        tensor_parallel.random.get_cuda_rng_tracker,
                        parallel_state.get_tensor_model_parallel_group(),
                        hidden_states,
                    )
                else:
                    shared_expert_output = tensor_parallel.checkpoint(
                        self.shared_experts, False, hidden_states
                    )
            else:
                shared_expert_output = self.shared_experts(hidden_states)

        return shared_expert_output

    def routed_experts_compute(
        self, hidden_states: torch.Tensor, probs: torch.Tensor, residual: torch.Tensor
    ):
        """Computes the output of the routed experts on the dispatched tokens.

        This method first post-processes the dispatched input to get permuted tokens
        for each expert. It then passes the tokens through the local experts.
        The output from the experts is preprocessed for the combine step.
        """
        debug_log(logger, logging.INFO,f"in moe layer line 228, hidden_states: {hidden_states.shape}")
        debug_log(logger, logging.INFO,f"in moe layer line 228, hidden_states has nan: {torch.isnan(hidden_states).any()}")
        debug_log(logger, logging.INFO,f"in moe layer line 228, hidden_states has inf: {torch.isinf(hidden_states).any()}")
        debug_log(logger, logging.INFO,f"in moe layer line 228, hidden_states has -inf: {torch.isneginf(hidden_states).any()}")
        if is_debug_enabled():

            if torch.isnan(hidden_states).any():

                raise ValueError(f"hidden_states contains NaN values at line 228. Shape: {hidden_states.shape}")
            if torch.isinf(hidden_states).any():
                raise ValueError(f"hidden_states contains inf values at line 228. Shape: {hidden_states.shape}")
            if torch.isneginf(hidden_states).any():
                raise ValueError(f"hidden_states contains -inf values at line 228. Shape: {hidden_states.shape}")
        debug_log(logger, logging.INFO,f"in moe layer line 228, probs: {probs.shape}")
        debug_log(logger, logging.INFO,f"in moe layer line 228, probs has nan: {torch.isnan(probs).any()}")
        debug_log(logger, logging.INFO,f"in moe layer line 228, probs has inf: {torch.isinf(probs).any()}")
        debug_log(logger, logging.INFO,f"in moe layer line 228, probs has -inf: {torch.isneginf(probs).any()}")
        if is_debug_enabled():

            if torch.isnan(probs).any():

                raise ValueError(f"probs contains NaN values at line 228. Shape: {probs.shape}")
            if torch.isinf(probs).any():
                raise ValueError(f"probs contains inf values at line 228. Shape: {probs.shape}")
            if torch.isneginf(probs).any():
                raise ValueError(f"probs contains -inf values at line 228. Shape: {probs.shape}")
        dispatched_input, tokens_per_expert, permuted_probs = (
            self.token_dispatcher.dispatch_postprocess(hidden_states, probs)
        )
        if dispatched_input is not None:
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 231, dispatched_input: {dispatched_input.shape}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 231, dispatched_input has nan: {torch.isnan(dispatched_input).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 231, dispatched_input has inf: {torch.isinf(dispatched_input).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 231, dispatched_input has -inf: {torch.isneginf(dispatched_input).any()}")
            if is_debug_enabled():

                if torch.isnan(dispatched_input).any():

                    raise ValueError(f"dispatched_input contains NaN values at line 231. Shape: {dispatched_input.shape}")
                if torch.isinf(dispatched_input).any():
                    raise ValueError(f"dispatched_input contains inf values at line 231. Shape: {dispatched_input.shape}")
                if torch.isneginf(dispatched_input).any():
                    raise ValueError(f"dispatched_input contains -inf values at line 231. Shape: {dispatched_input.shape}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 231, tokens_per_expert: {tokens_per_expert}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 231, permuted_probs: {permuted_probs}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 231, permuted_probs has nan: {torch.isnan(permuted_probs).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 231, permuted_probs has inf: {torch.isinf(permuted_probs).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 231, permuted_probs has -inf: {torch.isneginf(permuted_probs).any()}")
            if permuted_probs is not None:
                if is_debug_enabled():

                    if torch.isnan(permuted_probs).any():

                        raise ValueError(f"permuted_probs contains NaN values at line 231. Shape: {permuted_probs.shape}")
                    if torch.isinf(permuted_probs).any():
                        raise ValueError(f"permuted_probs contains inf values at line 231. Shape: {permuted_probs.shape}")
                    if torch.isneginf(permuted_probs).any():
                        raise ValueError(f"permuted_probs contains -inf values at line 231. Shape: {permuted_probs.shape}")
        else:
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 231, dispatched_input is None")
        expert_output, mlp_bias = self.experts(dispatched_input, tokens_per_expert, permuted_probs)
        if expert_output is not None:
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 244, expert_output: {expert_output.shape}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 244, expert_output has nan: {torch.isnan(expert_output).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 244, expert_output has inf: {torch.isinf(expert_output).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 244, expert_output has -inf: {torch.isneginf(expert_output).any()}")
            if is_debug_enabled():

                if torch.isnan(expert_output).any():

                    raise ValueError(f"expert_output contains NaN values at line 244. Shape: {expert_output.shape}")
                if torch.isinf(expert_output).any():
                    raise ValueError(f"expert_output contains inf values at line 244. Shape: {expert_output.shape}")
                if torch.isneginf(expert_output).any():
                    raise ValueError(f"expert_output contains -inf values at line 244. Shape: {expert_output.shape}")
        else:
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 244, expert_output is None")
        if mlp_bias is not None:
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 244, mlp_bias: {mlp_bias}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 244, mlp_bias has nan: {torch.isnan(mlp_bias).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 244, mlp_bias has inf: {torch.isinf(mlp_bias).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 244, mlp_bias has -inf: {torch.isneginf(mlp_bias).any()}")
            if is_debug_enabled():

                if torch.isnan(mlp_bias).any():

                    raise ValueError(f"mlp_bias contains NaN values at line 244. Shape: {mlp_bias.shape}")
                if torch.isinf(mlp_bias).any():
                    raise ValueError(f"mlp_bias contains inf values at line 244. Shape: {mlp_bias.shape}")
                if torch.isneginf(mlp_bias).any():
                    raise ValueError(f"mlp_bias contains -inf values at line 244. Shape: {mlp_bias.shape}")
        else:
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 244, mlp_bias is None")
        debug_assert(mlp_bias is None, f"mlp_bias is not supported for {type(self.token_dispatcher)}")
        output = self.token_dispatcher.combine_preprocess(expert_output)
        if output is not None:
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 259, output: {output.shape}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 259, output has nan: {torch.isnan(output).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 259, output has inf: {torch.isinf(output).any()}")
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 259, output has -inf: {torch.isneginf(output).any()}")
            if is_debug_enabled():

                if torch.isnan(output).any():

                    raise ValueError(f"output contains NaN values at line 259. Shape: {output.shape}")
                if torch.isinf(output).any():
                    raise ValueError(f"output contains inf values at line 259. Shape: {output.shape}")
                if torch.isneginf(output).any():
                    raise ValueError(f"output contains -inf values at line 259. Shape: {output.shape}")
        else:
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 259, output is None")
        return output, mlp_bias

    def combine(self, output: torch.Tensor, shared_expert_output: Optional[torch.Tensor]):
        """Combines expert outputs via communication and adds shared expert output.

        This method uses the token dispatcher to combine the outputs from different
        experts (e.g., via an All-to-All communication). It then adds the output
        from the shared expert if it exists.
        """
        output = self.token_dispatcher.token_combine(output)
        output = self.token_dispatcher.combine_postprocess(output)
        if shared_expert_output is not None:
            output = output + shared_expert_output
        return output

    def forward(self, hidden_states: torch.Tensor):
        """Forward pass for the MoE layer.

        The forward pass comprises four main steps:
        1. Routing & Preprocessing: Route tokens to the assigned experts and prepare for dispatch.
        2. Dispatch: Tokens are sent to the expert devices using communication collectives.
        3. Expert Computation: Experts process the dispatched tokens.
        4. Combine: The outputs from the experts are combined and returned.

        Args:
            hidden_states (torch.Tensor): The input tensor to the MoE layer.

        Returns:
            A tuple containing the output tensor and the MLP bias, if any.
        """
        if self.training and self.attn_tp_group.size() > 1 and not self.config.sequence_parallel:
            raise ValueError(
                "During training, performance may degrade if MoE and tensor parallelism"
                "are enabled without also enabling sequence parallelism."
            )

        # MoE forward: route -> dispatch -> compute -> combine
        def custom_forward(hidden_states):
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 273, custom_forward is called")
            if hidden_states is not None:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 274, hidden_states: {hidden_states.shape}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 274, hidden_states has nan: {torch.isnan(hidden_states).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 274, hidden_states has inf: {torch.isinf(hidden_states).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 274, hidden_states has -inf: {torch.isneginf(hidden_states).any()}")
                if is_debug_enabled():

                    if torch.isnan(hidden_states).any():

                        raise ValueError(f"hidden_states contains NaN values at line 274. Shape: {hidden_states.shape}")
                    if torch.isinf(hidden_states).any():
                        raise ValueError(f"hidden_states contains inf values at line 274. Shape: {hidden_states.shape}")
                    if torch.isneginf(hidden_states).any():
                        raise ValueError(f"hidden_states contains -inf values at line 274. Shape: {hidden_states.shape}")
            else:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 274, hidden_states is None")
            shared_expert_output = self.shared_experts_compute(hidden_states)
            if shared_expert_output is not None:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 278, shared_expert_output: {shared_expert_output.shape}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 278, shared_expert_output has nan: {torch.isnan(shared_expert_output).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 278, shared_expert_output has inf: {torch.isinf(shared_expert_output).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 278, shared_expert_output has -inf: {torch.isneginf(shared_expert_output).any()}")
                if is_debug_enabled():

                    if torch.isnan(shared_expert_output).any():

                        raise ValueError(f"shared_expert_output contains NaN values at line 278. Shape: {shared_expert_output.shape}")
                    if torch.isinf(shared_expert_output).any():
                        raise ValueError(f"shared_expert_output contains inf values at line 278. Shape: {shared_expert_output.shape}")
                    if torch.isneginf(shared_expert_output).any():
                        raise ValueError(f"shared_expert_output contains -inf values at line 278. Shape: {shared_expert_output.shape}")
            else:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 278, shared_expert_output is None")
            hidden_states, probs, residual = self.router_and_preprocess(hidden_states)
            if hidden_states is not None:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 282, hidden_states: {hidden_states.shape}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 282, hidden_states has nan: {torch.isnan(hidden_states).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 282, hidden_states has inf: {torch.isinf(hidden_states).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 282, hidden_states has -inf: {torch.isneginf(hidden_states).any()}")
                if is_debug_enabled():

                    if torch.isnan(hidden_states).any():

                        raise ValueError(f"hidden_states contains NaN values at line 282. Shape: {hidden_states.shape}")
                    if torch.isinf(hidden_states).any():
                        raise ValueError(f"hidden_states contains inf values at line 282. Shape: {hidden_states.shape}")
                    if torch.isneginf(hidden_states).any():
                        raise ValueError(f"hidden_states contains -inf values at line 282. Shape: {hidden_states.shape}")
            else:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 282, hidden_states is None")
            dispatched_input, probs = self.dispatch(hidden_states, probs)
            if dispatched_input is not None:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 286, dispatched_input: {dispatched_input.shape}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 286, dispatched_input has nan: {torch.isnan(dispatched_input).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 286, dispatched_input has inf: {torch.isinf(dispatched_input).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 286, dispatched_input has -inf: {torch.isneginf(dispatched_input).any()}")
                if is_debug_enabled():

                    if torch.isnan(dispatched_input).any():

                        raise ValueError(f"dispatched_input contains NaN values at line 286. Shape: {dispatched_input.shape}")
                    if torch.isinf(dispatched_input).any():
                        raise ValueError(f"dispatched_input contains inf values at line 286. Shape: {dispatched_input.shape}")
                    if torch.isneginf(dispatched_input).any():
                        raise ValueError(f"dispatched_input contains -inf values at line 286. Shape: {dispatched_input.shape}")
            else:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 286, dispatched_input is None")
            output, mlp_bias = self.routed_experts_compute(dispatched_input, probs, residual)
            if output is not None:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 290, output: {output.shape}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 290, output has nan: {torch.isnan(output).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 290, output has inf: {torch.isinf(output).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 290, output has -inf: {torch.isneginf(output).any()}")
                if is_debug_enabled():

                    if torch.isnan(output).any():

                        raise ValueError(f"output contains NaN values at line 290. Shape: {output.shape}")
                    if torch.isinf(output).any():
                        raise ValueError(f"output contains inf values at line 290. Shape: {output.shape}")
                    if torch.isneginf(output).any():
                        raise ValueError(f"output contains -inf values at line 290. Shape: {output.shape}")
            else:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 290, output is None")
            if mlp_bias is not None:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 294, mlp_bias: {mlp_bias.shape}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 294, mlp_bias has nan: {torch.isnan(mlp_bias).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 294, mlp_bias has inf: {torch.isinf(mlp_bias).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 294, mlp_bias has -inf: {torch.isneginf(mlp_bias).any()}")
                if is_debug_enabled():

                    if torch.isnan(mlp_bias).any():

                        raise ValueError(f"mlp_bias contains NaN values at line 294. Shape: {mlp_bias.shape}")
                    if torch.isinf(mlp_bias).any():
                        raise ValueError(f"mlp_bias contains inf values at line 294. Shape: {mlp_bias.shape}")
                    if torch.isneginf(mlp_bias).any():
                        raise ValueError(f"mlp_bias contains -inf values at line 294. Shape: {mlp_bias.shape}")
            else:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 294, mlp_bias is None")
            output = self.combine(output, shared_expert_output)
            if output is not None:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 298, output: {output.shape}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 298, output has nan: {torch.isnan(output).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 298, output has inf: {torch.isinf(output).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 298, output has -inf: {torch.isneginf(output).any()}")
                if is_debug_enabled():

                    if torch.isnan(output).any():

                        raise ValueError(f"output contains NaN values at line 298. Shape: {output.shape}")
                if is_debug_enabled():
                    if torch.isinf(output).any():
                        raise ValueError(f"output contains inf values at line 298. Shape: {output.shape}")
                if is_debug_enabled():

                    if torch.isneginf(output).any():

                        raise ValueError(f"output contains -inf values at line 298. Shape: {output.shape}")
            else:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 298, output is None")
            if mlp_bias is not None:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 302, mlp_bias: {mlp_bias.shape}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 302, mlp_bias has nan: {torch.isnan(mlp_bias).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 302, mlp_bias has inf: {torch.isinf(mlp_bias).any()}")
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 302, mlp_bias has -inf: {torch.isneginf(mlp_bias).any()}")
                if is_debug_enabled():

                    if torch.isnan(mlp_bias).any():

                        raise ValueError(f"mlp_bias contains NaN values at line 302. Shape: {mlp_bias.shape}")
                if is_debug_enabled():
                    if torch.isinf(mlp_bias).any():
                        raise ValueError(f"mlp_bias contains inf values at line 302. Shape: {mlp_bias.shape}")
                if is_debug_enabled():

                    if torch.isneginf(mlp_bias).any():

                        raise ValueError(f"mlp_bias contains -inf values at line 302. Shape: {mlp_bias.shape}")
            else:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 302, mlp_bias is None")
            return output, mlp_bias

        if self.moe_layer_recompute:
            if is_debug_enabled():
                debug_log(logger, logging.INFO,f"in moe layer line 281, moe_layer_recompute is True")
            if self.config.fp8:
                output, mlp_bias = te_checkpoint(
                    custom_forward,
                    False,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    hidden_states,
                )
            else:
                if is_debug_enabled():
                    debug_log(logger, logging.INFO,f"in moe layer line 291, moe_layer_recompute is True and fp8 is False")
                output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
        else:
            output, mlp_bias = custom_forward(hidden_states)

        return output, mlp_bias

    def backward_dw(self):
        """Compute weight gradients for experts and shared experts."""
        self.experts.backward_dw()
        if self.use_shared_expert and not self.shared_expert_overlap:
            self.shared_experts.backward_dw()

    def set_for_recompute_pre_mlp_layernorm(self):
        """Set the MoE layer for recompute pre_mlp_layernorm. Only needed for fp8."""
        # If shared_experts_recompute is used, nothing needs to be done because the checkpoint
        # function will save the original input tensors.
        if self.shared_experts is not None and not self.shared_experts_recompute:
            from megatron.core.extensions.transformer_engine import set_save_original_input

            set_save_original_input(self.shared_experts.linear_fc1)
