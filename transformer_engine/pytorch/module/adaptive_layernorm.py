# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

"""Adaptive LayerNorm API"""
import os
import warnings
from typing import Union, Tuple, Optional

import torch
from torch.nn.parameter import Parameter
from torch.nn import init

import transformer_engine_torch as tex
from .base import TransformerEngineBaseModule
from ..cpp_extensions import (
    layernorm_fwd_inf,
)
from ..jit import no_torch_dynamo
from ..utils import cast_if_needed

__all__ = ["LayerNorm"]


class _AdaptiveLayerNorm(torch.autograd.Function):
    """functional LayerNorm"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        ln_weight: torch.Tensor,
        ln_bias: torch.Tensor,
        eps: float,
        fwd_ln_sm_margin: int,
        bwd_ln_sm_margin: int,
        inf_ln_sm_margin: int,
        zero_centered_gamma: bool,
        is_grad_enabled: bool,
        activation_dtype: torch.dtype,
        layout: str,
    ) -> torch.Tensor:
        # Make sure input dimensions are compatible
        in_features = ln_weight.shape[-1]
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert inp.shape[-1] == in_features, "AdaptiveLayerNorm not possible"
        assert ln_weight.shape == ln_bias.shape, "Inconsistent shape"
        assert len(ln_weight.shape) == 2, "Expected 2 dimensions for weight."
        inputmat = inp.view((-1, in_features))

        b = ln_weight.shape[0]
        if layout == "sbh":
            ln_weight = ln_weight.view(1, b, in_features)
            ln_bias = ln_bias.view(1, b, in_features)
        else:
            ln_weight = ln_weight.view(b, 1, in_features)
            ln_bias = ln_bias.view(b, 1, in_features)

        # Cast for native AMP
        inputmat = cast_if_needed(inputmat, activation_dtype)
        ln_weight = cast_if_needed(ln_weight, activation_dtype)
        ln_bias = cast_if_needed(ln_bias, activation_dtype)

        if is_grad_enabled:
            ln_out, mu, rsigma = tex.layernorm_fwd(
                inputmat, ln_weight, ln_bias, eps, fwd_ln_sm_margin, zero_centered_gamma
            )
            ctx.save_for_backward(inputmat, ln_weight, mu, rsigma)
            ctx.inp_shape = inp.shape
            ctx.bwd_ln_sm_margin = bwd_ln_sm_margin
            ctx.zero_centered_gamma = zero_centered_gamma
            ctx.b = b
            ctx.in_features = in_features
        else:
            ln_out, mu, rsigma = (
                layernorm_fwd_inf(
                    inputmat, ln_weight, ln_bias, eps, inf_ln_sm_margin, zero_centered_gamma
                ),
                None,
                None,
            )
        return ln_out.view_as(inp)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        inputmat, ln_weight, mu, rsigma = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        d_ln_out = grad_output.view(inputmat.shape)
        dxmat, dgamma, dbeta = tex.layernorm_bwd(
            d_ln_out, inputmat, mu, rsigma, ln_weight, ctx.bwd_ln_sm_margin, ctx.zero_centered_gamma
        )
        dgamma = dgamma.view(ctx.b, ctx.in_features)
        dbeta = dbeta.view(ctx.b, ctx.in_features)

        return dxmat.view(ctx.inp_shape), dgamma, dbeta, \
               None, None, None, None, None, None, None, None


class AdaptiveLayerNorm(torch.nn.Module):
    r"""
    Applies Layer Normalization over a mini-batch of inputs as described in
    the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} * \gamma + \beta

    :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
    size :attr:`hidden_size`

    Parameters
    ----------
    hidden_size : int
                size of each input sample.
    eps : float, default = 1e-5
        a value added to the denominator of layer normalization for numerical stability.
    zero_centered_gamma : bool, default = 'False'
                         if set to 'True', gamma parameter in LayerNorm is initialized to 0 and
                         the LayerNorm formula changes to

                         .. math::
                            y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \varepsilon}} *
                            (1 + \gamma) + \beta
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        zero_centered_gamma: bool = False,
        layout: str = "sbh",
    ) -> None:
        super().__init__()
        self.eps = eps
        self.zero_centered_gamma = zero_centered_gamma
        self.layout = layout
        assert self.layout in ["sbh", "bsh"], "Unsupported layout!"

        # These many SMs are subtracted from the total SM count when calling forward
        # and backward LayerNorm C APIs. These envvars can be used to prevent the LN
        # kernels from using all SMs in the device. This is useful for cases such as
        # communication overlap with LN.
        self.fwd_ln_sm_margin = int(os.getenv("NVTE_FWD_LAYERNORM_SM_MARGIN", "0"))
        self.bwd_ln_sm_margin = int(os.getenv("NVTE_BWD_LAYERNORM_SM_MARGIN", "0"))
        self.inf_ln_sm_margin = int(os.getenv("NVTE_INF_LAYERNORM_SM_MARGIN", "0"))

    @no_torch_dynamo()
    def forward(self, inp: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        """Adaptive LayerNorm FWD"""
        # Set the activation type for AMP.
        TransformerEngineBaseModule.set_activation_dtype(self, inp)
        weight = weight.contiguous()
        bias = bias.contiguous()

        if torch.is_grad_enabled():
            fwd_fn = _AdaptiveLayerNorm.apply
            args = []
        else:
            fwd_fn = _AdaptiveLayerNorm.forward
            args = [None]

        args += (
            inp,
            weight,
            bias,
            self.eps,
            self.fwd_ln_sm_margin,
            self.bwd_ln_sm_margin,
            self.inf_ln_sm_margin,
            self.zero_centered_gamma,
            torch.is_grad_enabled(),
            self.activation_dtype,
            self.layout,
        )

        return fwd_fn(*args)
