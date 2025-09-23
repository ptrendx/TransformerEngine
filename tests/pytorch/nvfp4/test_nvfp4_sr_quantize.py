# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

import pytest
import torch
from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

recipe_available, reason_for_no_recipe = FP8GlobalStateManager.is_nvfp4_available()

seed = 12345
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def unpack_fp4(x: torch.Tensor) -> torch.Tensor:
    repeated = x.repeat_interleave(2, dim=1)
    repeated[:, 0::2] &= 0x0F
    repeated[:, 1::2] >>= 4
    return repeated


_FP4_LUT = torch.tensor(
    [
        0.0,  # 0: 0000 - zero
        0.5,  # 1: 0001 - smallest positive normal
        1.0,  # 2: 0010
        1.5,  # 3: 0011
        2.0,  # 4: 0100
        3.0,  # 5: 0101
        4.0,  # 6: 0110
        6.0,  # 7: 0111 - largest positive normal
        -0.0,  # 8: 1000 - negative zero
        -0.5,  # 9: 1001 - smallest negative normal
        -1.0,  # 10: 1010
        -1.5,  # 11: 1011
        -2.0,  # 12: 1100
        -3.0,  # 13: 1101
        -4.0,  # 14: 1110
        -6.0,  # 15: 1111 - largest negative normal
    ],
    dtype=torch.float32,
)


def fp4_to_fp32(fp4: torch.Tensor) -> torch.Tensor:
    # Convert FP4 indices to their corresponding floating point values
    # Each index (0-15) represents a 4-bit FP4 value in E2M1 format
    # Values based on the FP4 E2M1 specification
    fp4_lut = _FP4_LUT.to(fp4.device)
    return fp4_lut[fp4.to(torch.long)]


def dequantize_fp4(qx: torch.Tensor, sx: torch.Tensor, amax: torch.Tensor) -> torch.Tensor:
    sf = sx.repeat_interleave(16, dim=1).view(torch.float8_e4m3fn).to(torch.float32)
    dequant = fp4_to_fp32(unpack_fp4(qx)) * sf * (amax / (6.0 * 448))
    return dequant


def quantize_fp4(x: torch.Tensor, use_stochastic_rounding: bool) -> torch.Tensor:
    nvfp4_quantizer = NVFP4Quantizer(
        rowwise=True,
        columnwise=False,
        with_amax_reduction=False,
        amax_reduction_group=None,
        with_rht=False,
        with_post_rht_amax=False,
        stochastic_rounding=use_stochastic_rounding,
    )

    x_nvfp4_sut = nvfp4_quantizer(x)
    # Extract data from NVFP4Tensor
    assert x_nvfp4_sut._rowwise_data is not None
    qx: torch.Tensor = x_nvfp4_sut._rowwise_data.view(dtype=torch.uint8)
    assert x_nvfp4_sut._rowwise_scale_inv is not None
    sx: torch.Tensor = x_nvfp4_sut._rowwise_scale_inv

    return qx, sx


def check_quantization_nvfp4_versus_reference(x_dtype: torch.dtype, M: int, N: int) -> None:
    device = "cuda"
    torch.manual_seed(seed)
    n_iters = 50

    # List of per-step signed mean error
    # Absolute error for each iteration
    # is expected to be worse for SR. But
    # when signed errors are summed over
    # a larger interval, SR should better
    # represent to true mean.
    mean_err_sr, mean_err_rn = [], []

    x = torch.randn((M, N), dtype=x_dtype, device=device) * 2 - 1
    amax = torch.max(torch.abs(x)).float()
    q_rn, s_rn = quantize_fp4(x, use_stochastic_rounding=False)
    dq_rn = dequantize_fp4(q_rn, s_rn, amax)
    error_rn = (dq_rn - x).float()
    me_rn = torch.sqrt((error_rn * error_rn).mean())
    sr_result = torch.zeros_like(x).float()
    for _ in range(n_iters):
        q_sr, s_sr = quantize_fp4(x, use_stochastic_rounding=True)

        dq_sr = dequantize_fp4(q_sr, s_sr, amax)

        sr_result += dq_sr.float()

    # Get the mean result of the stochastic rounding
    # It should be more accurate than the RN result
    sr_result /= n_iters
    error_sr = (sr_result - x).float()
    me_sr = torch.sqrt((error_sr * error_sr).mean())

    print(f"RMSE SR: {me_sr:.3e} | RMSE RN: {me_rn:.3e}")
    assert me_sr < me_rn, "Stochastic rounding failed - error is larger than the rount to nearest."


@pytest.mark.skipif(not recipe_available, reason=reason_for_no_recipe)
@pytest.mark.parametrize(
    "M, N",
    [
        (8192, 8192),
    ],
)
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16], ids=str)
def test_quantization_block_tiling_versus_reference(
    x_dtype: torch.dtype,
    M: int,
    N: int,
) -> None:
    check_quantization_nvfp4_versus_reference(
        x_dtype=x_dtype,
        M=M,
        N=N,
    )
