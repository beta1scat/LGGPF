# Copyright (c) 2023, Tri Dao.
# Implement residual + layer_norm / rms_norm.

import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False


def layer_norm_ref(x, weight, bias, residual=None, eps=1e-6, prenorm=False, upcast=False):
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    out = F.layer_norm(x.to(weight.dtype), x.shape[-1:], weight=weight, bias=bias, eps=eps).to(dtype)
    return out if not prenorm else (out, x)


def rms_norm_ref(x, weight, bias, residual=None, eps=1e-6, prenorm=False, upcast=False):
    dtype = x.dtype
    if upcast:
        weight = weight.float()
        bias = bias.float() if bias is not None else None
    if upcast:
        x = x.float()
        residual = residual.float() if residual is not None else residual
    if residual is not None:
        x = (x + residual).to(x.dtype)
    rstd = 1 / torch.sqrt((x.square()).mean(dim=-1, keepdim=True) + eps)
    out = (x * rstd * weight) + bias if bias is not None else (x * rstd * weight)
    out = out.to(dtype)
    return out if not prenorm else (out, x)


if _HAS_TRITON:
    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=1),
            triton.Config({}, num_warps=2),
            triton.Config({}, num_warps=4),
            triton.Config({}, num_warps=8),
            triton.Config({}, num_warps=16),
            triton.Config({}, num_warps=32),
        ],
        key=["N", "HAS_RESIDUAL", "STORE_RESIDUAL_OUT", "IS_RMS_NORM", "HAS_BIAS"],
    )
    @triton.jit
    def _layer_norm_fwd_1pass_kernel(
        X, Y, W, B, RESIDUAL, RESIDUAL_OUT, Mean, Rstd,
        stride_x_row, stride_y_row, stride_res_row, stride_res_out_row,
        N, eps,
        IS_RMS_NORM: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        STORE_RESIDUAL_OUT: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        row = tl.program_id(0)
        X += row * stride_x_row
        Y += row * stride_y_row
        if HAS_RESIDUAL:
            RESIDUAL += row * stride_res_row
        if STORE_RESIDUAL_OUT:
            RESIDUAL_OUT += row * stride_res_out_row
        cols = tl.arange(0, BLOCK_N)
        x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            residual = tl.load(RESIDUAL + cols, mask=cols < N, other=0.0).to(tl.float32)
            x += residual
        if STORE_RESIDUAL_OUT:
            tl.store(RESIDUAL_OUT + cols, x, mask=cols < N)
        if not IS_RMS_NORM:
            mean = tl.sum(x, axis=0) / N
            tl.store(Mean + row, mean)
            xbar = tl.where(cols < N, x - mean, 0.0)
            var = tl.sum(xbar * xbar, axis=0) / N
        else:
            xbar = tl.where(cols < N, x, 0.0)
            var = tl.sum(xbar * xbar, axis=0) / N
        rstd = 1 / tl.sqrt(var + eps)
        tl.store(Rstd + row, rstd)
        mask = cols < N
        w = tl.load(W + cols, mask=mask).to(tl.float32)
        if HAS_BIAS:
            b = tl.load(B + cols, mask=mask).to(tl.float32)
        x_hat = (x - mean) * rstd if not IS_RMS_NORM else x * rstd
        y = x_hat * w + b if HAS_BIAS else x_hat * w
        tl.store(Y + cols, y, mask=mask)


    def _layer_norm_fwd(
        x, weight, bias, eps, residual=None, out_dtype=None, residual_dtype=None, is_rms_norm=False
    ):
        if residual is not None:
            residual_dtype = residual.dtype
        M, N = x.shape
        assert x.stride(-1) == 1
        if residual is not None:
            assert residual.stride(-1) == 1
            assert residual.shape == (M, N)
        assert weight.shape == (N,)
        assert weight.stride(-1) == 1
        if bias is not None:
            assert bias.stride(-1) == 1
            assert bias.shape == (N,)
        y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
        assert y.stride(-1) == 1
        if residual is not None or (residual_dtype is not None and residual_dtype != x.dtype):
            residual_out = torch.empty(M, N, device=x.device, dtype=residual_dtype)
            assert residual_out.stride(-1) == 1
        else:
            residual_out = None
        mean = torch.empty((M,), dtype=torch.float32, device="cuda") if not is_rms_norm else None
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_N:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        with torch.cuda.device(x.device.index):
            _layer_norm_fwd_1pass_kernel[(M,)](
                x, y, weight, bias, residual, residual_out, mean, rstd,
                x.stride(0), y.stride(0),
                residual.stride(0) if residual is not None else 0,
                residual_out.stride(0) if residual_out is not None else 0,
                N, eps, is_rms_norm, BLOCK_N,
                residual is not None, residual_out is not None, bias is not None,
            )
        return y, mean, rstd, residual_out if residual_out is not None else x


    class LayerNormFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias, residual=None, eps=1e-6, prenorm=False, residual_in_fp32=False, is_rms_norm=False):
            x_shape_og = x.shape
            x = x.reshape(-1, x.shape[-1])
            if x.stride(-1) != 1:
                x = x.contiguous()
            if residual is not None:
                assert residual.shape == x_shape_og
                residual = residual.reshape(-1, residual.shape[-1])
                if residual.stride(-1) != 1:
                    residual = residual.contiguous()
            weight = weight.contiguous()
            if bias is not None:
                bias = bias.contiguous()
            residual_dtype = residual.dtype if residual is not None else (torch.float32 if residual_in_fp32 else None)
            y, mean, rstd, residual_out = _layer_norm_fwd(x, weight, bias, eps, residual, residual_dtype=residual_dtype, is_rms_norm=is_rms_norm)
            ctx.save_for_backward(residual_out, weight, bias, mean, rstd)
            ctx.x_shape_og = x_shape_og
            ctx.eps = eps
            ctx.is_rms_norm = is_rms_norm
            ctx.has_residual = residual is not None
            ctx.prenorm = prenorm
            ctx.x_dtype = x.dtype
            y = y.reshape(x_shape_og)
            return y if not prenorm else (y, residual_out.reshape(x_shape_og))

        @staticmethod
        def backward(ctx, dy, *args):
            raise NotImplementedError("Backward pass not supported in inference-only build")


    def layer_norm_fn(x, weight, bias, residual=None, eps=1e-6, prenorm=False, residual_in_fp32=False, is_rms_norm=False):
        return LayerNormFn.apply(x, weight, bias, residual, eps, prenorm, residual_in_fp32, is_rms_norm)

    def rms_norm_fn(x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-6):
        return LayerNormFn.apply(x, weight, bias, residual, eps, prenorm, residual_in_fp32, True)

else:
    def layer_norm_fn(x, weight, bias, residual=None, eps=1e-6, prenorm=False, residual_in_fp32=False, is_rms_norm=False):
        return layer_norm_ref(x, weight, bias, residual=residual, eps=eps, prenorm=prenorm)

    def rms_norm_fn(x, weight, bias, residual=None, prenorm=False, residual_in_fp32=False, eps=1e-6):
        return rms_norm_ref(x, weight, bias, residual=residual, eps=eps, prenorm=prenorm)


class RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, residual=None, prenorm=False, residual_in_fp32=False):
        return rms_norm_fn(
            x, self.weight, self.bias, residual=residual, eps=self.eps,
            prenorm=prenorm, residual_in_fp32=residual_in_fp32
        )
