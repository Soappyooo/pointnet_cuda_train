import os
import h5py
import time
import numpy as np
import torch
import triton
import triton.language as tl
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from functools import partial

sample_points = 64
mode = "train"
train_batch_size = 128
test_batch_size = 1024
lr = 1e-3
num_epochs = 100
target_accuracy = 0.75


class TritonKernels:
    # `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
    #   - A list of `triton.Config` objects that define different configurations of
    #       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
    #   - An auto-tuning *key* whose change in values will trigger evaluation of all the
    #       provided configs
    # @triton.autotune(
    #     configs=[
    #         triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
    #         triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    #         triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    #         triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    #         triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    #         triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    #         triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
    #         triton.Config({"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8}, num_stages=5, num_warps=2),
    #         # Good config for fp8 inputs.
    #         triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
    #         triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=3, num_warps=8),
    #         triton.Config({"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    #         triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    #         triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    #         triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    #         triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    #         triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8}, num_stages=4, num_warps=4),
    #     ],
    #     key=["M", "N", "K"],
    # )
    @triton.jit
    def matmul_kernel(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        # Matrix dimensions
        M,
        N,
        K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am,
        stride_ak,  #
        stride_bk,
        stride_bn,  #
        stride_cm,
        stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
    ):
        """Kernel for computing the matmul C = A x B.
        A has shape (M, K), B has shape (K, N) and C has shape (M, N)
        """
        # -----------------------------------------------------------
        # Map program ids `pid` to the block of C it should compute.
        # This is done in a grouped ordering to promote L2 data reuse.
        # See above `L2 Cache Optimizations` section for details.
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        # ----------------------------------------------------------
        # Create pointers for the first blocks of A and B.
        # We will advance this pointer as we move in the K direction
        # and accumulate
        # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
        # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
        # See above `Pointer Arithmetic` section for details
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        # -----------------------------------------------------------
        # Iterate to compute a block of the C matrix.
        # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
        c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            # Load the next block of A and B, generate a mask by checking the K dimension.
            # If it is out of bounds, set it to 0.
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            # We accumulate along the K dimension.
            c = tl.dot(a, b, c, input_precision="ieee")
            # Advance the ptrs to the next K block.
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # -----------------------------------------------------------
        # Write back the block of the output matrix C with masks.
        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    @triton.jit
    def bmm_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_ab,
        stride_am,
        stride_ak,
        stride_bb,
        stride_bk,
        stride_bn,
        stride_cb,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        # simple batch process
        batch_id = tl.program_id(axis=1)
        a_ptr += batch_id * stride_ab
        b_ptr += batch_id * stride_bb
        c_ptr += batch_id * stride_cb
        # the rest is the same as matmul
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

        c = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            c = tl.dot(a, b, c, input_precision="ieee")
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    @triton.jit
    def linear_kernel(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        M,
        N,
        K,
        stride_im,
        stride_ik,
        stride_wk,
        stride_wn,
        stride_om,
        stride_on,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_im = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_wn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        input_ptrs = input_ptr + (offs_im[:, None] * stride_im + offs_k[None, :] * stride_ik)
        weight_ptrs = weight_ptr + (offs_k[:, None] * stride_wk + offs_wn[None, :] * stride_wn)

        output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            input = tl.load(input_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            weight = tl.load(weight_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            output = tl.dot(input, weight, output, input_precision="ieee")
            input_ptrs += BLOCK_SIZE_K * stride_ik
            weight_ptrs += BLOCK_SIZE_K * stride_wk

        offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        output_ptrs = output_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
        o_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)

        bias_ptrs = bias_ptr + stride_on * offs_on[None, :]
        bias = tl.load(bias_ptrs, mask=offs_on[None, :] < N)
        tl.store(output_ptrs, output + bias, mask=o_mask)

    @triton.jit
    def conv1dk1_kernel(
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        M,
        N,
        K,
        stride_ib,
        stride_ik,
        stride_in,
        stride_wm,
        stride_wk,
        stride_ob,
        stride_om,
        stride_on,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        # input shape: B*K*N; weight shape: M*K; bias shape:M; output[i]=weight@input[i]+bias
        pid = tl.program_id(axis=0)
        batch_id = tl.program_id(axis=1)
        input_ptr += batch_id * stride_ib
        output_ptr += batch_id * stride_ob
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_wm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_in = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        weight_ptrs = weight_ptr + (offs_wm[:, None] * stride_wm + offs_k[None, :] * stride_wk)
        input_ptrs = input_ptr + (offs_k[:, None] * stride_ik + offs_in[None, :] * stride_in)

        output = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            input = tl.load(input_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            weight = tl.load(weight_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            output = tl.dot(weight, input, output, input_precision="ieee")
            input_ptrs += BLOCK_SIZE_K * stride_ik
            weight_ptrs += BLOCK_SIZE_K * stride_wk

        offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        output_ptrs = output_ptr + stride_om * offs_om[:, None] + stride_on * offs_on[None, :]
        o_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)

        bias_ptrs = bias_ptr + offs_om[:, None]
        bias = tl.load(bias_ptrs, mask=offs_om[:, None] < M)
        tl.store(output_ptrs, output + bias, mask=o_mask)


class TritonOps:
    def matmul(a, b):
        # Check constraints.
        assert a.shape[1] == b.shape[0], "Incompatible dimensions"
        # assert a.is_contiguous(), "Matrix A must be contiguous"
        # if not a.is_contiguous():
        a = a.contiguous()
        b = b.contiguous()
        M, K = a.shape
        K, N = b.shape
        # Allocates output.
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
        TritonKernels.matmul_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            c.stride(0),
            c.stride(1),
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8,
        )
        return c

    def bmm(a, b):
        assert a.shape[2] == b.shape[1], "Incompatible dimensions"
        # assert a.is_contiguous(), "Input must be contiguous"
        # if not a.is_contiguous():
        a = a.contiguous()
        b = b.contiguous()
        B, M, K = a.shape
        B, K, N = b.shape
        # Allocates output.
        c = torch.empty((B, M, N), device=a.device, dtype=torch.float32)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            B,
        )
        TritonKernels.bmm_kernel[grid](
            a,
            b,
            c,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            a.stride(2),
            b.stride(0),
            b.stride(1),
            b.stride(2),
            c.stride(0),
            c.stride(1),
            c.stride(2),
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8,
        )
        return c

    def linear(input, weight, bias):
        assert input.shape[-1] == weight.shape[1], "Incompatible dimensions"
        # assert input.is_contiguous(), "Input must be contiguous"
        # if not input.is_contiguous():
        input = input.contiguous()
        input_shape = input.shape
        input = input.view(-1, input.size(-1))
        weight = weight.t()
        M, K = input.shape
        K, N = weight.shape
        # Allocates output.
        output = torch.empty((M, N), device=input.device, dtype=torch.float32)
        # 1D launch kernel where each block gets its own program.
        grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),)
        TritonKernels.linear_kernel[grid](
            input,
            weight,
            bias,
            output,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8,
        )
        output = output.view(*input_shape[:-1], -1)
        return output

    def conv1d_k1(input, weight, bias):
        assert input.shape[-2] == weight.shape[1], "Incompatible dimensions"
        # assert input.is_contiguous(), "Input must be contiguous"
        # if not input.is_contiguous():
        input = input.contiguous()
        input_shape = input.shape
        weight = weight.squeeze(-1)
        input = input.view(-1, input.size(-2), input.size(-1))
        B, K, N = input.shape
        M, K = weight.shape
        # Allocates output.
        output = torch.empty((B, M, N), device=input.device, dtype=torch.float32)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
            B,
        )
        TritonKernels.conv1dk1_kernel[grid](
            input,
            weight,
            bias,
            output,
            M,
            N,
            K,
            input.stride(0),
            input.stride(1),
            input.stride(2),
            weight.stride(0),
            weight.stride(1),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            BLOCK_SIZE_M=64,
            BLOCK_SIZE_N=64,
            BLOCK_SIZE_K=32,
            GROUP_SIZE_M=8,
        )
        output = output.view(*input_shape[:-2], -1, input.size(-1))
        return output


class TritonFunctions:
    class LinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            ctx.save_for_backward(input, weight, bias)
            return TritonOps.linear(input, weight, bias)

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias = ctx.saved_tensors
            input_shape = input.shape
            input = input.view(-1, input.size(-1))
            grad_output = grad_output.view(-1, grad_output.size(-1))
            grad_input = TritonOps.matmul(grad_output, weight)
            grad_weight = TritonOps.matmul(grad_output.t(), input)
            grad_bias = grad_output.sum(0)
            grad_input = grad_input.view(*input_shape)
            return grad_input, grad_weight, grad_bias

    class Conv1dk1Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias):
            ctx.save_for_backward(input, weight, bias)
            return TritonOps.conv1d_k1(input, weight, bias)

        @staticmethod
        def backward(ctx, grad_output):
            input, weight, bias = ctx.saved_tensors
            input_shape = input.shape
            input = input.view(-1, input.size(-2), input.size(-1))
            grad_output = grad_output.view(-1, grad_output.size(-2), grad_output.size(-1))
            grad_weight = TritonOps.bmm(grad_output, input.transpose(1, 2)).sum(0).unsqueeze(-1)
            grad_input = TritonOps.conv1d_k1(grad_output, weight.transpose(0, 1), torch.zeros(weight.size(0), device=grad_output.device))
            grad_input = grad_input.view(*input_shape)
            grad_bias = grad_output.sum(0).sum(-1)
            return grad_input, grad_weight, grad_bias


class TritonLayers:
    class Linear(nn.Module):
        def __init__(self, in_features, out_features):
            super(TritonLayers.Linear, self).__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.float32))
            self.reset_parameters()

        def reset_parameters(self):
            stdv = 1.0 / np.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)

        def forward(self, input):
            return TritonFunctions.LinearFunction.apply(input, self.weight, self.bias)

    class Conv1dk1(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(TritonLayers.Conv1dk1, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.weight = nn.Parameter(torch.empty(out_channels, in_channels, 1, dtype=torch.float32))
            self.bias = nn.Parameter(torch.empty(out_channels, dtype=torch.float32))
            self.reset_parameters()

        def reset_parameters(self):
            stdv = 1.0 / np.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)

        def forward(self, input):
            return TritonFunctions.Conv1dk1Function.apply(input, self.weight, self.bias)


class PointNet:
    class STN3d(nn.Module):
        def __init__(self, channel):
            super(PointNet.STN3d, self).__init__()
            # self.conv1 = torch.nn.Conv1d(channel, 64, 1)
            # self.conv2 = torch.nn.Conv1d(64, 128, 1)
            # self.conv3 = torch.nn.Conv1d(128, 1024, 1)
            self.conv1 = TritonLayers.Conv1dk1(channel, 64)
            self.conv2 = TritonLayers.Conv1dk1(64, 128)
            self.conv3 = TritonLayers.Conv1dk1(128, 1024)
            # self.fc1 = nn.Linear(1024, 512)
            # self.fc2 = nn.Linear(512, 256)
            # self.fc3 = nn.Linear(256, 9)
            self.fc1 = TritonLayers.Linear(1024, 512)
            self.fc2 = TritonLayers.Linear(512, 256)
            self.fc3 = TritonLayers.Linear(256, 9)
            self.relu = nn.ReLU()

            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

        def forward(self, x):
            batchsize = x.size()[0]
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)

            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
            x = self.fc3(x)

            iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden.type_as(x)
            x = x.view(-1, 3, 3)
            return x

    class STNkd(nn.Module):
        def __init__(self, k=64):
            super(PointNet.STNkd, self).__init__()
            # self.conv1 = torch.nn.Conv1d(k, 64, 1)
            # self.conv2 = torch.nn.Conv1d(64, 128, 1)
            # self.conv3 = torch.nn.Conv1d(128, 1024, 1)
            self.conv1 = TritonLayers.Conv1dk1(k, 64)
            self.conv2 = TritonLayers.Conv1dk1(64, 128)
            self.conv3 = TritonLayers.Conv1dk1(128, 1024)
            # self.fc1 = nn.Linear(1024, 512)
            # self.fc2 = nn.Linear(512, 256)
            # self.fc3 = nn.Linear(256, k * k)
            self.fc1 = TritonLayers.Linear(1024, 512)
            self.fc2 = TritonLayers.Linear(512, 256)
            self.fc3 = TritonLayers.Linear(256, k * k)
            self.relu = nn.ReLU()

            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

            self.k = k

        def forward(self, x):
            batchsize = x.size()[0]
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)

            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
            x = self.fc3(x)

            iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(batchsize, 1)
            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden.type_as(x)
            x = x.view(-1, self.k, self.k)
            return x

    class PointNetEncoder(nn.Module):
        def __init__(self, global_feat=True, feature_transform=False, channel=3):
            super(PointNet.PointNetEncoder, self).__init__()
            self.stn = PointNet.STN3d(channel)
            # self.conv1 = torch.nn.Conv1d(channel, 64, 1)
            # self.conv2 = torch.nn.Conv1d(64, 128, 1)
            # self.conv3 = torch.nn.Conv1d(128, 1024, 1)
            self.conv1 = TritonLayers.Conv1dk1(channel, 64)
            self.conv2 = TritonLayers.Conv1dk1(64, 128)
            self.conv3 = TritonLayers.Conv1dk1(128, 1024)
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.global_feat = global_feat
            self.feature_transform = feature_transform
            if self.feature_transform:
                self.fstn = PointNet.STNkd(k=64)

        def forward(self, x):
            B, D, N = x.size()
            trans = self.stn(x)
            x = x.transpose(2, 1)
            if D > 3:
                feature = x[:, :, 3:]
                x = x[:, :, :3]
            x = torch.bmm(x, trans)
            if D > 3:
                x = torch.cat([x, feature], dim=2)
            x = x.transpose(2, 1)
            x = F.relu(self.bn1(self.conv1(x)))

            if self.feature_transform:
                trans_feat = self.fstn(x)
                x = x.transpose(2, 1)
                x = torch.bmm(x, trans_feat)
                x = x.transpose(2, 1)
            else:
                trans_feat = None

            pointfeat = x
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
            if self.global_feat:
                return x, trans, trans_feat
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, N)
                return torch.cat([x, pointfeat], 1), trans, trans_feat

    # 模型定义
    class PointNetClassifier(nn.Module):
        def __init__(self, k=10, normal_channel=False):
            super(PointNet.PointNetClassifier, self).__init__()
            if normal_channel:
                channel = 6
            else:
                channel = 3
            self.feat = PointNet.PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
            # self.fc1 = nn.Linear(1024, 512)
            # self.fc2 = nn.Linear(512, 256)
            # self.fc3 = nn.Linear(256, k)
            self.fc1 = TritonLayers.Linear(1024, 512)
            self.fc2 = TritonLayers.Linear(512, 256)
            self.fc3 = TritonLayers.Linear(256, k)
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)
            self.relu = nn.ReLU()

        def forward(self, x):
            x, trans, trans_feat = self.feat(x)
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
            x = self.fc3(x)
            return x, trans_feat

    class PointNetClassifierLoss(nn.Module):
        def __init__(self, mat_diff_loss_scale=0.001):
            super(PointNet.PointNetClassifierLoss, self).__init__()
            self.mat_diff_loss_scale = mat_diff_loss_scale

        def feature_transform_reguliarzer(self, trans):
            d = trans.size()[1]
            I = torch.eye(d)[None, :, :]
            if trans.is_cuda:
                I = I.cuda()
            loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
            return loss

        def forward(self, logits, target, trans_feat):
            loss = F.cross_entropy(logits, target)
            mat_diff_loss = self.feature_transform_reguliarzer(trans_feat)

            total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
            return total_loss


class IOUtils:
    @staticmethod
    def load_model_params_from_txt(model: torch.nn.Module, directory: str):
        for name, param in model.named_parameters():
            file_path = os.path.join(directory, f"{name}.txt")
            if os.path.exists(file_path):
                param_data = np.loadtxt(file_path).reshape(param.shape).astype(np.float32)
                param.data = torch.from_numpy(param_data).to(param.device)
            else:
                print(f"Warning: {file_path} does not exist and will be skipped.")

        for name, buffer in model.named_buffers():
            file_path = os.path.join(directory, f"{name}.txt")
            if os.path.exists(file_path):
                buffer_data = np.loadtxt(file_path).reshape(buffer.shape).astype(np.float32)
                buffer.data = torch.from_numpy(buffer_data).to(buffer.device)
            else:
                print(f"Warning: {file_path} does not exist and will be skipped.")

    @staticmethod
    def load_param_from_txt(param: torch.Tensor, file_path: str):
        if os.path.exists(file_path):
            param_data = np.loadtxt(file_path).reshape(param.shape).astype(np.float32)
            param.data = torch.from_numpy(param_data).to(param.device)
        else:
            print(f"Warning: {file_path} does not exist and will be skipped.")

    @staticmethod
    def save_model_params_and_buffers_to_txt(model: torch.nn.Module, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)

        for name, param in model.named_parameters():
            np.savetxt(os.path.join(directory, f"{name}.txt"), param.detach().cpu().numpy().flatten())

        for name, buffer in model.named_buffers():
            np.savetxt(os.path.join(directory, f"{name}.txt"), buffer.detach().cpu().numpy().flatten())


class DatasetUtils:
    @staticmethod
    def sample_collate_fn(batch, num_points=1024):
        sampled_batch = []
        for points, target in batch:
            if points.shape[0] >= num_points:
                choice = np.random.choice(points.shape[0], num_points, replace=False)
            else:
                choice = np.random.choice(points.shape[0], num_points, replace=True)
            points = points[choice, :]
            sampled_batch.append((points, target))
        return torch.utils.data.dataloader.default_collate(sampled_batch)

    @staticmethod
    def shift_point_cloud(batch_data, shift_range=0.1):
        """Randomly shift point cloud. Shift is per point cloud.
        Input:
        BxNx3 tensor, original batch of point clouds
        Return:
        BxNx3 tensor, shifted batch of point clouds
        """
        B, N, C = batch_data.shape
        shifts = torch.FloatTensor(B, 3).uniform_(-shift_range, shift_range).to(batch_data.device)
        batch_data[:, :, :3] += shifts[:, None, :]
        return batch_data

    @staticmethod
    def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
        """Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 tensor, original batch of point clouds
        Return:
            BxNx3 tensor, scaled batch of point clouds
        """
        B, N, C = batch_data.shape
        scales = torch.FloatTensor(B).uniform_(scale_low, scale_high).to(batch_data.device)
        batch_data[:, :, :3] *= scales[:, None, None]
        return batch_data


class PointCloudDataset(Dataset):
    def __init__(self, root, split):
        self.list_of_points = []
        self.list_of_labels = []
        self.root = root
        self.split = split

        # with h5py.File(f"{split}_point_clouds.h5","r") as hf:
        with h5py.File(f"{self.root}/{self.split}_point_clouds.h5", "r") as hf:
            for k in hf.keys():
                self.list_of_points.append(hf[k]["points"][:].astype(np.float32))
                self.list_of_labels.append(hf[k].attrs["label"])

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, idx):
        points = self.list_of_points[idx]
        label = self.list_of_labels[idx]
        return points, label


# class PointCloudDataLoader(DataLoader):
#     def __init__(self, *args, shift_range=0.1, scale_low=0.8, scale_high=1.25, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.shift_range = shift_range
#         self.scale_low = scale_low
#         self.scale_high = scale_high

#     def __iter__(self):
#         for batch in super().__iter__():
#             points, labels = batch
#             points = DatasetUtils.shift_point_cloud(points, self.shift_range)
#             points = DatasetUtils.random_scale_point_cloud(points, self.scale_low, self.scale_high)
#             yield points, labels


def create_dataloader(root, split, batch_size, shuffle=True):
    dataset = PointCloudDataset(root, split)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=partial(DatasetUtils.sample_collate_fn, num_points=sample_points), num_workers=0
    )
    return dataloader


def do_inference(test_loader, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    correct = 0
    total = 0
    for points, target in test_loader:
        points = points.to(device)
        target = target.to(device)
        logits, _ = model(points.transpose(1, 2))
        pred_choice = logits.data.max(1)[1]
        correct += pred_choice.eq(target.data).cpu().sum()
        total += points.size()[0]
    accuracy_rate = correct.item() / total
    return accuracy_rate


def do_train(model, train_loader, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = PointNet.PointNetClassifierLoss()
    for epoch in range(num_epochs):
        model.train()
        loss_gather = []
        correct_count = 0
        total_count = 0
        for i, (points, target) in enumerate(train_loader):
            points = points.to(device)
            target = target.to(device)
            points = DatasetUtils.shift_point_cloud(points, 0.1)
            points = DatasetUtils.random_scale_point_cloud(points, 0.8, 1.25)
            model_output = model(points.transpose(1, 2))
            loss = criterion(model_output[0], target, model_output[1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_gather.append(loss.item())
            correct_count += torch.sum(torch.argmax(model_output[0], dim=1) == target).item()
            total_count += points.size(0)
        print(f"Epoch {epoch}, Average Loss {np.mean(loss_gather)}, Accuracy {correct_count / total_count}")

        accuracy = do_inference(test_loader, model)
        print(f"Epoch {epoch}, Validation Accuracy {accuracy}")
        if accuracy > target_accuracy:
            break


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    match mode:
        case "train":
            dir = "./models/weights"
            # 读取训练集数据
            data_path = "./data"
            train_loader = create_dataloader(data_path, "train", batch_size=train_batch_size)
            test_loader = create_dataloader(data_path, "test", batch_size=test_batch_size)
            # 搭建模型
            model = PointNet.PointNetClassifier().to(device)
            start = time.time()
            do_train(model, train_loader, test_loader)
            # 结束计时
            end = time.time()
            ms = end - start

            # 保存参数文件，请确保你提交的训练程序可以读取本程序存储的参数文件
            IOUtils.save_model_params_and_buffers_to_txt(model, dir)

            # 输出结果，请严格保持此输出格式，请不要输出除了此结果之外的任何内容！！！
            print(f"{ms:.4f}")
        case "test":
            # dir = os.path.dirname(__file__)  # 保存模型参数文件(.txt)的文件夹路径
            dir = "./models/weights"

            # 读取模型参数
            model = PointNet.PointNetClassifier().to(device)
            IOUtils.load_model_params_from_txt(model, dir)

            # 读取训练集数据
            test_loader = create_dataloader("./data", "test", batch_size=test_batch_size, shuffle=False)
            # warm up
            do_inference(test_loader, model)

            # 开始计时
            start = time.time()
            accuracy_rate = do_inference(test_loader, model)
            # 结束计时
            end = time.time()
            ms = end - start

            # 输出结果，请严格保持此输出格式，并把0.0001替换成实际的准确率，请不要输出除了此结果之外的任何内容！！！
            print(f"{ms:.4f}:{accuracy_rate:.4f}")

        case "debug":
            in_features = 987
            out_features = 679
            x_len = 351
            conv_torch = nn.Conv1d(in_features, out_features, 1).to(device)
            conv_triton = TritonLayers.Conv1dk1(in_features, out_features).to(device)
            x1 = torch.randn(x_len, in_features).to(device).transpose(-1, -2)
            x1.requires_grad = True
            x2 = x1.clone().detach()
            x2.requires_grad = True
            weight = torch.randn(out_features, in_features, 1).to(device)
            bias = torch.randn(out_features).to(device)
            # bias = torch.zeros(out_features).to(device)
            conv_torch.weight.data = weight
            conv_torch.bias.data = bias
            conv_triton.weight.data = weight
            conv_triton.bias.data = bias
            y_torch = conv_torch(x1)
            y_triton = conv_triton(x2)
            print(torch.allclose(y_torch, y_triton, rtol=1e-1))
            # print max diff
            print(torch.max(torch.abs(y_torch - y_triton)))

            y_torch.sum().backward()
            y_triton.sum().backward()
            print(torch.allclose(conv_torch.weight.grad, conv_triton.weight.grad, rtol=1e-1))
            print(x1.grad)
            print(x2.grad)
