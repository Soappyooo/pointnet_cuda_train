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
    # https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
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

    @triton.jit
    def max_d2_kernel(
        input_ptr,
        output_ptr,
        indices_ptr,
        stride_i,
        stride_o,
        N,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        input_ptr += pid * stride_i
        offs_i = tl.arange(0, BLOCK_SIZE)
        input_ptrs = input_ptr + offs_i
        input = tl.load(input_ptrs, mask=offs_i < N, other=-float("inf"))
        output, indice = tl.max(input, axis=0, return_indices=True)
        output_ptr += pid * stride_o
        output_ptrs = output_ptr + offs_i
        indices_ptr += pid * stride_o
        indices_ptrs = indices_ptr + offs_i
        # store the output only for the first pointer
        tl.store(output_ptrs, output, mask=offs_i < 1)
        tl.store(indices_ptrs, indice, mask=offs_i < 1)

    # https://github.com/BobMcDear/attorch/blob/main/attorch/batch_norm_kernels.py
    @triton.jit
    def batch_norm_forward_kernel(
        input_pointer,
        weight_pointer,
        bias_pointer,
        mean_pointer,
        inv_std_pointer,
        pre_act_add_pointer,
        pre_act_pointer,
        output_pointer,
        running_mean_pointer,
        running_var_pointer,
        batch_dim,
        spatial_dim,
        input_batch_stride,
        input_feat_stride,
        input_spatial_stride,
        pre_act_add_batch_stride,
        pre_act_add_feat_stride,
        pre_act_add_spatial_stride,
        pre_act_batch_stride,
        pre_act_feat_stride,
        pre_act_spatial_stride,
        output_batch_stride,
        output_feat_stride,
        output_spatial_stride,
        momentum,
        eps,
        param,
        affine: tl.constexpr,
        save_stats: tl.constexpr,
        track_running_stats: tl.constexpr,
        is_train: tl.constexpr,
        add_pre_act: tl.constexpr,
        act_func: tl.constexpr,
        save_pre_act: tl.constexpr,
        BLOCK_SIZE_BATCH: tl.constexpr,
        BLOCK_SIZE_SPATIAL: tl.constexpr,
    ):
        feat_pid = tl.program_id(axis=0)

        batch_offset = tl.arange(0, BLOCK_SIZE_BATCH)
        batch_mask = batch_offset < batch_dim

        if is_train or not track_running_stats:
            count = 0
            mean = 0.0
            var = 0.0

            for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
                spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
                spatial_mask = spatial_offset < spatial_dim

                curr_input_pointer = (
                    input_pointer
                    + input_feat_stride * feat_pid
                    + input_batch_stride * batch_offset[:, None]
                    + input_spatial_stride * spatial_offset[None, :]
                )
                curr_input = tl.load(curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]).to(tl.float32)

                spatial_count = min(BLOCK_SIZE_SPATIAL, spatial_dim - block_ind * BLOCK_SIZE_SPATIAL)
                curr_count = spatial_count * batch_dim
                count += curr_count

                prev_mean = mean
                mean += (tl.sum(curr_input) - curr_count * mean) / count
                deltas = tl.where(batch_mask[:, None] & spatial_mask[None, :], (curr_input - mean) * (curr_input - prev_mean), 0.0)
                var += tl.sum(deltas)

            var /= count
            inv_std = tl.rsqrt(var + eps)

            if save_stats:
                tl.store(feat_pid + mean_pointer, mean)
                tl.store(feat_pid + inv_std_pointer, inv_std)

            if track_running_stats:
                running_mean_pointer += feat_pid
                running_var_pointer += feat_pid

                running_mean = tl.load(running_mean_pointer)
                running_var = tl.load(running_var_pointer)

                n = batch_dim * spatial_dim
                tl.store(running_mean_pointer, (1 - momentum) * running_mean + momentum * mean)
                tl.store(running_var_pointer, (1 - momentum) * running_var + momentum * var * n / (n - 1))

        else:
            mean = tl.load(feat_pid + running_mean_pointer)
            inv_std = tl.rsqrt(tl.load(feat_pid + running_var_pointer) + eps)

        if affine:
            weight = tl.load(feat_pid + weight_pointer)
            bias = tl.load(feat_pid + bias_pointer)

        else:
            weight = 1.0
            bias = 0.0

        for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
            spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
            spatial_mask = spatial_offset < spatial_dim

            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_output_pointer = (
                output_pointer
                + output_feat_stride * feat_pid
                + output_batch_stride * batch_offset[:, None]
                + output_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]).to(tl.float32)
            output = weight * (curr_input - mean) * inv_std + bias

            if add_pre_act:
                curr_pre_act_add_pointer = (
                    pre_act_add_pointer
                    + pre_act_add_feat_stride * feat_pid
                    + pre_act_add_batch_stride * batch_offset[:, None]
                    + pre_act_add_spatial_stride * spatial_offset[None, :]
                )
                curr_pre_act_add = tl.load(curr_pre_act_add_pointer, mask=batch_mask[:, None] & spatial_mask[None, :])
                output += curr_pre_act_add

            if act_func is not None:
                if save_pre_act:
                    curr_pre_act_pointer = (
                        pre_act_pointer
                        + pre_act_feat_stride * feat_pid
                        + pre_act_batch_stride * batch_offset[:, None]
                        + pre_act_spatial_stride * spatial_offset[None, :]
                    )
                    tl.store(curr_pre_act_pointer, output, mask=batch_mask[:, None] & spatial_mask[None, :])

                output = apply_act_func(output, None, None, None, param, act_func, False)

            tl.store(curr_output_pointer, output, mask=batch_mask[:, None] & spatial_mask[None, :])

    @triton.jit
    def batch_norm_backward_kernel(
        output_grad_pointer,
        input_pointer,
        mean_pointer,
        inv_std_pointer,
        weight_pointer,
        input_grad_pointer,
        weight_grad_pointer,
        bias_grad_pointer,
        batch_dim,
        spatial_dim,
        output_grad_batch_stride,
        output_grad_feat_stride,
        output_grad_spatial_stride,
        input_batch_stride,
        input_feat_stride,
        input_spatial_stride,
        input_grad_batch_stride,
        input_grad_feat_stride,
        input_grad_spatial_stride,
        affine: tl.constexpr,
        BLOCK_SIZE_BATCH: tl.constexpr,
        BLOCK_SIZE_SPATIAL: tl.constexpr,
    ):
        feat_pid = tl.program_id(axis=0)

        batch_offset = tl.arange(0, BLOCK_SIZE_BATCH)
        batch_mask = batch_offset < batch_dim

        mean = tl.load(feat_pid + mean_pointer)
        inv_std = tl.load(feat_pid + inv_std_pointer)

        term1 = 0.0
        term2 = 0.0

        for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
            spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
            spatial_mask = spatial_offset < spatial_dim

            curr_output_grad_pointer = (
                output_grad_pointer
                + output_grad_feat_stride * feat_pid
                + output_grad_batch_stride * batch_offset[:, None]
                + output_grad_spatial_stride * spatial_offset[None, :]
            )
            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]).to(tl.float32)
            curr_pre_lin = (curr_input - mean) * inv_std
            curr_output_grad = tl.load(curr_output_grad_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]).to(tl.float32)

            term1 += tl.sum(curr_pre_lin * curr_output_grad)
            term2 += tl.sum(curr_output_grad)

        if affine:
            weight = tl.load(feat_pid + weight_pointer)
            weight_grad = 0.0
            bias_grad = 0.0

        else:
            weight = 1.0

        count = batch_dim * spatial_dim
        term1 *= weight / count
        term2 *= weight / count

        for block_ind in range(0, tl.cdiv(spatial_dim, BLOCK_SIZE_SPATIAL)):
            spatial_offset = block_ind * BLOCK_SIZE_SPATIAL + tl.arange(0, BLOCK_SIZE_SPATIAL)
            spatial_mask = spatial_offset < spatial_dim

            curr_output_grad_pointer = (
                output_grad_pointer
                + output_grad_feat_stride * feat_pid
                + output_grad_batch_stride * batch_offset[:, None]
                + output_grad_spatial_stride * spatial_offset[None, :]
            )
            curr_input_pointer = (
                input_pointer
                + input_feat_stride * feat_pid
                + input_batch_stride * batch_offset[:, None]
                + input_spatial_stride * spatial_offset[None, :]
            )
            curr_input_grad_pointer = (
                input_grad_pointer
                + input_grad_feat_stride * feat_pid
                + input_grad_batch_stride * batch_offset[:, None]
                + input_grad_spatial_stride * spatial_offset[None, :]
            )

            curr_input = tl.load(curr_input_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]).to(tl.float32)
            curr_pre_lin = (curr_input - mean) * inv_std
            curr_output_grad = tl.load(curr_output_grad_pointer, mask=batch_mask[:, None] & spatial_mask[None, :]).to(tl.float32)
            curr_input_grad = inv_std * (weight * curr_output_grad - (term1 * curr_pre_lin + term2))
            tl.store(curr_input_grad_pointer, curr_input_grad, mask=batch_mask[:, None] & spatial_mask[None, :])

            if affine:
                weight_grad += tl.sum(curr_pre_lin * curr_output_grad)
                bias_grad += tl.sum(curr_output_grad)

        if affine:
            tl.store(feat_pid + weight_grad_pointer, weight_grad)
            tl.store(feat_pid + bias_grad_pointer, bias_grad)

    # https://github.com/BobMcDear/attorch/blob/main/attorch/cross_entropy_loss_kernels.py
    @triton.jit
    def cross_entropy_loss_forward_kernel(
        input_pointer,
        target_pointer,
        weight_pointer,
        sum_weights_pointer,
        output_pointer,
        batch_dim,
        feat_dim,
        input_batch_stride,
        input_feat_stride,
        weighted: tl.constexpr,
        BLOCK_SIZE_BATCH: tl.constexpr,
        BLOCK_SIZE_FEAT: tl.constexpr,
    ):
        # This program processes BLOCK_SIZE_BATCH rows and BLOCK_SIZE_FEAT columns.
        batch_pid = tl.program_id(axis=0)

        batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
        feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)

        batch_mask = batch_offset < batch_dim
        feat_mask = feat_offset < feat_dim

        target = tl.load(target_pointer + batch_offset, mask=batch_mask)

        pred_pointer = input_pointer + input_feat_stride * target + input_batch_stride * batch_offset
        input_pointer += input_batch_stride * batch_offset[:, None] + input_feat_stride * feat_offset[None, :]

        input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :], other=-float("inf")).to(tl.float32)
        pred = tl.load(pred_pointer, mask=batch_mask).to(tl.float32)
        mx = tl.max(input, axis=1)
        input -= mx[:, None]
        loss = tl.log(tl.sum(tl.exp(input), axis=1)) - pred + mx

        if weighted:
            weight = tl.load(weight_pointer + target, mask=batch_mask).to(tl.float32)
            loss *= weight
            tl.store(sum_weights_pointer + batch_pid, tl.sum(weight))

        else:
            loss /= batch_dim

        tl.store(output_pointer + batch_pid, tl.sum(loss))

    @triton.jit
    def cross_entropy_loss_backward_kernel(
        output_grad_pointer,
        target_pointer,
        input_pointer,
        weight_pointer,
        sum_weights_pointer,
        input_grad_pointer,
        batch_dim,
        feat_dim,
        input_batch_stride,
        input_feat_stride,
        input_grad_batch_stride,
        input_grad_feat_stride,
        weighted: tl.constexpr,
        BLOCK_SIZE_BATCH: tl.constexpr,
        BLOCK_SIZE_FEAT: tl.constexpr,
    ):
        # This program processes BLOCK_SIZE_BATCH rows and BLOCK_SIZE_FEAT columns.
        batch_pid = tl.program_id(axis=0)

        batch_offset = batch_pid * BLOCK_SIZE_BATCH + tl.arange(0, BLOCK_SIZE_BATCH)
        feat_offset = tl.arange(0, BLOCK_SIZE_FEAT)

        batch_mask = batch_offset < batch_dim
        feat_mask = feat_offset < feat_dim

        input_pointer += input_batch_stride * batch_offset[:, None] + input_feat_stride * feat_offset[None, :]
        input_grad_pointer += input_grad_batch_stride * batch_offset[:, None] + input_grad_feat_stride * feat_offset[None, :]

        input = tl.load(input_pointer, mask=batch_mask[:, None] & feat_mask[None, :], other=-float("inf")).to(tl.float32)
        input -= tl.max(input, axis=1)[:, None]
        numerator = tl.exp(input)
        softmax = numerator / tl.sum(numerator, axis=1)[:, None]

        output_grad = tl.load(output_grad_pointer).to(tl.float32)
        target = tl.load(target_pointer + batch_offset, mask=batch_mask)
        broadcasted_feat_offset = tl.broadcast_to(feat_offset[None, :], (BLOCK_SIZE_BATCH, BLOCK_SIZE_FEAT))
        broadcasted_target = tl.broadcast_to(target[:, None], (BLOCK_SIZE_BATCH, BLOCK_SIZE_FEAT))
        input_grad = output_grad * (softmax - (broadcasted_feat_offset == broadcasted_target))

        if weighted:
            weight = tl.load(weight_pointer + target, mask=batch_mask).to(tl.float32)
            sum_weights = tl.load(sum_weights_pointer)
            input_grad *= weight[:, None] / sum_weights

        else:
            input_grad /= batch_dim

        tl.store(input_grad_pointer, input_grad, mask=batch_mask[:, None] & feat_mask[None, :])


class TritonOps:
    def matmul(a, b) -> torch.Tensor:
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

    def bmm(a, b) -> torch.Tensor:
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

    def linear(input, weight, bias) -> torch.Tensor:
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

    def conv1d_k1(input, weight, bias) -> torch.Tensor:
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

    def max_d2(input) -> torch.Tensor:
        assert input.dim() == 3, "Input must be 3D"
        # assert input.is_contiguous(), "Input must be contiguous"
        # if not input.is_contiguous():
        input = input.contiguous()
        B, M, N = input.shape
        # Allocates output.
        output = torch.empty((B, M), device=input.device, dtype=torch.float32)
        indices = torch.empty((B, M), device=input.device, dtype=torch.int64)
        input = input.view(-1, N)
        output = output.view(-1)
        indices = indices.view(-1)
        grid = lambda META: (input.size(0),)
        TritonKernels.max_d2_kernel[grid](
            input,
            output,
            indices,
            input.stride(0),
            output.stride(0),
            N,
            BLOCK_SIZE=triton.next_power_of_2(N),
        )

        output = output.view(B, M)
        indices = indices.view(B, M)
        return output, indices


class TritonFunctions:
    class LinearFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input, weight, bias) -> torch.Tensor:
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
        def forward(ctx, input, weight, bias) -> torch.Tensor:
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

    class ReLUFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input) -> torch.Tensor:
            ctx.save_for_backward(input)
            return input.clamp(min=0)

        @staticmethod
        def backward(ctx, grad_output):
            (input,) = ctx.saved_tensors
            return grad_output * (input > 0).float()

    def relu(input) -> torch.Tensor:
        return TritonFunctions.ReLUFunction.apply(input)

    class BmmFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a, b) -> torch.Tensor:
            ctx.save_for_backward(a, b)
            return TritonOps.bmm(a, b)

        @staticmethod
        def backward(ctx, grad_output):
            a, b = ctx.saved_tensors
            grad_a = TritonOps.bmm(grad_output, b.transpose(1, 2))
            grad_b = TritonOps.bmm(a.transpose(1, 2), grad_output)
            return grad_a, grad_b

    def bmm(a, b) -> torch.Tensor:
        return TritonFunctions.BmmFunction.apply(a, b)

    class MaxD2Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input) -> torch.Tensor:
            output = TritonOps.max_d2(input)
            ctx.save_for_backward(output[1], input)
            return output

        @staticmethod
        def backward(ctx, grad_output, grad_indices):
            (indices, input) = ctx.saved_tensors
            grad_input = torch.zeros_like(input)
            grad_input.scatter_(-1, indices.unsqueeze(-1), grad_output.unsqueeze(-1))
            return grad_input

    def max_d2(input, keepdim=False) -> torch.Tensor:
        if not keepdim:
            return TritonFunctions.MaxD2Function.apply(input)
        else:
            output, indices = TritonFunctions.MaxD2Function.apply(input)
            return output.unsqueeze(-1), indices.unsqueeze(-1)

    # https://github.com/BobMcDear/attorch/blob/main/attorch/batch_norm_layer.py
    class BatchNormFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            input,
            training: bool,
            weight,
            bias,
            running_mean,
            running_var,
            momentum: float = 0.1,
            eps: float = 1e-5,
            track_running_stats: bool = True,
            pre_act_add=None,
            act_func=None,
        ):
            def make_3d_for_bn(input):
                if input.ndim == 2:
                    input = input.unsqueeze(-1)

                elif input.ndim == 4:
                    input = input.flatten(2, -1)

                return input

            param = None
            if act_func is not None and "_" in act_func:
                comps = act_func.split("_")
                act_func = "_".join(comps[:-1])
                param = float(comps[-1])

            ctx.param = param
            ctx.act_func = act_func

            add_pre_act = pre_act_add is not None
            pre_act_add = pre_act_add if add_pre_act else torch.empty((1, 1, 1), device="cuda")

            input_3d = make_3d_for_bn(input)
            pre_act_add = make_3d_for_bn(pre_act_add)
            transpose = False

            if input_3d.shape[-1] > 1:
                input_3d = input_3d.transpose(0, -1)
                pre_act_add = pre_act_add.transpose(0, -1)
                transpose = True

            affine = weight is not None and bias is not None
            requires_grad = input.requires_grad or pre_act_add.requires_grad or (affine and weight.requires_grad) or (affine and bias.requires_grad)
            save_pre_act = requires_grad and (act_func is not None)

            batch_dim, feat_dim, spatial_dim = input_3d.shape
            output = torch.empty_like(input_3d)
            pre_act = torch.empty_like(input_3d) if save_pre_act else output

            if requires_grad:
                mean = torch.empty(feat_dim, device=input.device, dtype=torch.float32)
                inv_std = torch.empty(feat_dim, device=input.device, dtype=torch.float32)

            else:
                mean = inv_std = None

            running_mean = input if running_mean is None else running_mean
            running_var = input if running_var is None else running_var

            # Launches 1D grid where each program operates over one feature.
            grid = lambda _: (feat_dim,)
            TritonKernels.batch_norm_forward_kernel[grid](
                input_3d,
                weight,
                bias,
                mean,
                inv_std,
                pre_act_add,
                pre_act,
                output,
                running_mean,
                running_var,
                batch_dim,
                spatial_dim,
                *input_3d.stride(),
                *pre_act_add.stride(),
                *pre_act.stride(),
                *output.stride(),
                momentum,
                eps,
                param,
                affine=affine,
                save_stats=requires_grad,
                track_running_stats=track_running_stats,
                is_train=training,
                add_pre_act=add_pre_act,
                act_func=act_func,
                save_pre_act=save_pre_act,
                BLOCK_SIZE_BATCH=triton.next_power_of_2(batch_dim),
                BLOCK_SIZE_SPATIAL=min(max(2**14 // triton.next_power_of_2(batch_dim), 1), triton.next_power_of_2(spatial_dim)),
            )

            if transpose:
                output = output.transpose(0, -1)
                if save_pre_act:
                    pre_act = pre_act.transpose(0, -1)

            ctx.affine = affine
            ctx.act_func = act_func
            ctx.add_pre_act = add_pre_act
            if requires_grad:
                ctx.save_for_backward(input, mean, inv_std, weight, pre_act if save_pre_act else None)

            return output.view_as(input)

        @staticmethod
        def backward(
            ctx,
            output_grad,
        ):
            def make_3d_for_bn(input):
                if input.ndim == 2:
                    input = input.unsqueeze(-1)

                elif input.ndim == 4:
                    input = input.flatten(2, -1)

                return input

            (input, mean, inv_std, weight, pre_act) = ctx.saved_tensors
            input_3d = make_3d_for_bn(input)

            if ctx.act_func is None:
                pre_act_grad = make_3d_for_bn(output_grad)

            else:
                raise NotImplementedError

            transpose = False
            if input_3d.shape[-1] > 1:
                input_3d = input_3d.transpose(0, -1)
                pre_act_grad = pre_act_grad.transpose(0, -1)
                transpose = True

            batch_dim, feat_dim, spatial_dim = input_3d.shape
            input_grad = torch.empty_like(input_3d)

            if ctx.affine:
                weight_grad = torch.empty((feat_dim,), device=input.device)
                bias_grad = torch.empty_like(weight_grad)

            else:
                weight_grad = bias_grad = None

            # Launches 1D grid where each program operates over one feature.
            grid = lambda _: (feat_dim,)
            TritonKernels.batch_norm_backward_kernel[grid](
                pre_act_grad,
                input_3d,
                mean,
                inv_std,
                weight,
                input_grad,
                weight_grad,
                bias_grad,
                batch_dim,
                spatial_dim,
                *pre_act_grad.stride(),
                *input_3d.stride(),
                *input_grad.stride(),
                affine=ctx.affine,
                BLOCK_SIZE_BATCH=triton.next_power_of_2(batch_dim),
                BLOCK_SIZE_SPATIAL=min(max(2**14 // triton.next_power_of_2(batch_dim), 1), triton.next_power_of_2(spatial_dim)),
            )

            if transpose:
                input_grad = input_grad.transpose(0, -1)
                pre_act_grad = pre_act_grad.transpose(0, -1)

            # Pads output with None because a gradient is necessary for
            # all input arguments.
            return (
                input_grad.view_as(input),
                None,
                weight_grad,
                bias_grad,
                None,
                None,
                None,
                None,
                None,
                pre_act_grad.view_as(input) if ctx.add_pre_act else None,
                None,
            )

    # https://github.com/BobMcDear/attorch/blob/main/attorch/cross_entropy_loss_layer.py
    class CrossEntropyLossFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            input,
            target,
            weight=None,
        ):
            assert input.ndim == 2, f"Inputs of rank other than 2 not valid"
            assert len(input) == len(target), f"Incompatible input shape ({input.shape}) and target shape ({target.shape})"
            assert (
                weight is None or len(weight) == input.shape[1]
            ), f"Dimensionality of weight vector ({len(weight)}) and input features ({input.shape[1]}) not equal"

            batch_dim, feat_dim = input.shape
            weighted = weight is not None

            # output_dtype = get_output_dtype(input.dtype, autocast="fp32")
            output = torch.empty(batch_dim, dtype=input.dtype, device=input.device)

            if weighted:
                sum_weights = torch.empty_like(output, dtype=torch.float32)

            else:
                sum_weights = None

            # Launches 1D grid where each program operates over BLOCK_SIZE_BATCH rows.
            grid = lambda META: (triton.cdiv(len(input), META["BLOCK_SIZE_BATCH"]),)
            TritonKernels.cross_entropy_loss_forward_kernel[grid](
                input,
                target,
                weight,
                sum_weights,
                output,
                batch_dim,
                feat_dim,
                *input.stride(),
                weighted=weighted,
                BLOCK_SIZE_BATCH=1,
                BLOCK_SIZE_FEAT=triton.next_power_of_2(feat_dim),
            )
            output = output.sum()

            if weighted:
                sum_weights = sum_weights.sum()
                output /= sum_weights

            ctx.sum_weights = sum_weights
            ctx.weight = weight
            ctx.output_dtype = output.dtype
            if input.requires_grad:
                ctx.save_for_backward(input, target)

            return output

        @staticmethod
        def backward(
            ctx,
            output_grad,
        ):
            (input, target) = ctx.saved_tensors
            batch_dim, feat_dim = input.shape
            input_grad = torch.empty_like(input, dtype=ctx.output_dtype)

            # Launches 1D grid where each program operates over BLOCK_SIZE_BATCH rows.
            grid = lambda META: (triton.cdiv(len(input), META["BLOCK_SIZE_BATCH"]),)
            TritonKernels.cross_entropy_loss_backward_kernel[grid](
                output_grad,
                target,
                input,
                ctx.weight,
                ctx.sum_weights,
                input_grad,
                batch_dim,
                feat_dim,
                *input.stride(),
                *input_grad.stride(),
                weighted=ctx.weight is not None,
                BLOCK_SIZE_BATCH=1,
                BLOCK_SIZE_FEAT=triton.next_power_of_2(feat_dim),
            )

            # Pads output with None because a gradient is necessary for
            # all input arguments.
            return input_grad, None, None


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

    class BatchNorm1d(nn.Module):
        def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
            super(TritonLayers.BatchNorm1d, self).__init__()
            self.num_features = num_features
            self.eps = eps
            self.momentum = momentum
            self.affine = affine
            self.track_running_stats = track_running_stats
            if self.affine:
                self.weight = nn.Parameter(torch.empty(num_features, dtype=torch.float32))
                self.bias = nn.Parameter(torch.empty(num_features, dtype=torch.float32))
            else:
                self.register_parameter("weight", None)
                self.register_parameter("bias", None)
            if self.track_running_stats:
                self.register_buffer("running_mean", torch.zeros(num_features, dtype=torch.float32))
                self.register_buffer("running_var", torch.ones(num_features, dtype=torch.float32))
            else:
                self.register_parameter("running_mean", None)
                self.register_parameter("running_var", None)
            self.reset_parameters()

        def reset_parameters(self):
            if self.track_running_stats:
                self.running_mean.zero_()
                self.running_var.fill_(1)
            if self.affine:
                self.weight.data.uniform_()
                self.bias.data.zero_()

        def forward(self, input):
            return TritonFunctions.BatchNormFunction.apply(
                input,
                self.training,
                self.weight,
                self.bias,
                self.running_mean,
                self.running_var,
                self.momentum,
                self.eps,
                self.track_running_stats,
                None,
                None,
            )

    class CrossEntropyLoss(nn.CrossEntropyLoss):
        def __init__(
            self,
            reduction: str = "mean",
            size_average=None,
            weight=None,
            ignore_index: int = -100,
        ) -> None:
            super().__init__(weight, size_average, ignore_index, reduction=reduction)

        def forward(self, input, target):
            return TritonFunctions.CrossEntropyLossFunction.apply(input, target, self.weight)


class PointNet:
    class STN3d(nn.Module):
        def __init__(self, channel):
            super(PointNet.STN3d, self).__init__()
            self.conv1 = TritonLayers.Conv1dk1(channel, 64)
            self.conv2 = TritonLayers.Conv1dk1(64, 128)
            self.conv3 = TritonLayers.Conv1dk1(128, 1024)
            self.fc1 = TritonLayers.Linear(1024, 512)
            self.fc2 = TritonLayers.Linear(512, 256)
            self.fc3 = TritonLayers.Linear(256, 9)
            self.bn1 = TritonLayers.BatchNorm1d(64)
            self.bn2 = TritonLayers.BatchNorm1d(128)
            self.bn3 = TritonLayers.BatchNorm1d(1024)
            self.bn4 = TritonLayers.BatchNorm1d(512)
            self.bn5 = TritonLayers.BatchNorm1d(256)

        def forward(self, x):
            batchsize = x.size()[0]
            x = TritonFunctions.relu(self.bn1(self.conv1(x)))
            x = TritonFunctions.relu(self.bn2(self.conv2(x)))
            x = TritonFunctions.relu(self.bn3(self.conv3(x)))
            x = TritonFunctions.max_d2(x, keepdim=True)[0]
            x = x.view(-1, 1024)

            x = TritonFunctions.relu(self.bn4(self.fc1(x)))
            x = TritonFunctions.relu(self.bn5(self.fc2(x)))
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
            self.conv1 = TritonLayers.Conv1dk1(k, 64)
            self.conv2 = TritonLayers.Conv1dk1(64, 128)
            self.conv3 = TritonLayers.Conv1dk1(128, 1024)
            self.fc1 = TritonLayers.Linear(1024, 512)
            self.fc2 = TritonLayers.Linear(512, 256)
            self.fc3 = TritonLayers.Linear(256, k * k)
            self.bn1 = TritonLayers.BatchNorm1d(64)
            self.bn2 = TritonLayers.BatchNorm1d(128)
            self.bn3 = TritonLayers.BatchNorm1d(1024)
            self.bn4 = TritonLayers.BatchNorm1d(512)
            self.bn5 = TritonLayers.BatchNorm1d(256)

            self.k = k

        def forward(self, x):
            batchsize = x.size()[0]
            x = TritonFunctions.relu(self.bn1(self.conv1(x)))
            x = TritonFunctions.relu(self.bn2(self.conv2(x)))
            x = TritonFunctions.relu(self.bn3(self.conv3(x)))
            x = TritonFunctions.max_d2(x, keepdim=True)[0]
            x = x.view(-1, 1024)

            x = TritonFunctions.relu(self.bn4(self.fc1(x)))
            x = TritonFunctions.relu(self.bn5(self.fc2(x)))
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
            self.conv1 = TritonLayers.Conv1dk1(channel, 64)
            self.conv2 = TritonLayers.Conv1dk1(64, 128)
            self.conv3 = TritonLayers.Conv1dk1(128, 1024)
            self.bn1 = TritonLayers.BatchNorm1d(64)
            self.bn2 = TritonLayers.BatchNorm1d(128)
            self.bn3 = TritonLayers.BatchNorm1d(1024)
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
            x = TritonFunctions.bmm(x, trans)
            if D > 3:
                x = torch.cat([x, feature], dim=2)
            x = x.transpose(2, 1)
            x = TritonFunctions.relu(self.bn1(self.conv1(x)))

            if self.feature_transform:
                trans_feat = self.fstn(x)
                x = x.transpose(2, 1)
                x = TritonFunctions.bmm(x, trans_feat)
                x = x.transpose(2, 1)
            else:
                trans_feat = None

            pointfeat = x
            x = TritonFunctions.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
            x = TritonFunctions.max_d2(x, keepdim=True)[0]
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
            self.fc1 = TritonLayers.Linear(1024, 512)
            self.fc2 = TritonLayers.Linear(512, 256)
            self.fc3 = TritonLayers.Linear(256, k)
            self.bn1 = TritonLayers.BatchNorm1d(512)
            self.bn2 = TritonLayers.BatchNorm1d(256)

        def forward(self, x):
            x, trans, trans_feat = self.feat(x)
            x = TritonFunctions.relu(self.bn1(self.fc1(x)))
            x = TritonFunctions.relu(self.bn2(self.fc2(x)))
            x = self.fc3(x)
            return x, trans_feat


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
    with torch.no_grad():
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
    criterion = TritonLayers.CrossEntropyLoss()
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
            loss = criterion(model_output[0], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_gather.append(loss.item())
            correct_count += torch.sum(torch.argmax(model_output[0], dim=1) == target).item()
            total_count += points.size(0)
        # print(f"Epoch {epoch}, Average Loss {np.mean(loss_gather)}, Accuracy {correct_count / total_count}")

        accuracy = do_inference(test_loader, model)
        # print(f"Epoch {epoch}, Validation Accuracy {accuracy}")
        if accuracy > target_accuracy:
            break


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if mode == "train":
        dir = "./models/weights"
        # 读取训练集数据
        data_path = "./data"
        train_loader = create_dataloader(data_path, "train", batch_size=train_batch_size)
        test_loader = create_dataloader(data_path, "test", batch_size=test_batch_size)
        # 搭建模型
        model = PointNet.PointNetClassifier().to(device)
        # 开始计时
        start = time.time()
        do_train(model, train_loader, test_loader)
        # 结束计时
        end = time.time()
        ms = end - start

        # 保存参数文件，请确保你提交的训练程序可以读取本程序存储的参数文件
        IOUtils.save_model_params_and_buffers_to_txt(model, dir)

        # 输出结果，请严格保持此输出格式，请不要输出除了此结果之外的任何内容！！！
        print(f"{ms:.4f}")

    elif mode == "test":
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
