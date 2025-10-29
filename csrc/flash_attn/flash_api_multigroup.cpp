/******************************************************************************
 * Copyright (c) 2024, Multi-Group Flash Attention Implementation.
 *
 * Python bindings for multi-group varlen attention.
 ******************************************************************************/

// Include these 2 headers instead of torch/extension.h since we don't need all of the torch headers.
#include <torch/python.h>
#include <torch/nn/functional.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <cutlass/numeric_types.h>

#include "namespace_config.h"
#include "hardware_info.h"
#include "flash_multigroup.h"
#include "static_switch.h"

#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Parameter Setup Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

void set_params_fprop_multigroup(
    Flash_fwd_multigroup_params &params,
    const std::vector<at::Tensor> &q_list,
    const at::Tensor &k,
    const at::Tensor &v,
    std::vector<at::Tensor> &out_list,
    const std::vector<at::Tensor> &cu_seqlens_q_list,
    const std::vector<at::Tensor> &cu_seqlens_k_list,
    const at::Tensor &kv_endpoints,
    const std::vector<int64_t> &max_seqlen_q_list,
    const std::vector<int64_t> &max_seqlen_k_list,
    float softmax_scale,
    float softcap,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float p_dropout,
    const c10::optional<at::Tensor> &alibi_slopes_,
    bool unpadded_lse
) {
    // Reset params
    params = {};

    // Extract basic dimensions
    const int num_groups = q_list.size();
    TORCH_CHECK(num_groups >= 1 && num_groups <= 8, "num_groups must be between 1 and 8");
    TORCH_CHECK(q_list.size() == cu_seqlens_q_list.size(),
                "q_list and cu_seqlens_q_list must have same size");
    TORCH_CHECK(q_list.size() == cu_seqlens_k_list.size(),
                "q_list and cu_seqlens_k_list must have same size");
    TORCH_CHECK(q_list.size() == out_list.size(),
                "q_list and out_list must have same size");

    params.num_groups = num_groups;

    // Get batch size from cu_seqlens
    const int batch_size = cu_seqlens_q_list[0].numel() - 1;
    params.batch_size = batch_size;

    // Get dimensions from first Q tensor
    const int num_heads = q_list[0].size(1);
    const int head_size = q_list[0].size(2);
    const int num_heads_k = k.size(1);

    params.h = num_heads;
    params.h_k = num_heads_k;
    params.h_h_k_ratio = num_heads / num_heads_k;
    params.head_size = head_size;

    // Round head size
    auto round_multiple = [](int x, int m) { return (x + m - 1) / m * m; };
    params.head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);

    // Data type
    params.is_bf16 = q_list[0].dtype() == torch::kBFloat16;

    // Allocate per-group pointer arrays
    // Note: These will be managed by the caller and should persist until kernel completes
    params.q_ptr_list = new void*[num_groups];
    params.out_ptr_list = new void*[num_groups];
    params.softmax_lse_ptr_list = new void*[num_groups];
    params.cu_seqlens_q_list = new int*[num_groups];
    params.cu_seqlens_k_list = new int*[num_groups];

    params.q_row_stride_list = new int64_t[num_groups];
    params.q_head_stride_list = new int64_t[num_groups];
    params.o_row_stride_list = new int64_t[num_groups];
    params.o_head_stride_list = new int64_t[num_groups];

    params.max_seqlen_q_list = new int[num_groups];
    params.max_seqlen_k_list = new int[num_groups];
    params.total_q_list = new int[num_groups];

    // Fill per-group arrays
    for (int g = 0; g < num_groups; g++) {
        // Validate Q tensor
        TORCH_CHECK(q_list[g].dtype() == q_list[0].dtype(),
                    "All Q tensors must have same dtype");
        TORCH_CHECK(q_list[g].size(1) == num_heads,
                    "All Q tensors must have same number of heads");
        TORCH_CHECK(q_list[g].size(2) == head_size,
                    "All Q tensors must have same head dimension");
        CHECK_DEVICE(q_list[g]);
        TORCH_CHECK(q_list[g].stride(-1) == 1, "Q tensor must have contiguous last dimension");

        // Q pointers and strides
        params.q_ptr_list[g] = q_list[g].data_ptr();
        params.q_row_stride_list[g] = q_list[g].stride(0);
        params.q_head_stride_list[g] = q_list[g].stride(1);

        // Output pointers and strides
        params.out_ptr_list[g] = out_list[g].data_ptr();
        params.o_row_stride_list[g] = out_list[g].stride(0);
        params.o_head_stride_list[g] = out_list[g].stride(1);

        // Sequence length metadata
        params.cu_seqlens_q_list[g] = static_cast<int*>(cu_seqlens_q_list[g].data_ptr());
        params.cu_seqlens_k_list[g] = static_cast<int*>(cu_seqlens_k_list[g].data_ptr());

        params.max_seqlen_q_list[g] = max_seqlen_q_list[g];
        params.max_seqlen_k_list[g] = max_seqlen_k_list[g];

        // Total Q tokens for this group
        params.total_q_list[g] = q_list[g].size(0);
    }

    // Shared K, V pointers and strides
    CHECK_DEVICE(k); CHECK_DEVICE(v);
    TORCH_CHECK(k.stride(-1) == 1, "K tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "V tensor must have contiguous last dimension");
    TORCH_CHECK(k.dtype() == q_list[0].dtype(), "K must have same dtype as Q");
    TORCH_CHECK(v.dtype() == q_list[0].dtype(), "V must have same dtype as Q");

    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.k_row_stride = k.stride(0);
    params.k_head_stride = k.stride(1);
    params.v_row_stride = v.stride(0);
    params.v_head_stride = v.stride(1);

    // KV endpoints
    CHECK_DEVICE(kv_endpoints);
    CHECK_CONTIGUOUS(kv_endpoints);
    TORCH_CHECK(kv_endpoints.dtype() == torch::kInt32, "kv_endpoints must have dtype int32");
    TORCH_CHECK(kv_endpoints.size(0) == num_groups,
                "kv_endpoints must have shape [num_groups, batch_size]");
    TORCH_CHECK(kv_endpoints.size(1) == batch_size,
                "kv_endpoints must have shape [num_groups, batch_size]");
    params.kv_endpoints = static_cast<int*>(kv_endpoints.data_ptr());

    // Softmax scale
    if (softcap > 0.0f) {
        params.softcap = softmax_scale / softcap;
        params.scale_softmax = softcap;
        params.scale_softmax_log2 = softcap * M_LOG2E;
    } else {
        params.softcap = 0.0f;
        params.scale_softmax = softmax_scale;
        params.scale_softmax_log2 = softmax_scale * M_LOG2E;
    }

    // Masking
    params.is_causal = is_causal;
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;

    // Dropout
    params.p_dropout = 1.0f - p_dropout;  // Convert to keep probability
    params.p_dropout_in_uint8_t = uint8_t(std::floor(params.p_dropout * 255.0));
    params.rp_dropout = 1.0f / params.p_dropout;
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    // Alibi slopes
    if (alibi_slopes_.has_value()) {
        auto alibi_slopes = alibi_slopes_.value();
        TORCH_CHECK(alibi_slopes.dtype() == torch::kFloat32, "ALiBi slopes must have dtype fp32");
        CHECK_DEVICE(alibi_slopes);
        TORCH_CHECK(alibi_slopes.stride(-1) == 1, "ALiBi slopes tensor must have contiguous last dimension");
        params.alibi_slopes_ptr = alibi_slopes.data_ptr();
        params.alibi_slopes_batch_stride = alibi_slopes.dim() == 2 ? alibi_slopes.stride(0) : 0;
    } else {
        params.alibi_slopes_ptr = nullptr;
    }

    // Format flags
    params.unpadded_lse = unpadded_lse;
    params.is_seqlens_k_cumulative = true;

    // Debug buffer (disabled by default)
    params.debug_buffer = nullptr;
}

void set_params_dgrad_multigroup(
    Flash_bwd_multigroup_params &params,
    const std::vector<at::Tensor> &q_list,
    const at::Tensor &k,
    const at::Tensor &v,
    const std::vector<at::Tensor> &out_list,
    const std::vector<at::Tensor> &cu_seqlens_q_list,
    const std::vector<at::Tensor> &cu_seqlens_k_list,
    const at::Tensor &kv_endpoints,
    const std::vector<at::Tensor> &dout_list,
    std::vector<at::Tensor> &dq_list,
    at::Tensor &dk,
    at::Tensor &dv,
    const std::vector<at::Tensor> &softmax_lse_list,
    const std::vector<int64_t> &max_seqlen_q_list,
    const std::vector<int64_t> &max_seqlen_k_list,
    float softmax_scale,
    float softcap,
    bool is_causal,
    int window_size_left,
    int window_size_right,
    float p_dropout,
    const c10::optional<at::Tensor> &alibi_slopes_,
    bool deterministic,
    bool unpadded_lse
) {
    // First, setup forward parameters
    std::vector<at::Tensor> out_list_mut = out_list;
    set_params_fprop_multigroup(
        params, q_list, k, v, out_list_mut, cu_seqlens_q_list, cu_seqlens_k_list,
        kv_endpoints, max_seqlen_q_list, max_seqlen_k_list, softmax_scale, softcap,
        is_causal, window_size_left, window_size_right, p_dropout, alibi_slopes_, unpadded_lse
    );

    const int num_groups = q_list.size();

    // Allocate backward-specific arrays
    params.do_ptr_list = new void*[num_groups];
    params.dq_ptr_list = new void*[num_groups];
    params.dq_accum_ptr_list = new void*[num_groups];
    params.dsoftmax_sum_list = new void*[num_groups];

    params.do_row_stride_list = new int64_t[num_groups];
    params.do_head_stride_list = new int64_t[num_groups];
    params.dq_row_stride_list = new int64_t[num_groups];
    params.dq_head_stride_list = new int64_t[num_groups];

    // Fill backward-specific arrays
    for (int g = 0; g < num_groups; g++) {
        // dO pointers and strides
        CHECK_DEVICE(dout_list[g]);
        TORCH_CHECK(dout_list[g].stride(-1) == 1, "dout must have contiguous last dimension");
        params.do_ptr_list[g] = dout_list[g].data_ptr();
        params.do_row_stride_list[g] = dout_list[g].stride(0);
        params.do_head_stride_list[g] = dout_list[g].stride(1);

        // dQ pointers and strides
        CHECK_DEVICE(dq_list[g]);
        TORCH_CHECK(dq_list[g].stride(-1) == 1, "dq must have contiguous last dimension");
        params.dq_ptr_list[g] = dq_list[g].data_ptr();
        params.dq_row_stride_list[g] = dq_list[g].stride(0);
        params.dq_head_stride_list[g] = dq_list[g].stride(1);

        // LSE (already set in forward params, but need to update for backward)
        params.softmax_lse_ptr_list[g] = softmax_lse_list[g].data_ptr();

        // dQ accum and dsoftmax_sum will be allocated if needed
        params.dq_accum_ptr_list[g] = nullptr;
        params.dsoftmax_sum_list[g] = nullptr;
    }

    // Shared dK, dV pointers and strides
    CHECK_DEVICE(dk); CHECK_DEVICE(dv);
    TORCH_CHECK(dk.stride(-1) == 1, "dK must have contiguous last dimension");
    TORCH_CHECK(dv.stride(-1) == 1, "dV must have contiguous last dimension");

    params.dk_ptr = dk.data_ptr();
    params.dv_ptr = dv.data_ptr();
    params.dk_row_stride = dk.stride(0);
    params.dk_head_stride = dk.stride(1);
    params.dv_row_stride = dv.stride(0);
    params.dv_head_stride = dv.stride(1);

    // Accumulators (disabled for now - will use atomics)
    params.dk_accum_ptr = nullptr;
    params.dv_accum_ptr = nullptr;

    // Backward flags
    params.deterministic = deterministic;
    params.dq_accum_split_stride = 0;
    params.grad_accum_strategy = deterministic ? 1 : 0;  // 0=atomic, 1=two-pass
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Forward Python Binding
////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<std::vector<at::Tensor>, std::vector<at::Tensor>>
mha_varlen_multigroup_fwd(
    const std::vector<at::Tensor> &q_list,
    const at::Tensor &k,
    const at::Tensor &v,
    const std::vector<at::Tensor> &cu_seqlens_q_list,
    const std::vector<at::Tensor> &cu_seqlens_k_list,
    const at::Tensor &kv_endpoints,
    const std::vector<int64_t> &max_seqlen_q_list,
    const std::vector<int64_t> &max_seqlen_k_list,
    const float p_dropout,
    const float softmax_scale,
    const bool is_causal,
    const int window_size_left,
    const int window_size_right,
    const float softcap,
    const c10::optional<at::Tensor> &alibi_slopes_,
    const bool return_softmax
) {
    // Set CUDA device guard
    at::cuda::CUDAGuard device_guard{q_list[0].device()};

    // Check GPU compute capability
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x_min = cc_major >= 8;
    TORCH_CHECK(is_sm8x_min, "Multi-group FlashAttention only supports Ampere GPUs or newer.");

    const int num_groups = q_list.size();
    TORCH_CHECK(num_groups >= 1 && num_groups <= 8, "num_groups must be between 1 and 8");

    // Validate input tensors
    auto q_dtype = q_list[0].dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "FlashAttention only supports fp16 and bf16 data type");

    const int num_heads = q_list[0].size(1);
    const int head_size = q_list[0].size(2);

    TORCH_CHECK(head_size <= 256, "FlashAttention forward only supports head dimension at most 256");
    TORCH_CHECK(head_size % 8 == 0, "head_size must be a multiple of 8");

    // Allocate output tensors
    std::vector<at::Tensor> out_list;
    std::vector<at::Tensor> lse_list;

    auto opts = q_list[0].options();

    for (int g = 0; g < num_groups; g++) {
        // Output tensor: same shape as Q
        out_list.push_back(torch::empty_like(q_list[g]));

        // LSE tensor: [nheads, total_q] for unpadded format
        const int total_q = q_list[g].size(0);
        lse_list.push_back(torch::empty({num_heads, total_q}, opts.dtype(at::kFloat)));
    }

    // Setup parameters
    Flash_fwd_multigroup_params params;
    set_params_fprop_multigroup(
        params, q_list, k, v, out_list, cu_seqlens_q_list, cu_seqlens_k_list,
        kv_endpoints, max_seqlen_q_list, max_seqlen_k_list,
        softmax_scale, softcap, is_causal, window_size_left, window_size_right,
        p_dropout, alibi_slopes_, /*unpadded_lse=*/true
    );

    // Update LSE pointers
    for (int g = 0; g < num_groups; g++) {
        params.softmax_lse_ptr_list[g] = lse_list[g].data_ptr();
    }

    // Setup RNG for dropout
    if (p_dropout > 0.0f) {
        auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
        auto rng_state = torch::empty({2}, options.dtype(torch::kInt64));
        params.rng_state = reinterpret_cast<uint64_t*>(rng_state.data_ptr());

        // Note: For multi-group, we need separate RNG state per group
        // This is a simplification - production code would need per-group RNG
        auto gen = at::get_generator_or_default<at::CUDAGeneratorImpl>(
            c10::nullopt, at::cuda::detail::getDefaultCUDAGenerator());
        std::lock_guard<std::mutex> lock(gen->mutex_);
        int64_t counter_offset = params.batch_size * params.h * 32;
        params.philox_args = gen->philox_cuda_state(counter_offset);
    }

    // Validate parameters
    validate_multigroup_params(params);

    // Get CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // TODO: Dispatch to kernel - for now, throw error
    TORCH_CHECK(false, "Multi-group forward kernel not yet implemented. "
                       "Please implement Phase 2 CUDA kernels first.");

    // Once kernels are implemented, use:
    // HEADDIM_SWITCH(head_size, [&] {
    //     NUMGROUPS_SWITCH(num_groups, [&] {
    //         BOOL_SWITCH(is_causal, Is_causal, [&] {
    //             if (params.is_bf16) {
    //                 run_mha_fwd_multigroup<cutlass::bfloat16_t, kHeadDim, kNumGroups, Is_causal>(
    //                     params, stream
    //                 );
    //             } else {
    //                 run_mha_fwd_multigroup<cutlass::half_t, kHeadDim, kNumGroups, Is_causal>(
    //                     params, stream
    //                 );
    //             }
    //         });
    //     });
    // });

    // Cleanup (caller manages pointer lifetime)

    return std::make_tuple(out_list, lse_list);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Backward Python Binding
////////////////////////////////////////////////////////////////////////////////////////////////////

std::tuple<std::vector<at::Tensor>, at::Tensor, at::Tensor>
mha_varlen_multigroup_bwd(
    const std::vector<at::Tensor> &dout_list,
    const std::vector<at::Tensor> &q_list,
    const at::Tensor &k,
    const at::Tensor &v,
    const std::vector<at::Tensor> &out_list,
    const std::vector<at::Tensor> &softmax_lse_list,
    const std::vector<at::Tensor> &cu_seqlens_q_list,
    const std::vector<at::Tensor> &cu_seqlens_k_list,
    const at::Tensor &kv_endpoints,
    const std::vector<int64_t> &max_seqlen_q_list,
    const std::vector<int64_t> &max_seqlen_k_list,
    const float p_dropout,
    const float softmax_scale,
    const bool is_causal,
    const int window_size_left,
    const int window_size_right,
    const float softcap,
    const c10::optional<at::Tensor> &alibi_slopes_,
    const bool deterministic
) {
    // Set CUDA device guard
    at::cuda::CUDAGuard device_guard{q_list[0].device()};

    const int num_groups = q_list.size();
    TORCH_CHECK(num_groups >= 1 && num_groups <= 8, "num_groups must be between 1 and 8");

    // Allocate gradient tensors
    std::vector<at::Tensor> dq_list;
    for (int g = 0; g < num_groups; g++) {
        dq_list.push_back(torch::empty_like(q_list[g]));
    }

    // Shared K, V gradients
    at::Tensor dk = torch::zeros_like(k);  // Must be zeros for accumulation
    at::Tensor dv = torch::zeros_like(v);

    // Setup backward parameters
    Flash_bwd_multigroup_params params;
    set_params_dgrad_multigroup(
        params, q_list, k, v, out_list, cu_seqlens_q_list, cu_seqlens_k_list,
        kv_endpoints, dout_list, dq_list, dk, dv, softmax_lse_list,
        max_seqlen_q_list, max_seqlen_k_list, softmax_scale, softcap,
        is_causal, window_size_left, window_size_right, p_dropout, alibi_slopes_,
        deterministic, /*unpadded_lse=*/true
    );

    // Validate parameters
    validate_multigroup_params(params);

    // Get CUDA stream
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // TODO: Dispatch to kernel
    TORCH_CHECK(false, "Multi-group backward kernel not yet implemented. "
                       "Please implement Phase 3 CUDA kernels first.");

    return std::make_tuple(dq_list, dk, dv);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PyTorch Module Registration
////////////////////////////////////////////////////////////////////////////////////////////////////

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwd", &mha_varlen_multigroup_fwd,
          "Multi-group varlen flash attention forward",
          py::arg("q_list"),
          py::arg("k"),
          py::arg("v"),
          py::arg("cu_seqlens_q_list"),
          py::arg("cu_seqlens_k_list"),
          py::arg("kv_endpoints"),
          py::arg("max_seqlen_q_list"),
          py::arg("max_seqlen_k_list"),
          py::arg("dropout_p") = 0.0,
          py::arg("softmax_scale") = 0.0,
          py::arg("causal") = false,
          py::arg("window_size_left") = -1,
          py::arg("window_size_right") = -1,
          py::arg("softcap") = 0.0,
          py::arg("alibi_slopes") = c10::nullopt,
          py::arg("return_softmax") = false);

    m.def("bwd", &mha_varlen_multigroup_bwd,
          "Multi-group varlen flash attention backward",
          py::arg("dout_list"),
          py::arg("q_list"),
          py::arg("k"),
          py::arg("v"),
          py::arg("out_list"),
          py::arg("softmax_lse_list"),
          py::arg("cu_seqlens_q_list"),
          py::arg("cu_seqlens_k_list"),
          py::arg("kv_endpoints"),
          py::arg("max_seqlen_q_list"),
          py::arg("max_seqlen_k_list"),
          py::arg("dropout_p") = 0.0,
          py::arg("softmax_scale") = 0.0,
          py::arg("causal") = false,
          py::arg("window_size_left") = -1,
          py::arg("window_size_right") = -1,
          py::arg("softcap") = 0.0,
          py::arg("alibi_slopes") = c10::nullopt,
          py::arg("deterministic") = false);
}

}  // namespace FLASH_NAMESPACE
