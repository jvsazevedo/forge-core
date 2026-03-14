// Forge SiLU (Swish) CUDA kernel.
//
// Forward:  silu(x) = x / (1 + exp(-x))   [equivalent to x * sigmoid(x)]
// Backward: grad * s * (1 + x * (1 - s))  where s = sigmoid(x)
//
// Reference: PyTorch aten/src/ATen/native/cuda/ActivationSiluKernel.cu

#include "cuda_utils.cuh"
#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ─── Forward ────────────────────────────────────────────────────────────

#define SILU_FWD(TYPENAME, EXPFN, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *info, \
    const TYPENAME *inp, \
    TYPENAME *out \
) { \
    const size_t *dims = info; \
    const size_t *strides = info + num_dims; \
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = inp[i]; \
            out[i] = x / (TYPENAME(1.0) + EXPFN(-x)); \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            TYPENAME x = inp[strided_i]; \
            out[i] = x / (TYPENAME(1.0) + EXPFN(-x)); \
        } \
    } \
}

SILU_FWD(float, expf, silu_fwd_f32)
SILU_FWD(double, exp, silu_fwd_f64)

#if __CUDA_ARCH__ >= 530
extern "C" __global__ void silu_fwd_f16(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const __half *inp,
    __half *out
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            float x = __half2float(inp[i]);
            out[i] = __float2half(x / (1.0f + expf(-x)));
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            float x = __half2float(inp[strided_i]);
            out[i] = __float2half(x / (1.0f + expf(-x)));
        }
    }
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void silu_fwd_bf16(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const __nv_bfloat16 *inp,
    __nv_bfloat16 *out
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    if (info == nullptr || is_contiguous(num_dims, dims, strides)) {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            float x = __bfloat162float(inp[i]);
            out[i] = __float2bfloat16(x / (1.0f + expf(-x)));
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            float x = __bfloat162float(inp[strided_i]);
            out[i] = __float2bfloat16(x / (1.0f + expf(-x)));
        }
    }
}
#endif

// ─── Backward ───────────────────────────────────────────────────────────
// grad_input = grad_output * s * (1 + x * (1 - s))   where s = sigmoid(x)
// Saves original input x (not output), recomputes sigmoid inline.

#define SILU_BWD(TYPENAME, EXPFN, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const TYPENAME *grad_out, \
    const TYPENAME *inp, \
    TYPENAME *grad_in \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        TYPENAME x = inp[i]; \
        TYPENAME s = TYPENAME(1.0) / (TYPENAME(1.0) + EXPFN(-x)); \
        grad_in[i] = grad_out[i] * s * (TYPENAME(1.0) + x * (TYPENAME(1.0) - s)); \
    } \
}

SILU_BWD(float, expf, silu_bwd_f32)
SILU_BWD(double, exp, silu_bwd_f64)

#if __CUDA_ARCH__ >= 530
extern "C" __global__ void silu_bwd_f16(
    const size_t numel,
    const __half *grad_out,
    const __half *inp,
    __half *grad_in
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        float x = __half2float(inp[i]);
        float g = __half2float(grad_out[i]);
        float s = 1.0f / (1.0f + expf(-x));
        grad_in[i] = __float2half(g * s * (1.0f + x * (1.0f - s)));
    }
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void silu_bwd_bf16(
    const size_t numel,
    const __nv_bfloat16 *grad_out,
    const __nv_bfloat16 *inp,
    __nv_bfloat16 *grad_in
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        float x = __bfloat162float(inp[i]);
        float g = __bfloat162float(grad_out[i]);
        float s = 1.0f / (1.0f + expf(-x));
        grad_in[i] = __float2bfloat16(g * s * (1.0f + x * (1.0f - s)));
    }
}
#endif
