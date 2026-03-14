// Forge softplus CUDA kernel.
//
// Forward:  softplus(x) = log1p(exp(x))   with threshold trick for large x
// Backward: grad * sigmoid(x)
//
// Reference: PyTorch uses threshold=20 — for x > 20, softplus(x) ≈ x

#include "cuda_utils.cuh"
#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define SOFTPLUS_THRESHOLD 20.0f

// ─── Forward ────────────────────────────────────────────────────────────

#define SOFTPLUS_FWD(TYPENAME, EXPFN, LOG1PFN, FN_NAME) \
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
            out[i] = (x > TYPENAME(SOFTPLUS_THRESHOLD)) ? x : LOG1PFN(EXPFN(x)); \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            TYPENAME x = inp[strided_i]; \
            out[i] = (x > TYPENAME(SOFTPLUS_THRESHOLD)) ? x : LOG1PFN(EXPFN(x)); \
        } \
    } \
}

SOFTPLUS_FWD(float, expf, log1pf, softplus_fwd_f32)
SOFTPLUS_FWD(double, exp, log1p, softplus_fwd_f64)

#if __CUDA_ARCH__ >= 530
extern "C" __global__ void softplus_fwd_f16(
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
            out[i] = __float2half((x > SOFTPLUS_THRESHOLD) ? x : log1pf(expf(x)));
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            float x = __half2float(inp[strided_i]);
            out[i] = __float2half((x > SOFTPLUS_THRESHOLD) ? x : log1pf(expf(x)));
        }
    }
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void softplus_fwd_bf16(
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
            out[i] = __float2bfloat16((x > SOFTPLUS_THRESHOLD) ? x : log1pf(expf(x)));
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            float x = __bfloat162float(inp[strided_i]);
            out[i] = __float2bfloat16((x > SOFTPLUS_THRESHOLD) ? x : log1pf(expf(x)));
        }
    }
}
#endif

// ─── Backward ───────────────────────────────────────────────────────────
// d/dx softplus(x) = sigmoid(x)
// For x > threshold: grad passes through (derivative ≈ 1)

#define SOFTPLUS_BWD(TYPENAME, EXPFN, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const TYPENAME *grad_out, \
    const TYPENAME *inp, \
    TYPENAME *grad_in \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        TYPENAME x = inp[i]; \
        if (x > TYPENAME(SOFTPLUS_THRESHOLD)) { \
            grad_in[i] = grad_out[i]; \
        } else { \
            TYPENAME s = TYPENAME(1.0) / (TYPENAME(1.0) + EXPFN(-x)); \
            grad_in[i] = grad_out[i] * s; \
        } \
    } \
}

SOFTPLUS_BWD(float, expf, softplus_bwd_f32)
SOFTPLUS_BWD(double, exp, softplus_bwd_f64)

#if __CUDA_ARCH__ >= 530
extern "C" __global__ void softplus_bwd_f16(
    const size_t numel,
    const __half *grad_out,
    const __half *inp,
    __half *grad_in
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        float x = __half2float(inp[i]);
        float g = __half2float(grad_out[i]);
        if (x > SOFTPLUS_THRESHOLD) {
            grad_in[i] = grad_out[i];
        } else {
            float s = 1.0f / (1.0f + expf(-x));
            grad_in[i] = __float2half(g * s);
        }
    }
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void softplus_bwd_bf16(
    const size_t numel,
    const __nv_bfloat16 *grad_out,
    const __nv_bfloat16 *inp,
    __nv_bfloat16 *grad_in
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        float x = __bfloat162float(inp[i]);
        float g = __bfloat162float(grad_out[i]);
        if (x > SOFTPLUS_THRESHOLD) {
            grad_in[i] = grad_out[i];
        } else {
            float s = 1.0f / (1.0f + expf(-x));
            grad_in[i] = __float2bfloat16(g * s);
        }
    }
}
#endif
