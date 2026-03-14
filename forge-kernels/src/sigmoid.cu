// Forge sigmoid CUDA kernel.
//
// Forward:  sigmoid(x) = 1 / (1 + exp(-x))
// Backward: grad * output * (1 - output)   (reuses forward result)
//
// Reference: PyTorch aten/src/ATen/native/cuda/UnarySpecialOpsKernel.cu

#include "cuda_utils.cuh"
#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ─── Forward ────────────────────────────────────────────────────────────

// Generic forward: works for f32/f64 where arithmetic operators are unambiguous
#define SIGMOID_FWD(TYPENAME, EXPFN, FN_NAME) \
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
            out[i] = TYPENAME(1.0) / (TYPENAME(1.0) + EXPFN(-x)); \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides); \
            TYPENAME x = inp[strided_i]; \
            out[i] = TYPENAME(1.0) / (TYPENAME(1.0) + EXPFN(-x)); \
        } \
    } \
}

SIGMOID_FWD(float, expf, sigmoid_fwd_f32)
SIGMOID_FWD(double, exp, sigmoid_fwd_f64)

// f16/bf16: promote to float for computation (matches PyTorch's opmath_t pattern)
#if __CUDA_ARCH__ >= 530
extern "C" __global__ void sigmoid_fwd_f16(
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
            out[i] = __float2half(1.0f / (1.0f + expf(-x)));
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            float x = __half2float(inp[strided_i]);
            out[i] = __float2half(1.0f / (1.0f + expf(-x)));
        }
    }
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void sigmoid_fwd_bf16(
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
            out[i] = __float2bfloat16(1.0f / (1.0f + expf(-x)));
        }
    } else {
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
            unsigned strided_i = get_strided_index(i, num_dims, dims, strides);
            float x = __bfloat162float(inp[strided_i]);
            out[i] = __float2bfloat16(1.0f / (1.0f + expf(-x)));
        }
    }
}
#endif

// ─── Backward ───────────────────────────────────────────────────────────
// grad_input = grad_output * sigmoid_output * (1 - sigmoid_output)
// Backward tensors are always contiguous (candle guarantees this for grads).

#define SIGMOID_BWD(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const TYPENAME *grad_out, \
    const TYPENAME *sigmoid_out, \
    TYPENAME *grad_in \
) { \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        TYPENAME s = sigmoid_out[i]; \
        grad_in[i] = grad_out[i] * s * (TYPENAME(1.0) - s); \
    } \
}

SIGMOID_BWD(float, sigmoid_bwd_f32)
SIGMOID_BWD(double, sigmoid_bwd_f64)

#if __CUDA_ARCH__ >= 530
extern "C" __global__ void sigmoid_bwd_f16(
    const size_t numel,
    const __half *grad_out,
    const __half *sigmoid_out,
    __half *grad_in
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        float s = __half2float(sigmoid_out[i]);
        float g = __half2float(grad_out[i]);
        grad_in[i] = __float2half(g * s * (1.0f - s));
    }
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void sigmoid_bwd_bf16(
    const size_t numel,
    const __nv_bfloat16 *grad_out,
    const __nv_bfloat16 *sigmoid_out,
    __nv_bfloat16 *grad_in
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        float s = __bfloat162float(sigmoid_out[i]);
        float g = __bfloat162float(grad_out[i]);
        grad_in[i] = __float2bfloat16(g * s * (1.0f - s));
    }
}
#endif
