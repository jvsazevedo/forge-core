// Forge RMS Norm CUDA kernel.
//
// Forward:  y = x * rsqrt(mean(x²) + eps) * gamma
// Backward: dX per-row reduction, dgamma column-wise reduction
//
// Unlike element-wise ops (sigmoid, silu), this is a reduction kernel:
// 1 thread block per row, warp shuffle + shared memory for cross-warp combine.
//
// Reference: PyTorch aten/src/ATen/native/cuda/layer_norm_kernel.cu

#include <stdint.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ─── Forward ────────────────────────────────────────────────────────────
// Each block processes one row of the [M, N] input.
// blockDim.x threads cooperate to reduce sum(x²) across N elements.

#define RMS_NORM_FWD(TYPENAME, FN_NAME)                                        \
extern "C" __global__ void FN_NAME(                                            \
    const TYPENAME *X,                                                         \
    const TYPENAME *gamma,                                                     \
    TYPENAME *Y,                                                               \
    const size_t N,                                                            \
    const float eps                                                            \
) {                                                                            \
    const size_t row = blockIdx.x;                                             \
    const TYPENAME *x_row = X + row * N;                                       \
    TYPENAME *y_row = Y + row * N;                                             \
                                                                               \
    /* Step 1: each thread accumulates partial sum of x² */                    \
    float partial_sum = 0.0f;                                                  \
    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {                    \
        float val = (float)x_row[j];                                           \
        partial_sum += val * val;                                              \
    }                                                                          \
                                                                               \
    /* Step 2: warp-level reduction via shuffle */                             \
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {              \
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);      \
    }                                                                          \
                                                                               \
    /* Step 3: cross-warp reduction via shared memory */                       \
    __shared__ float shared[32]; /* max 32 warps per block (1024 threads) */   \
    int lane = threadIdx.x % warpSize;                                         \
    int warp_id = threadIdx.x / warpSize;                                      \
    if (lane == 0) shared[warp_id] = partial_sum;                              \
    __syncthreads();                                                           \
                                                                               \
    /* First warp reduces the per-warp sums */                                 \
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;                   \
    partial_sum = (threadIdx.x < (unsigned)num_warps) ? shared[threadIdx.x] : 0.0f; \
    if (warp_id == 0) {                                                        \
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {          \
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);  \
        }                                                                      \
    }                                                                          \
                                                                               \
    /* Step 4: broadcast rstd to all threads */                                \
    __shared__ float s_rstd;                                                   \
    if (threadIdx.x == 0) {                                                    \
        s_rstd = rsqrtf(partial_sum / (float)N + eps);                         \
    }                                                                          \
    __syncthreads();                                                           \
    float rstd = s_rstd;                                                       \
                                                                               \
    /* Step 5: normalize */                                                    \
    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {                    \
        y_row[j] = (TYPENAME)((float)x_row[j] * rstd * (float)gamma[j]);      \
    }                                                                          \
}

RMS_NORM_FWD(float, rms_norm_fwd_f32)
RMS_NORM_FWD(double, rms_norm_fwd_f64)

// f16: promote to float for accumulation
#if __CUDA_ARCH__ >= 530
extern "C" __global__ void rms_norm_fwd_f16(
    const __half *X,
    const __half *gamma,
    __half *Y,
    const size_t N,
    const float eps
) {
    const size_t row = blockIdx.x;
    const __half *x_row = X + row * N;
    __half *y_row = Y + row * N;

    float partial_sum = 0.0f;
    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
        float val = __half2float(x_row[j]);
        partial_sum += val * val;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared[warp_id] = partial_sum;
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    partial_sum = (threadIdx.x < (unsigned)num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(partial_sum / (float)N + eps);
    __syncthreads();
    float rstd = s_rstd;

    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
        float x = __half2float(x_row[j]);
        float g = __half2float(gamma[j]);
        y_row[j] = __float2half(x * rstd * g);
    }
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void rms_norm_fwd_bf16(
    const __nv_bfloat16 *X,
    const __nv_bfloat16 *gamma,
    __nv_bfloat16 *Y,
    const size_t N,
    const float eps
) {
    const size_t row = blockIdx.x;
    const __nv_bfloat16 *x_row = X + row * N;
    __nv_bfloat16 *y_row = Y + row * N;

    float partial_sum = 0.0f;
    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
        float val = __bfloat162float(x_row[j]);
        partial_sum += val * val;
    }

    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);

    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared[warp_id] = partial_sum;
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    partial_sum = (threadIdx.x < (unsigned)num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, offset);
    }

    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(partial_sum / (float)N + eps);
    __syncthreads();
    float rstd = s_rstd;

    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
        float x = __bfloat162float(x_row[j]);
        float g = __bfloat162float(gamma[j]);
        y_row[j] = __float2bfloat16(x * rstd * g);
    }
}
#endif

// ─── Backward dX ────────────────────────────────────────────────────────
// 1 block per row. Recomputes rstd from X (Option C — no saved state).
// dX[j] = rstd * (gamma[j] * dY[j] - X[j] * rstd * s / N)
// where s = sum_j(dY[j] * gamma[j] * X[j] * rstd)

#define RMS_NORM_BWD_INPUT(TYPENAME, FN_NAME)                                  \
extern "C" __global__ void FN_NAME(                                            \
    const TYPENAME *dY,                                                        \
    const TYPENAME *X,                                                         \
    const TYPENAME *gamma,                                                     \
    TYPENAME *dX,                                                              \
    const size_t N,                                                            \
    const float eps                                                            \
) {                                                                            \
    const size_t row = blockIdx.x;                                             \
    const TYPENAME *dy_row = dY + row * N;                                     \
    const TYPENAME *x_row = X + row * N;                                       \
    TYPENAME *dx_row = dX + row * N;                                           \
                                                                               \
    /* Recompute rstd: sum(x²) */                                              \
    float sum_x2 = 0.0f;                                                       \
    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {                    \
        float val = (float)x_row[j];                                           \
        sum_x2 += val * val;                                                   \
    }                                                                          \
    for (int off = warpSize/2; off > 0; off >>= 1)                            \
        sum_x2 += __shfl_down_sync(0xffffffff, sum_x2, off);                  \
    __shared__ float shared[32];                                               \
    int lane = threadIdx.x % warpSize;                                         \
    int warp_id = threadIdx.x / warpSize;                                      \
    if (lane == 0) shared[warp_id] = sum_x2;                                   \
    __syncthreads();                                                           \
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;                   \
    sum_x2 = (threadIdx.x < (unsigned)num_warps) ? shared[threadIdx.x] : 0.0f;\
    if (warp_id == 0) {                                                        \
        for (int off = warpSize/2; off > 0; off >>= 1)                        \
            sum_x2 += __shfl_down_sync(0xffffffff, sum_x2, off);              \
    }                                                                          \
    __shared__ float s_rstd;                                                   \
    if (threadIdx.x == 0) s_rstd = rsqrtf(sum_x2 / (float)N + eps);           \
    __syncthreads();                                                           \
    float rstd = s_rstd;                                                       \
                                                                               \
    /* Compute s = sum_j(dY[j] * gamma[j] * X[j]) * rstd */                   \
    float partial_s = 0.0f;                                                    \
    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {                    \
        partial_s += (float)dy_row[j] * (float)gamma[j] * (float)x_row[j];    \
    }                                                                          \
    partial_s *= rstd;                                                         \
    for (int off = warpSize/2; off > 0; off >>= 1)                            \
        partial_s += __shfl_down_sync(0xffffffff, partial_s, off);             \
    if (lane == 0) shared[warp_id] = partial_s;                                \
    __syncthreads();                                                           \
    partial_s = (threadIdx.x < (unsigned)num_warps) ? shared[threadIdx.x] : 0.0f; \
    if (warp_id == 0) {                                                        \
        for (int off = warpSize/2; off > 0; off >>= 1)                        \
            partial_s += __shfl_down_sync(0xffffffff, partial_s, off);         \
    }                                                                          \
    __shared__ float s_dot;                                                    \
    if (threadIdx.x == 0) s_dot = partial_s;                                   \
    __syncthreads();                                                           \
    float dot = s_dot;                                                         \
                                                                               \
    /* Compute dX */                                                           \
    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {                    \
        float dy = (float)dy_row[j];                                           \
        float x = (float)x_row[j];                                            \
        float g = (float)gamma[j];                                             \
        dx_row[j] = (TYPENAME)(rstd * (g * dy - x * rstd * dot / (float)N));  \
    }                                                                          \
}

RMS_NORM_BWD_INPUT(float, rms_norm_bwd_input_f32)
RMS_NORM_BWD_INPUT(double, rms_norm_bwd_input_f64)

#if __CUDA_ARCH__ >= 530
extern "C" __global__ void rms_norm_bwd_input_f16(
    const __half *dY, const __half *X, const __half *gamma,
    __half *dX, const size_t N, const float eps
) {
    const size_t row = blockIdx.x;
    const __half *dy_row = dY + row * N;
    const __half *x_row = X + row * N;
    __half *dx_row = dX + row * N;

    float sum_x2 = 0.0f;
    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
        float val = __half2float(x_row[j]);
        sum_x2 += val * val;
    }
    for (int off = warpSize/2; off > 0; off >>= 1)
        sum_x2 += __shfl_down_sync(0xffffffff, sum_x2, off);
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared[warp_id] = sum_x2;
    __syncthreads();
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    sum_x2 = (threadIdx.x < (unsigned)num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        for (int off = warpSize/2; off > 0; off >>= 1)
            sum_x2 += __shfl_down_sync(0xffffffff, sum_x2, off);
    }
    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(sum_x2 / (float)N + eps);
    __syncthreads();
    float rstd = s_rstd;

    float partial_s = 0.0f;
    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
        partial_s += __half2float(dy_row[j]) * __half2float(gamma[j]) * __half2float(x_row[j]);
    }
    partial_s *= rstd;
    for (int off = warpSize/2; off > 0; off >>= 1)
        partial_s += __shfl_down_sync(0xffffffff, partial_s, off);
    if (lane == 0) shared[warp_id] = partial_s;
    __syncthreads();
    partial_s = (threadIdx.x < (unsigned)num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        for (int off = warpSize/2; off > 0; off >>= 1)
            partial_s += __shfl_down_sync(0xffffffff, partial_s, off);
    }
    __shared__ float s_dot;
    if (threadIdx.x == 0) s_dot = partial_s;
    __syncthreads();
    float dot = s_dot;

    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
        float dy = __half2float(dy_row[j]);
        float x = __half2float(x_row[j]);
        float g = __half2float(gamma[j]);
        dx_row[j] = __float2half(rstd * (g * dy - x * rstd * dot / (float)N));
    }
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void rms_norm_bwd_input_bf16(
    const __nv_bfloat16 *dY, const __nv_bfloat16 *X, const __nv_bfloat16 *gamma,
    __nv_bfloat16 *dX, const size_t N, const float eps
) {
    const size_t row = blockIdx.x;
    const __nv_bfloat16 *dy_row = dY + row * N;
    const __nv_bfloat16 *x_row = X + row * N;
    __nv_bfloat16 *dx_row = dX + row * N;

    float sum_x2 = 0.0f;
    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
        float val = __bfloat162float(x_row[j]);
        sum_x2 += val * val;
    }
    for (int off = warpSize/2; off > 0; off >>= 1)
        sum_x2 += __shfl_down_sync(0xffffffff, sum_x2, off);
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared[warp_id] = sum_x2;
    __syncthreads();
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    sum_x2 = (threadIdx.x < (unsigned)num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        for (int off = warpSize/2; off > 0; off >>= 1)
            sum_x2 += __shfl_down_sync(0xffffffff, sum_x2, off);
    }
    __shared__ float s_rstd;
    if (threadIdx.x == 0) s_rstd = rsqrtf(sum_x2 / (float)N + eps);
    __syncthreads();
    float rstd = s_rstd;

    float partial_s = 0.0f;
    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
        partial_s += __bfloat162float(dy_row[j]) * __bfloat162float(gamma[j]) * __bfloat162float(x_row[j]);
    }
    partial_s *= rstd;
    for (int off = warpSize/2; off > 0; off >>= 1)
        partial_s += __shfl_down_sync(0xffffffff, partial_s, off);
    if (lane == 0) shared[warp_id] = partial_s;
    __syncthreads();
    partial_s = (threadIdx.x < (unsigned)num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        for (int off = warpSize/2; off > 0; off >>= 1)
            partial_s += __shfl_down_sync(0xffffffff, partial_s, off);
    }
    __shared__ float s_dot;
    if (threadIdx.x == 0) s_dot = partial_s;
    __syncthreads();
    float dot = s_dot;

    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
        float dy = __bfloat162float(dy_row[j]);
        float x = __bfloat162float(x_row[j]);
        float g = __bfloat162float(gamma[j]);
        dx_row[j] = __float2bfloat16(rstd * (g * dy - x * rstd * dot / (float)N));
    }
}
#endif

// ─── Backward dgamma ───────────────────────────────────────────────────
// Column-wise reduction: dgamma[j] = sum_i(dY[i,j] * X[i,j] * rstd[i])
// Each block handles a chunk of columns. Grid: ceil(N / blockDim.x) blocks.
// Each thread loops over all M rows for its column(s).

#define RMS_NORM_BWD_WEIGHT(TYPENAME, FN_NAME)                                 \
extern "C" __global__ void FN_NAME(                                            \
    const TYPENAME *dY,                                                        \
    const TYPENAME *X,                                                         \
    const float *rstd,                                                         \
    TYPENAME *dgamma,                                                          \
    const size_t M,                                                            \
    const size_t N                                                             \
) {                                                                            \
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;                    \
    if (j >= N) return;                                                        \
                                                                               \
    float acc = 0.0f;                                                          \
    for (size_t i = 0; i < M; i++) {                                           \
        acc += (float)dY[i * N + j] * (float)X[i * N + j] * rstd[i];          \
    }                                                                          \
    dgamma[j] = (TYPENAME)acc;                                                 \
}

RMS_NORM_BWD_WEIGHT(float, rms_norm_bwd_weight_f32)
RMS_NORM_BWD_WEIGHT(double, rms_norm_bwd_weight_f64)

#if __CUDA_ARCH__ >= 530
extern "C" __global__ void rms_norm_bwd_weight_f16(
    const __half *dY, const __half *X, const float *rstd,
    __half *dgamma, const size_t M, const size_t N
) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float acc = 0.0f;
    for (size_t i = 0; i < M; i++) {
        acc += __half2float(dY[i * N + j]) * __half2float(X[i * N + j]) * rstd[i];
    }
    dgamma[j] = __float2half(acc);
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void rms_norm_bwd_weight_bf16(
    const __nv_bfloat16 *dY, const __nv_bfloat16 *X, const float *rstd,
    __nv_bfloat16 *dgamma, const size_t M, const size_t N
) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float acc = 0.0f;
    for (size_t i = 0; i < M; i++) {
        acc += __bfloat162float(dY[i * N + j]) * __bfloat162float(X[i * N + j]) * rstd[i];
    }
    dgamma[j] = __float2bfloat16(acc);
}
#endif

// ─── Helper: compute rstd per row (for backward dgamma) ────────────────
// Separate kernel so backward can recompute rstd into a [M] buffer for dgamma.

#define RMS_NORM_RSTD(TYPENAME, FN_NAME)                                       \
extern "C" __global__ void FN_NAME(                                            \
    const TYPENAME *X,                                                         \
    float *rstd_out,                                                           \
    const size_t N,                                                            \
    const float eps                                                            \
) {                                                                            \
    const size_t row = blockIdx.x;                                             \
    const TYPENAME *x_row = X + row * N;                                       \
                                                                               \
    float partial_sum = 0.0f;                                                  \
    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {                    \
        float val = (float)x_row[j];                                           \
        partial_sum += val * val;                                              \
    }                                                                          \
    for (int off = warpSize/2; off > 0; off >>= 1)                            \
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, off);         \
    __shared__ float shared[32];                                               \
    int lane = threadIdx.x % warpSize;                                         \
    int warp_id = threadIdx.x / warpSize;                                      \
    if (lane == 0) shared[warp_id] = partial_sum;                              \
    __syncthreads();                                                           \
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;                   \
    partial_sum = (threadIdx.x < (unsigned)num_warps) ? shared[threadIdx.x] : 0.0f; \
    if (warp_id == 0) {                                                        \
        for (int off = warpSize/2; off > 0; off >>= 1)                        \
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, off);     \
    }                                                                          \
    if (threadIdx.x == 0) {                                                    \
        rstd_out[row] = rsqrtf(partial_sum / (float)N + eps);                  \
    }                                                                          \
}

RMS_NORM_RSTD(float, rms_norm_rstd_f32)
RMS_NORM_RSTD(double, rms_norm_rstd_f64)

#if __CUDA_ARCH__ >= 530
extern "C" __global__ void rms_norm_rstd_f16(
    const __half *X, float *rstd_out, const size_t N, const float eps
) {
    const size_t row = blockIdx.x;
    const __half *x_row = X + row * N;
    float partial_sum = 0.0f;
    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
        float val = __half2float(x_row[j]);
        partial_sum += val * val;
    }
    for (int off = warpSize/2; off > 0; off >>= 1)
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, off);
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared[warp_id] = partial_sum;
    __syncthreads();
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    partial_sum = (threadIdx.x < (unsigned)num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        for (int off = warpSize/2; off > 0; off >>= 1)
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, off);
    }
    if (threadIdx.x == 0) rstd_out[row] = rsqrtf(partial_sum / (float)N + eps);
}
#endif

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void rms_norm_rstd_bf16(
    const __nv_bfloat16 *X, float *rstd_out, const size_t N, const float eps
) {
    const size_t row = blockIdx.x;
    const __nv_bfloat16 *x_row = X + row * N;
    float partial_sum = 0.0f;
    for (size_t j = threadIdx.x; j < N; j += blockDim.x) {
        float val = __bfloat162float(x_row[j]);
        partial_sum += val * val;
    }
    for (int off = warpSize/2; off > 0; off >>= 1)
        partial_sum += __shfl_down_sync(0xffffffff, partial_sum, off);
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared[warp_id] = partial_sum;
    __syncthreads();
    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    partial_sum = (threadIdx.x < (unsigned)num_warps) ? shared[threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        for (int off = warpSize/2; off > 0; off >>= 1)
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, off);
    }
    if (threadIdx.x == 0) rstd_out[row] = rsqrtf(partial_sum / (float)N + eps);
}
#endif
