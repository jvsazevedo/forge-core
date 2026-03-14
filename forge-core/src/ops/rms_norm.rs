//! RMS Norm with custom CUDA kernel.
//!
//! Forward:  y = x * rsqrt(mean(x²) + eps) * gamma
//! Backward: recomputes rstd (Option C — no saved state, thread-safe)

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp2, DType, Layout, Result, Shape, Tensor};

pub struct RmsNorm {
    eps: f64,
}

impl RmsNorm {
    /// Apply RMS Norm. `x` is [M, N], `gamma` is [N].
    pub fn forward(x: &Tensor, gamma: &Tensor, eps: f64) -> Result<Tensor> {
        x.apply_op2(gamma, RmsNorm { eps })
    }
}

// ─── CPU fallback ──────────────────────────────────────────────────────

fn rms_norm_cpu_f32(
    x_data: &[f32],
    x_layout: &Layout,
    g_data: &[f32],
    g_layout: &Layout,
    eps: f64,
) -> Vec<f32> {
    let dims = x_layout.dims();
    let m = dims[..dims.len() - 1].iter().product::<usize>();
    let n = *dims.last().unwrap();
    let x_off = x_layout.start_offset();
    let g_off = g_layout.start_offset();
    let mut out = Vec::with_capacity(m * n);

    if x_layout.is_contiguous() {
        for i in 0..m {
            let row = &x_data[x_off + i * n..x_off + (i + 1) * n];
            let sum_sq: f32 = row.iter().map(|v| v * v).sum();
            let rstd = 1.0 / (sum_sq / n as f32 + eps as f32).sqrt();
            for j in 0..n {
                out.push(row[j] * rstd * g_data[g_off + j]);
            }
        }
    } else {
        let x_dims = x_layout.dims();
        let x_strides = x_layout.stride();
        for i in 0..m {
            // Compute sum of squares for this row
            let mut sum_sq: f32 = 0.0;
            for j in 0..n {
                let flat_idx = i * n + j;
                let mut src_idx = x_off;
                let mut remaining = flat_idx;
                for d in (0..x_dims.len()).rev() {
                    src_idx += (remaining % x_dims[d]) * x_strides[d];
                    remaining /= x_dims[d];
                }
                let x = x_data[src_idx];
                sum_sq += x * x;
            }
            let rstd = 1.0 / (sum_sq / n as f32 + eps as f32).sqrt();
            for j in 0..n {
                let flat_idx = i * n + j;
                let mut src_idx = x_off;
                let mut remaining = flat_idx;
                for d in (0..x_dims.len()).rev() {
                    src_idx += (remaining % x_dims[d]) * x_strides[d];
                    remaining /= x_dims[d];
                }
                out.push(x_data[src_idx] * rstd * g_data[g_off + j]);
            }
        }
    }
    out
}

fn rms_norm_cpu_f64(
    x_data: &[f64],
    x_layout: &Layout,
    g_data: &[f64],
    g_layout: &Layout,
    eps: f64,
) -> Vec<f64> {
    let dims = x_layout.dims();
    let m = dims[..dims.len() - 1].iter().product::<usize>();
    let n = *dims.last().unwrap();
    let x_off = x_layout.start_offset();
    let g_off = g_layout.start_offset();
    let mut out = Vec::with_capacity(m * n);

    if x_layout.is_contiguous() {
        for i in 0..m {
            let row = &x_data[x_off + i * n..x_off + (i + 1) * n];
            let sum_sq: f64 = row.iter().map(|v| v * v).sum();
            let rstd = 1.0 / (sum_sq / n as f64 + eps).sqrt();
            for j in 0..n {
                out.push(row[j] * rstd * g_data[g_off + j]);
            }
        }
    } else {
        let x_dims = x_layout.dims();
        let x_strides = x_layout.stride();
        for i in 0..m {
            let mut sum_sq: f64 = 0.0;
            for j in 0..n {
                let flat_idx = i * n + j;
                let mut src_idx = x_off;
                let mut remaining = flat_idx;
                for d in (0..x_dims.len()).rev() {
                    src_idx += (remaining % x_dims[d]) * x_strides[d];
                    remaining /= x_dims[d];
                }
                let x = x_data[src_idx];
                sum_sq += x * x;
            }
            let rstd = 1.0 / (sum_sq / n as f64 + eps).sqrt();
            for j in 0..n {
                let flat_idx = i * n + j;
                let mut src_idx = x_off;
                let mut remaining = flat_idx;
                for d in (0..x_dims.len()).rev() {
                    src_idx += (remaining % x_dims[d]) * x_strides[d];
                    remaining /= x_dims[d];
                }
                out.push(x_data[src_idx] * rstd * g_data[g_off + j]);
            }
        }
    }
    out
}

// ─── CustomOp2 ─────────────────────────────────────────────────────────

impl CustomOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "forge_rms_norm"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let shape = l1.shape().clone();
        let out = match (s1, s2) {
            (CpuStorage::F32(x), CpuStorage::F32(g)) => {
                CpuStorage::F32(rms_norm_cpu_f32(x, l1, g, l2, self.eps))
            }
            (CpuStorage::F64(x), CpuStorage::F64(g)) => {
                CpuStorage::F64(rms_norm_cpu_f64(x, l1, g, l2, self.eps))
            }
            _ => {
                return Err(candle_core::Error::UnsupportedDTypeForOp(
                    s1.dtype(),
                    "rms_norm",
                ))
            }
        };
        Ok((out, shape))
    }

    fn cuda_fwd(
        &self,
        s1: &CudaStorage,
        l1: &Layout,
        s2: &CudaStorage,
        l2: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, WrapErr};

        let device = s1.device().clone();
        let shape = l1.shape().clone();
        let dims = l1.dims();
        let n = *dims.last().unwrap();
        let m = dims[..dims.len() - 1].iter().product::<usize>();

        let kernel_name = match s1.dtype() {
            DType::F32 => "rms_norm_fwd_f32",
            DType::F64 => "rms_norm_fwd_f64",
            DType::F16 => "rms_norm_fwd_f16",
            DType::BF16 => "rms_norm_fwd_bf16",
            dtype => return Err(candle_core::Error::UnsupportedDTypeForOp(dtype, "rms_norm")),
        };

        let func = device.get_or_load_custom_func(
            kernel_name,
            "forge_rms_norm",
            forge_kernels::RMS_NORM.ptx(),
        )?;

        // 1 block per row, 256 threads per block
        let threads = 256u32.min(n as u32);
        let cfg = LaunchConfig {
            grid_dim: (m as u32, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let eps = self.eps as f32;

        let slice = match s1.dtype() {
            DType::F32 => {
                let x = s1.as_cuda_slice::<f32>()?;
                let x = x.slice(l1.start_offset()..);
                let g = s2.as_cuda_slice::<f32>()?;
                let g = g.slice(l2.start_offset()..);
                let out = device.alloc_zeros::<f32>(m * n)?;
                let mut builder = func.builder();
                builder.arg(&x);
                builder.arg(&g);
                builder.arg(&out);
                builder.arg(&n);
                builder.arg(&eps);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F32(out)
            }
            DType::F64 => {
                let x = s1.as_cuda_slice::<f64>()?;
                let x = x.slice(l1.start_offset()..);
                let g = s2.as_cuda_slice::<f64>()?;
                let g = g.slice(l2.start_offset()..);
                let out = device.alloc_zeros::<f64>(m * n)?;
                let mut builder = func.builder();
                builder.arg(&x);
                builder.arg(&g);
                builder.arg(&out);
                builder.arg(&n);
                builder.arg(&eps);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F64(out)
            }
            dtype => return Err(candle_core::Error::UnsupportedDTypeForOp(dtype, "rms_norm")),
        };

        let storage = CudaStorage { slice, device };
        Ok((storage, shape))
    }

    fn bwd(
        &self,
        arg1: &Tensor,
        arg2: &Tensor,
        _res: &Tensor,
        grad_res: &Tensor,
    ) -> Result<(Option<Tensor>, Option<Tensor>)> {
        // arg1 = X [M, N], arg2 = gamma [N], grad_res = dY [M, N]
        // Recompute rstd per row from X
        let dims = arg1.dims();
        let n = *dims.last().unwrap();
        let eps = self.eps;

        // rstd = 1/sqrt(mean(x²) + eps) — per row
        let x_sq = arg1.sqr()?;
        let mean_x_sq = x_sq.mean_keepdim(dims.len() - 1)?; // [M, 1]
        let variance_plus_eps = (mean_x_sq + eps)?;
        let rstd = variance_plus_eps.sqrt()?.recip()?; // [M, 1]

        // dX: rstd * (gamma * dY - X * rstd * dot / N)
        // where dot = sum_j(dY[j] * gamma[j] * X[j]) * rstd per row
        let gamma_dy = grad_res.broadcast_mul(arg2)?; // [M, N]
        let x_gamma_dy = (arg1 * &gamma_dy)?; // [M, N]
        let dot = x_gamma_dy
            .sum_keepdim(dims.len() - 1)?
            .broadcast_mul(&rstd)?; // [M, 1]
        let x_rstd = arg1.broadcast_mul(&rstd)?;
        let correction = x_rstd.broadcast_mul(&dot)? / n as f64;
        let dx = rstd.broadcast_mul(&(&gamma_dy - correction?)?)?;

        // dgamma: sum_i(dY[i,j] * X[i,j] * rstd[i])
        let x_norm = arg1.broadcast_mul(&rstd)?; // [M, N]
        let dgamma_full = (grad_res * &x_norm)?; // [M, N]
        // Sum over all dims except last
        let mut dgamma = dgamma_full;
        for d in (0..dims.len() - 1).rev() {
            dgamma = dgamma.sum(d)?;
        }

        Ok((Some(dx), Some(dgamma)))
    }
}
