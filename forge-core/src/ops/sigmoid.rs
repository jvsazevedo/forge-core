//! Sigmoid with custom CUDA kernel.
//!
//! Forward:  sigmoid(x) = 1 / (1 + exp(-x))
//! Backward: grad * output * (1 - output)

use candle_core::backend::BackendStorage;
use candle_core::{CpuStorage, CudaStorage, CustomOp1, DType, Layout, Result, Shape, Tensor};

pub struct Sigmoid;

impl Sigmoid {
    /// Apply sigmoid to a tensor. Works on both CPU and CUDA.
    pub fn forward(x: &Tensor) -> Result<Tensor> {
        x.apply_op1(Sigmoid)
    }
}

// ─── CPU helpers ────────────────────────────────────────────────────────

fn sigmoid_f32(data: &[f32], layout: &Layout) -> Vec<f32> {
    let el = layout.shape().elem_count();
    let mut out = Vec::with_capacity(el);
    if layout.is_contiguous() {
        let offset = layout.start_offset();
        for i in 0..el {
            let x = data[offset + i];
            out.push(1.0f32 / (1.0 + (-x).exp()));
        }
    } else {
        // Non-contiguous: compute strided index manually
        let dims = layout.dims();
        let strides = layout.stride();
        let offset = layout.start_offset();
        for i in 0..el {
            let mut src_idx = offset;
            let mut remaining = i;
            for d in (0..dims.len()).rev() {
                src_idx += (remaining % dims[d]) * strides[d];
                remaining /= dims[d];
            }
            out.push(1.0f32 / (1.0 + (-data[src_idx]).exp()));
        }
    }
    out
}

fn sigmoid_f64(data: &[f64], layout: &Layout) -> Vec<f64> {
    let el = layout.shape().elem_count();
    let mut out = Vec::with_capacity(el);
    if layout.is_contiguous() {
        let offset = layout.start_offset();
        for i in 0..el {
            let x = data[offset + i];
            out.push(1.0f64 / (1.0 + (-x).exp()));
        }
    } else {
        let dims = layout.dims();
        let strides = layout.stride();
        let offset = layout.start_offset();
        for i in 0..el {
            let mut src_idx = offset;
            let mut remaining = i;
            for d in (0..dims.len()).rev() {
                src_idx += (remaining % dims[d]) * strides[d];
                remaining /= dims[d];
            }
            out.push(1.0f64 / (1.0 + (-data[src_idx]).exp()));
        }
    }
    out
}

// ─── CustomOp1 implementation ───────────────────────────────────────────

impl CustomOp1 for Sigmoid {
    fn name(&self) -> &'static str {
        "forge_sigmoid"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let shape = layout.shape().clone();
        let out = match storage {
            CpuStorage::F32(data) => CpuStorage::F32(sigmoid_f32(data, layout)),
            CpuStorage::F64(data) => CpuStorage::F64(sigmoid_f64(data, layout)),
            _ => {
                return Err(candle_core::Error::UnsupportedDTypeForOp(
                    storage.dtype(),
                    "sigmoid",
                ))
            }
        };
        Ok((out, shape))
    }

    fn cuda_fwd(&self, storage: &CudaStorage, layout: &Layout) -> Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
        use candle_core::cuda_backend::{CudaStorageSlice, SlicePtrOrNull, WrapErr};

        let device = storage.device().clone();
        let shape = layout.shape().clone();
        let el = shape.elem_count();

        let kernel_name = match storage.dtype() {
            DType::F32 => "sigmoid_fwd_f32",
            DType::F64 => "sigmoid_fwd_f64",
            DType::F16 => "sigmoid_fwd_f16",
            DType::BF16 => "sigmoid_fwd_bf16",
            dtype => return Err(candle_core::Error::UnsupportedDTypeForOp(dtype, "sigmoid")),
        };

        let func = device.get_or_load_custom_func(
            kernel_name,
            "forge_sigmoid",
            forge_kernels::SIGMOID.ptx(),
        )?;
        let cfg = LaunchConfig::for_num_elems(el as u32);
        let ds = SlicePtrOrNull::params_from_layout(&device, layout)?;

        let slice = match storage.dtype() {
            DType::F32 => {
                let src = storage.as_cuda_slice::<f32>()?;
                let src = src.slice(layout.start_offset()..);
                let out = device.alloc_zeros::<f32>(el)?;
                let mut builder = func.builder();
                builder.arg(&el);
                let num_dims = layout.dims().len();
                builder.arg(&num_dims);
                ds.builder_arg(&mut builder);
                builder.arg(&src);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F32(out)
            }
            DType::F64 => {
                let src = storage.as_cuda_slice::<f64>()?;
                let src = src.slice(layout.start_offset()..);
                let out = device.alloc_zeros::<f64>(el)?;
                let mut builder = func.builder();
                builder.arg(&el);
                let num_dims = layout.dims().len();
                builder.arg(&num_dims);
                ds.builder_arg(&mut builder);
                builder.arg(&src);
                builder.arg(&out);
                unsafe { builder.launch(cfg) }.w()?;
                CudaStorageSlice::F64(out)
            }
            dtype => return Err(candle_core::Error::UnsupportedDTypeForOp(dtype, "sigmoid")),
        };

        let storage = CudaStorage { slice, device };
        Ok((storage, shape))
    }

    fn bwd(&self, _arg: &Tensor, res: &Tensor, grad_res: &Tensor) -> Result<Option<Tensor>> {
        // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
        let one_minus = (1.0 - res)?;
        let grad = (grad_res * res)?.mul(&one_minus)?;
        Ok(Some(grad))
    }
}
