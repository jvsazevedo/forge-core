//! Forge CUDA kernels — compiled to PTX at build time.
//!
//! Each kernel module provides a `Module` with embedded PTX that can be loaded
//! by candle's `CudaDevice::get_or_load_func()`.

mod ptx {
    include!(concat!(env!("OUT_DIR"), "/ptx.rs"));
}

pub struct Module {
    ptx: &'static str,
}

impl Module {
    pub fn ptx(&self) -> &'static str {
        self.ptx
    }
}

pub const SIGMOID: Module = Module { ptx: ptx::SIGMOID };
pub const SILU: Module = Module { ptx: ptx::SILU };
pub const SOFTPLUS: Module = Module { ptx: ptx::SOFTPLUS };
pub const RMS_NORM: Module = Module { ptx: ptx::RMS_NORM };
