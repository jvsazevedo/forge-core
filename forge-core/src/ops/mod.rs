//! Custom ops with CUDA kernel implementations.
//!
//! These fill the gaps where candle doesn't have CUDA kernels:
//! sigmoid, silu, softplus, rms_norm.

pub mod sigmoid;
pub mod silu;
pub mod softplus;
pub mod rms_norm;
