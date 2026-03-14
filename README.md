# Forge Core

Rust compute backend for the [Forge](https://github.com/jvsazevedo/forge) ML framework. Built on [candle](https://github.com/huggingface/candle) with custom CUDA kernels and C FFI for F# interop.

## What's inside

```
forge-core/       Rust library — candle ops, custom ops, FFI exports
forge-kernels/    CUDA kernels compiled to PTX at build time
forge-smoke/      Benchmark runner (WIP)
examples/mlp-poc/ MLP proof-of-concept
```

### Custom CUDA Kernels

Candle is missing CUDA kernels for several ops needed by transformer models (e.g. SmolLM2). We implement them in `forge-kernels/`:

| Kernel | Type | File |
|---|---|---|
| Sigmoid | element-wise | `sigmoid.cu` |
| SiLU (Swish) | element-wise | `silu.cu` |
| Softplus | element-wise | `softplus.cu` |
| RMS Norm | reduction (warp shuffle + shared mem) | `rms_norm.cu` |

All kernels support f32, f64, f16, and bf16. Each has a CPU fallback and autograd backward support.

## Prerequisites

- Rust (nightly or stable with edition 2024 support)
- CUDA toolkit (nvcc)
- [just](https://github.com/casey/just) (optional, for task runner)

## Build

```bash
just build          # or: cargo build
just build-release  # produces libforge_core.so
```

## Test

```bash
just test   # or: cargo test
just smoke  # smoke tests only (with output)
```

## Acknowledgments

We are grateful to the [furst_sharp](https://github.com/FulecoRafa/furst_sharp) project for the library that enabled us to perform interop with F#.

## License

[MIT](LICENSE)
