//! Forge Core — Rust backend for the Forge ML framework.
//!
//! Wraps HuggingFace's candle for tensor ops, autograd, and CUDA,
//! and exposes everything via C FFI for consumption by F#.

pub mod error;
pub mod ops;
pub mod tensor;

#[cfg(test)]
mod smoke_tests {
    use candle_core::{DType, Device, IndexOp, Module, Tensor, Var};
    use candle_nn::{AdamW, Optimizer, ParamsAdamW};

    fn get_device() -> Device {
        Device::cuda_if_available(0).unwrap()
    }

    // ── 1. CUDA availability ────────────────────────────────────────────

    #[test]
    fn test_cuda_available() {
        let dev = get_device();
        match &dev {
            Device::Cpu => println!("SMOKE: running on CPU"),
            Device::Cuda(_) => println!("SMOKE: running on CUDA"),
            _ => println!("SMOKE: running on other device"),
        }
    }

    // ── 2. Tensor creation on device ────────────────────────────────────

    #[test]
    fn test_tensor_on_device() {
        let dev = get_device();
        let t = Tensor::zeros((2, 3), DType::F32, &dev).unwrap();
        assert_eq!(t.dims(), &[2, 3]);
    }

    // ── 3. Matmul ───────────────────────────────────────────────────────

    #[test]
    fn test_matmul() {
        let dev = get_device();
        let a = Tensor::from_slice(
            &[1f32, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            (4, 3),
            &dev,
        ).unwrap();
        let b = Tensor::from_slice(
            &[1f32, 2., 3., 4., 5., 6.],
            (3, 2),
            &dev,
        ).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.dims(), &[4, 2]);
        let vals: Vec<f32> = c.flatten_all().unwrap().to_vec1().unwrap();
        // Row 0: 1*1+2*3+3*5=22, 1*2+2*4+3*6=28
        assert_eq!(vals[0], 22.0);
        assert_eq!(vals[1], 28.0);
    }

    // ── 4. Softmax ──────────────────────────────────────────────────────

    #[test]
    fn test_softmax() {
        let dev = get_device();
        let t = Tensor::from_slice(&[1f32, 2., 3.], (1, 3), &dev).unwrap();
        let sm = candle_nn::ops::softmax(&t, 1).unwrap();
        let vals: Vec<f32> = sm.flatten_all().unwrap().to_vec1().unwrap();
        let sum: f32 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax sum={}, expected 1.0", sum);
        assert!(vals[0] < vals[1] && vals[1] < vals[2]);
    }

    // ── 5. Sigmoid (custom CUDA kernel) ───────────────────────────────

    #[test]
    fn test_sigmoid_cpu() {
        let t = Tensor::from_slice(&[0f32, -100., 100.], 3, &Device::Cpu).unwrap();
        let s = crate::ops::sigmoid::Sigmoid::forward(&t).unwrap();
        let vals: Vec<f32> = s.to_vec1().unwrap();
        assert!((vals[0] - 0.5).abs() < 1e-5, "sigmoid(0)={}", vals[0]);
        assert!(vals[1] < 1e-5, "sigmoid(-100)={}", vals[1]);
        assert!((vals[2] - 1.0).abs() < 1e-5, "sigmoid(100)={}", vals[2]);
    }

    #[test]
    fn test_sigmoid_cuda() {
        let dev = get_device();
        let t = Tensor::from_slice(&[0f32, -100., 100.], 3, &dev).unwrap();
        let s = crate::ops::sigmoid::Sigmoid::forward(&t).unwrap();
        let vals: Vec<f32> = s.to_vec1().unwrap();
        assert!((vals[0] - 0.5).abs() < 1e-5, "sigmoid(0)={}", vals[0]);
        assert!(vals[1] < 1e-5, "sigmoid(-100)={}", vals[1]);
        assert!((vals[2] - 1.0).abs() < 1e-5, "sigmoid(100)={}", vals[2]);
    }

    // ── 6. SiLU (custom CUDA kernel) ──────────────────────────────────

    #[test]
    fn test_silu_cpu() {
        let t = Tensor::from_slice(&[0f32, 1., -1.], 3, &Device::Cpu).unwrap();
        let silu = crate::ops::silu::Silu::forward(&t).unwrap();
        let sig = crate::ops::sigmoid::Sigmoid::forward(&t).unwrap();
        let expected = t.mul(&sig).unwrap();
        let diff = silu.sub(&expected).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(0).unwrap().to_scalar().unwrap();
        assert!(max_diff < 1e-5, "silu vs x*sigmoid(x) max_diff={}", max_diff);
    }

    #[test]
    fn test_silu_cuda() {
        let dev = get_device();
        let t = Tensor::from_slice(&[0f32, 1., -1., 5., -5.], 5, &dev).unwrap();
        let silu = crate::ops::silu::Silu::forward(&t).unwrap();
        let vals: Vec<f32> = silu.to_vec1().unwrap();
        // silu(0) = 0, silu(1) ≈ 0.7311, silu(-1) ≈ -0.2689
        assert!(vals[0].abs() < 1e-5, "silu(0)={}", vals[0]);
        assert!((vals[1] - 0.7311).abs() < 1e-3, "silu(1)={}", vals[1]);
        assert!((vals[2] - (-0.2689)).abs() < 1e-3, "silu(-1)={}", vals[2]);
    }

    // ── 7. Clamp ────────────────────────────────────────────────────────

    #[test]
    fn test_clamp() {
        let dev = get_device();
        let t = Tensor::from_slice(&[-5f32, 0., 3., 10.], 4, &dev).unwrap();
        let clamped = t.clamp(-1f32, 5f32).unwrap();
        let vals: Vec<f32> = clamped.to_vec1().unwrap();
        assert_eq!(vals, vec![-1.0, 0.0, 3.0, 5.0]);
    }

    // ── 8. Cat + Reshape ────────────────────────────────────────────────

    #[test]
    fn test_cat_reshape() {
        let dev = get_device();
        let a = Tensor::from_slice(&[1f32, 2., 3.], (1, 3), &dev).unwrap();
        let b = Tensor::from_slice(&[4f32, 5., 6.], (1, 3), &dev).unwrap();
        let cat = Tensor::cat(&[&a, &b], 0).unwrap();
        assert_eq!(cat.dims(), &[2, 3]);
        let reshaped = cat.reshape((3, 2)).unwrap();
        assert_eq!(reshaped.dims(), &[3, 2]);
        let vals: Vec<f32> = reshaped.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals, vec![1., 2., 3., 4., 5., 6.]);
    }

    // ── 9. Softplus (custom CUDA kernel) ─────────────────────────────

    #[test]
    fn test_softplus_cpu() {
        let t = Tensor::from_slice(&[-2f32, 0., 1., 5., 25.], 5, &Device::Cpu).unwrap();
        let sp = crate::ops::softplus::Softplus::forward(&t).unwrap();
        let vals: Vec<f32> = sp.to_vec1().unwrap();
        let expected = [-2f32, 0., 1., 5.].map(|x| (1.0 + x.exp()).ln());
        for (got, exp) in vals.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-4, "softplus: got={}, expected={}", got, exp);
        }
        // x=25 > threshold(20) → returns x directly
        assert!((vals[4] - 25.0).abs() < 1e-4, "softplus(25)={}, expected 25", vals[4]);
    }

    #[test]
    fn test_softplus_cuda() {
        let dev = get_device();
        let t = Tensor::from_slice(&[-2f32, 0., 1., 5., 25.], 5, &dev).unwrap();
        let sp = crate::ops::softplus::Softplus::forward(&t).unwrap();
        let vals: Vec<f32> = sp.to_vec1().unwrap();
        let expected = [-2f32, 0., 1., 5.].map(|x| (1.0 + x.exp()).ln());
        for (got, exp) in vals.iter().zip(expected.iter()) {
            assert!((got - exp).abs() < 1e-4, "softplus: got={}, expected={}", got, exp);
        }
        assert!((vals[4] - 25.0).abs() < 1e-4, "softplus(25)={}, expected 25", vals[4]);
    }

    // ── 10. Autograd basic ──────────────────────────────────────────────

    #[test]
    fn test_autograd_basic() {
        // f(x) = x^2, df/dx = 2x. At x=3, grad should be 6.
        let x = Var::from_tensor(&Tensor::from_slice(&[3f32], 1, &Device::Cpu).unwrap()).unwrap();
        let loss = x.as_tensor().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let grad = grads.get(x.as_tensor()).unwrap();
        let vals: Vec<f32> = grad.to_vec1().unwrap();
        assert!((vals[0] - 6.0).abs() < 1e-5, "grad={}, expected 6.0", vals[0]);
    }

    // ── 11. Autograd MLP ────────────────────────────────────────────────

    #[test]
    fn test_autograd_mlp() {
        let dev = Device::Cpu;
        let w1 = Var::from_tensor(
            &Tensor::randn(0f32, 1., (2, 4), &dev).unwrap()
        ).unwrap();
        let w2 = Var::from_tensor(
            &Tensor::randn(0f32, 1., (4, 1), &dev).unwrap()
        ).unwrap();

        let input = Tensor::from_slice(&[1f32, 2.], (1, 2), &dev).unwrap();
        let target = Tensor::from_slice(&[0.5f32], (1, 1), &dev).unwrap();

        // Forward
        let h = input.matmul(w1.as_tensor()).unwrap();
        let h = candle_nn::Activation::Silu.forward(&h).unwrap();
        let out = h.matmul(w2.as_tensor()).unwrap();

        // MSE loss
        let diff = out.sub(&target).unwrap();
        let loss = diff.sqr().unwrap().mean_all().unwrap();

        // Backward
        let grads = loss.backward().unwrap();
        let g1 = grads.get(w1.as_tensor()).unwrap();
        let g2 = grads.get(w2.as_tensor()).unwrap();

        let g1_norm: f32 = g1.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
        let g2_norm: f32 = g2.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(g1_norm > 0.0, "w1 grad is zero");
        assert!(g2_norm > 0.0, "w2 grad is zero");
    }

    // ── 12. AdamW step ──────────────────────────────────────────────────

    #[test]
    fn test_adamw_step() {
        let dev = Device::Cpu;
        let w = Var::from_tensor(
            &Tensor::from_slice(&[1f32, 2., 3., 4.], (2, 2), &dev).unwrap()
        ).unwrap();

        let before: Vec<f32> = w.as_tensor().flatten_all().unwrap().to_vec1().unwrap();

        let mut opt = AdamW::new(
            vec![w.clone()],
            ParamsAdamW { lr: 0.01, ..Default::default() },
        ).unwrap();

        // Forward + backward + step
        let loss = w.as_tensor().sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();
        opt.step(&grads).unwrap();

        let after: Vec<f32> = w.as_tensor().flatten_all().unwrap().to_vec1().unwrap();
        assert_ne!(before, after, "AdamW step should change parameters");
    }

    // ── 13. Cross entropy ───────────────────────────────────────────────

    #[test]
    fn test_cross_entropy() {
        let dev = get_device();
        let logits = Tensor::from_slice(
            &[2f32, 1., 0.1, 0.1, 1., 2.],
            (2, 3),
            &dev,
        ).unwrap();
        let targets = Tensor::from_slice(&[0u32, 2], 2, &dev).unwrap();
        let loss = candle_nn::loss::cross_entropy(&logits, &targets).unwrap();
        let val: f32 = loss.to_scalar().unwrap();
        assert!(val > 0.0 && val.is_finite(), "cross_entropy={}", val);
    }

    // ── 14. RMSNorm (candle reference, CPU only) ──────────────────────
    // NOTE: candle has no CUDA kernel for rms-norm — runs on CPU only.

    #[test]
    fn test_rms_norm_candle_ref() {
        let dev = Device::Cpu;
        let input = Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6.], (2, 3), &dev).unwrap();
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &dev);
        let rms = candle_nn::rms_norm(3, 1e-5, vb.pp("rms")).unwrap();
        let out = rms.forward(&input).unwrap();
        assert_eq!(out.dims(), input.dims());
        let row0: Vec<f32> = out.i(0).unwrap().to_vec1().unwrap();
        assert_eq!(row0.len(), 3);
    }

    // ── 15. Forge RMS Norm (custom CUDA kernel) ─────────────────────

    #[test]
    fn test_forge_rms_norm_cpu() {
        let dev = Device::Cpu;
        let x = Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6.], (2, 3), &dev).unwrap();
        let gamma = Tensor::ones(3, DType::F32, &dev).unwrap();
        let out = crate::ops::rms_norm::RmsNorm::forward(&x, &gamma, 1e-5).unwrap();
        assert_eq!(out.dims(), &[2, 3]);

        // Manual check row 0: x=[1,2,3], rms=sqrt((1+4+9)/3)=sqrt(14/3)
        // rstd = 1/sqrt(14/3 + 1e-5)
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let rms0 = (14.0f32 / 3.0 + 1e-5).sqrt();
        assert!((vals[0] - 1.0 / rms0).abs() < 1e-5, "row0[0]={}", vals[0]);
        assert!((vals[1] - 2.0 / rms0).abs() < 1e-5, "row0[1]={}", vals[1]);
        assert!((vals[2] - 3.0 / rms0).abs() < 1e-5, "row0[2]={}", vals[2]);
    }

    #[test]
    fn test_forge_rms_norm_cpu_vs_candle() {
        // Compare our CPU impl against candle_nn::rms_norm with ones weight
        let dev = Device::Cpu;
        let x = Tensor::from_slice(
            &[0.5f32, -1.2, 3.0, 0.1, -0.7, 2.5, 1.0, 0.0, -1.0, 0.3, 0.8, -0.5],
            (4, 3),
            &dev,
        ).unwrap();
        let gamma = Tensor::ones(3, DType::F32, &dev).unwrap();
        let forge_out = crate::ops::rms_norm::RmsNorm::forward(&x, &gamma, 1e-5).unwrap();

        // candle_nn rms_norm uses weight initialized from VarBuilder
        // We'll compute reference manually: y = x * rsqrt(mean(x²) + eps)
        let x_sq = x.sqr().unwrap();
        let mean_sq = x_sq.mean_keepdim(1).unwrap();
        let rstd = (mean_sq + 1e-5f64).unwrap().sqrt().unwrap().recip().unwrap();
        let ref_out = x.broadcast_mul(&rstd).unwrap();

        let diff = forge_out.sub(&ref_out).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        assert!(max_diff < 1e-5, "CPU vs reference max_diff={}", max_diff);
    }

    #[test]
    fn test_forge_rms_norm_cuda() {
        let dev = get_device();
        let x = Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6.], (2, 3), &dev).unwrap();
        let gamma = Tensor::ones(3, DType::F32, &dev).unwrap();
        let out = crate::ops::rms_norm::RmsNorm::forward(&x, &gamma, 1e-5).unwrap();
        assert_eq!(out.dims(), &[2, 3]);

        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let rms0 = (14.0f32 / 3.0 + 1e-5).sqrt();
        assert!((vals[0] - 1.0 / rms0).abs() < 1e-5, "cuda row0[0]={}", vals[0]);
        assert!((vals[1] - 2.0 / rms0).abs() < 1e-5, "cuda row0[1]={}", vals[1]);
        assert!((vals[2] - 3.0 / rms0).abs() < 1e-5, "cuda row0[2]={}", vals[2]);
    }

    #[test]
    fn test_forge_rms_norm_cuda_vs_cpu() {
        let cpu = Device::Cpu;
        let cuda = get_device();
        let data = [0.5f32, -1.2, 3.0, 0.1, -0.7, 2.5, 1.0, 0.0, -1.0, 0.3, 0.8, -0.5];

        let x_cpu = Tensor::from_slice(&data, (4, 3), &cpu).unwrap();
        let g_cpu = Tensor::ones(3, DType::F32, &cpu).unwrap();
        let out_cpu = crate::ops::rms_norm::RmsNorm::forward(&x_cpu, &g_cpu, 1e-5).unwrap();

        let x_cuda = Tensor::from_slice(&data, (4, 3), &cuda).unwrap();
        let g_cuda = Tensor::ones(3, DType::F32, &cuda).unwrap();
        let out_cuda = crate::ops::rms_norm::RmsNorm::forward(&x_cuda, &g_cuda, 1e-5).unwrap();

        let cpu_vals: Vec<f32> = out_cpu.flatten_all().unwrap().to_vec1().unwrap();
        let cuda_vals: Vec<f32> = out_cuda.flatten_all().unwrap().to_vec1().unwrap();
        for (i, (c, g)) in cpu_vals.iter().zip(cuda_vals.iter()).enumerate() {
            assert!((c - g).abs() < 1e-5, "elem {} cpu={} cuda={}", i, c, g);
        }
    }

    #[test]
    fn test_forge_rms_norm_with_gamma() {
        let dev = get_device();
        let x = Tensor::from_slice(&[1f32, 2., 3., 4., 5., 6.], (2, 3), &dev).unwrap();
        let gamma = Tensor::from_slice(&[0.5f32, 1.0, 2.0], 3, &dev).unwrap();
        let out = crate::ops::rms_norm::RmsNorm::forward(&x, &gamma, 1e-5).unwrap();

        // With gamma=[0.5, 1.0, 2.0], output should be x_norm * gamma
        let gamma_ones = Tensor::ones(3, DType::F32, &dev).unwrap();
        let out_ones = crate::ops::rms_norm::RmsNorm::forward(&x, &gamma_ones, 1e-5).unwrap();
        let expected = out_ones.broadcast_mul(&gamma).unwrap();

        let diff = out.sub(&expected).unwrap().abs().unwrap();
        let max_diff: f32 = diff.max(0).unwrap().max(0).unwrap().to_scalar().unwrap();
        assert!(max_diff < 1e-5, "gamma scaling max_diff={}", max_diff);
    }

    #[test]
    fn test_forge_rms_norm_backward_input() {
        // Verify dX gradient via finite differences
        let dev = Device::Cpu;
        let eps_norm = 1e-5f64;
        let h = 1e-4f64;
        let data = [0.5f32, -1.2, 3.0, 0.1, -0.7, 2.5];
        let gamma_data = [1.0f32, 0.5, 2.0];

        let x = Var::from_tensor(
            &Tensor::from_slice(&data, (2, 3), &dev).unwrap()
        ).unwrap();
        let gamma = Tensor::from_slice(&gamma_data, 3, &dev).unwrap();

        // Autograd backward
        let out = crate::ops::rms_norm::RmsNorm::forward(x.as_tensor(), &gamma, eps_norm).unwrap();
        let loss = out.sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let dx = grads.get(x.as_tensor()).unwrap();
        let dx_vals: Vec<f32> = dx.flatten_all().unwrap().to_vec1().unwrap();

        // Finite differences
        for idx in 0..6 {
            let mut data_plus = data;
            let mut data_minus = data;
            data_plus[idx] += h as f32;
            data_minus[idx] -= h as f32;

            let x_plus = Tensor::from_slice(&data_plus, (2, 3), &dev).unwrap();
            let x_minus = Tensor::from_slice(&data_minus, (2, 3), &dev).unwrap();
            let out_plus = crate::ops::rms_norm::RmsNorm::forward(&x_plus, &gamma, eps_norm).unwrap();
            let out_minus = crate::ops::rms_norm::RmsNorm::forward(&x_minus, &gamma, eps_norm).unwrap();
            let l_plus: f32 = out_plus.sum_all().unwrap().to_scalar().unwrap();
            let l_minus: f32 = out_minus.sum_all().unwrap().to_scalar().unwrap();
            let fd = (l_plus - l_minus) as f64 / (2.0 * h);

            assert!(
                (dx_vals[idx] as f64 - fd).abs() < 1e-2,
                "dX[{}]: autograd={} fd={}", idx, dx_vals[idx], fd
            );
        }
    }

    #[test]
    fn test_forge_rms_norm_backward_gamma() {
        // Verify dgamma gradient via finite differences
        let dev = Device::Cpu;
        let eps_norm = 1e-5f64;
        let h = 1e-4f64;
        let x_data = [0.5f32, -1.2, 3.0, 0.1, -0.7, 2.5];
        let gamma_data = [1.0f32, 0.5, 2.0];

        let x = Tensor::from_slice(&x_data, (2, 3), &dev).unwrap();
        let gamma = Var::from_tensor(
            &Tensor::from_slice(&gamma_data, 3, &dev).unwrap()
        ).unwrap();

        let out = crate::ops::rms_norm::RmsNorm::forward(&x, gamma.as_tensor(), eps_norm).unwrap();
        let loss = out.sum_all().unwrap();
        let grads = loss.backward().unwrap();
        let dgamma = grads.get(gamma.as_tensor()).unwrap();
        let dgamma_vals: Vec<f32> = dgamma.to_vec1().unwrap();

        for idx in 0..3 {
            let mut gp = gamma_data;
            let mut gm = gamma_data;
            gp[idx] += h as f32;
            gm[idx] -= h as f32;

            let gp_t = Tensor::from_slice(&gp, 3, &dev).unwrap();
            let gm_t = Tensor::from_slice(&gm, 3, &dev).unwrap();
            let out_p = crate::ops::rms_norm::RmsNorm::forward(&x, &gp_t, eps_norm).unwrap();
            let out_m = crate::ops::rms_norm::RmsNorm::forward(&x, &gm_t, eps_norm).unwrap();
            let lp: f32 = out_p.sum_all().unwrap().to_scalar().unwrap();
            let lm: f32 = out_m.sum_all().unwrap().to_scalar().unwrap();
            let fd = (lp - lm) as f64 / (2.0 * h);

            assert!(
                (dgamma_vals[idx] as f64 - fd).abs() < 1e-2,
                "dgamma[{}]: autograd={} fd={}", idx, dgamma_vals[idx], fd
            );
        }
    }

    #[test]
    fn test_forge_rms_norm_autograd_e2e() {
        // End-to-end: forward → loss → backward → grads ≠ 0
        let dev = Device::Cpu;
        let x = Var::from_tensor(
            &Tensor::randn(0f32, 1., (4, 8), &dev).unwrap()
        ).unwrap();
        let gamma = Var::from_tensor(
            &Tensor::ones(8, DType::F32, &dev).unwrap()
        ).unwrap();

        let out = crate::ops::rms_norm::RmsNorm::forward(
            x.as_tensor(), gamma.as_tensor(), 1e-5
        ).unwrap();
        let loss = out.sqr().unwrap().sum_all().unwrap();
        let grads = loss.backward().unwrap();

        let gx = grads.get(x.as_tensor()).unwrap();
        let gg = grads.get(gamma.as_tensor()).unwrap();
        let gx_norm: f32 = gx.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
        let gg_norm: f32 = gg.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();
        assert!(gx_norm > 0.0, "dX grad norm is zero");
        assert!(gg_norm > 0.0, "dgamma grad norm is zero");
    }
}
