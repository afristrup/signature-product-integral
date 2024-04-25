use std::ops::Mul;
use candle_core::{DType, Device, Error, Tensor};

pub(crate) fn rk4_prodint<F>(fn_mat: F, s: f64, t: f64, n: usize) -> Result<Tensor, Error>
    where
        F: Fn(f64) -> Tensor + Sync + Send + 'static,
{
    let device = Device::new_cuda(0)?;
    let h =(t - s) / n as f64;
    let half = Tensor::try_from(0.5)?;
    let mut x = s;
    let x_tensor = Tensor::try_from(x)?;
    let mut y = Tensor::eye(fn_mat(s).dims()[0], DType::F64, &device)?; // assuming square tensors

    for _ in 0..n {
        let k1 = y.matmul(&fn_mat(x))?;
        // let inner1 = h.mul(&0.5).unwrap();
        let inner1 = h * 0.5;
        let k2 = y.clone().add(&(k1.clone().mul(inner1).unwrap()))?.matmul(&fn_mat(x + inner1))?;
        let k3 = y.clone().add(&(k2.clone().mul(inner1).unwrap()))?.matmul(&fn_mat(x + inner1))?;
        let k4 = y.clone().add(&(k3.clone().mul(h).unwrap()))?.matmul(&fn_mat(x + h))?;

        y = y.add(&k1)?
            .add(&(k2.mul( 2.0).unwrap()))?
            .add(&(k3.mul( 2.0).unwrap()))?
            .add(&k4)?;
        y = y.mul(h / 6.0).unwrap(); // Assuming scale is a method to multiply tensor by scalar that handles Result
        x += h;
    }

    Ok(y)
}
