mod product_integral;
mod signatures;
mod tensor_ops;
mod utils;

use candle_core::{Device, Error, Tensor};


/// Function to compute a matrix with dynamic modifications based on `x`.
fn fn_mat(x: f64) -> Result<Tensor, Error> {
    let device = Device::new_cuda(0)?;

    // Initialize a 3x3 matrix with zeros
    let mut data = vec![0.0; 9]; // 3x3 matrix

    // Set specific elements in the matrix
    data[2] = 0.0005 + 10f64.powf(5.88 + 0.038 * x - 10.0); // A[0, 2]
    data[1] = 0.0004 + 10f64.powf(4.54 + 0.06 * x - 10.0);  // A[0, 1]
    data[3] = 2.0058 * f64::exp(-0.117 * x);                // A[1, 0]

    // Conditionally set A[1, 2]
    data[5] = data[2] * if x > 65.0 { 1.0 } else { 2.0 };

    // Create tensor from data
    let mut a = Tensor::from_vec(data, &[3, 3], &device)?;

    // Compute the sum of each column, prepare a vector for diag_embed
    let evec = Tensor::from_vec(vec![1.0, 1.0, 1.0], &[3, 1], &device)?; // Column vector
    let dsum = a.matmul(&evec)?; // This should be a 3x1 tensor

    // Convert column sums into a diagonal matrix
    let dsum_diag = diag_embed(&dsum.squeeze(1)?, &device)?; // Ensure to squeeze the correct dimension

    // Subtract the diagonal matrix from A to adjust its diagonal entries
    let result = a.sub(&dsum_diag)?;

    Ok(result)
}

fn fn_mat_wrapper(x: f64) -> Tensor {
    fn_mat(x).unwrap()
}

/// Creates a diagonal matrix from a 1D tensor.
fn diag_embed(vector: &Tensor, device: &Device) -> Result<Tensor, Error> {
    let len = vector.dims()[0];
    let mut data = vec![0.0; len * len]; // create a square matrix filled with zeros

    for i in 0..len {
        let value = vector.get(i)?;  // Assuming `get` method retrieves the i-th element from the tensor
        data[i * len + i] = f64::try_from(value)?;
    }

    Tensor::from_vec(data, &[len, len], &device)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_cuda(0)?;

    let input =
        Tensor::from_vec((0..27).map(|num| num as f32).collect(), &[3, 3, 3], &device).unwrap();
    let indices = Tensor::from_vec(
        (vec![0, 1, 1, 2, 2, 0])
            .iter()
            .map(|num| *num as f32)
            .collect(),
        &[3, 2],
        &device,
    )
    .unwrap();
    println!("Input Tensor: {:?}", input);
    println!("Indices Tensor: {:?}", indices);
    // println!("Selected Tensor: {:?}", selected);

    // Example usage with a simple matrix function
    // let fn_mat = Arc::new(|x: f64| {
    //     let device = Device::new_cuda(0).unwrap();
    //     Tensor::from_vec(
    //         vec![x as f32, 0.0, 0.0, 0.0, x as f32, 0.0, 0.0, 0.0, x as f32],
    //         &[3, 3],
    //         &device,
    //     )
    //     .unwrap()
    // });

    // fn fn_mat(x: f64) -> Tensor {
    //     let device = Device::new_cuda(0).unwrap();
    //     Tensor::from_vec(
    //         vec![x as f64, 0.0, 0.0, 0.0, x as f64, 0.0, 0.0, 0.0, x as f64],
    //         &[3, 3],
    //         &device,
    //     )
    //     .unwrap()
    // }







    let s = 40.0;
    let t = 70.0;
    let n = 1000;
    let result = product_integral::rk4_prodint(&fn_mat_wrapper, s, t, n);

    println!("Resulting tensor:\n{:?}", result.unwrap().to_vec2::<f64>()?);

    Ok(())
}
