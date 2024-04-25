extern crate candle_core;
use candle_core::{Device, Error, Tensor};

pub(crate) fn otimes(input: &Tensor, other: &Tensor) -> Result<Tensor, Error> {
    let device = Device::new_cuda(0)?;
    let input = input.to_device(&device).unwrap();
    let other = other.to_device(&device).unwrap();

    let input_shape = input.shape().dims();
    let other_shape = other.shape().dims();

    let expanded_input = input
        .reshape(
            [
                input_shape,
                other_shape
                    .iter()
                    .map(|_| 1)
                    .collect::<Vec<usize>>()
                    .as_slice(),
            ]
            .concat(),
        )
        .unwrap();

    let expanded_other = other
        .reshape(
            [
                other_shape
                    .iter()
                    .map(|_| 1)
                    .collect::<Vec<usize>>()
                    .as_slice(),
                other_shape,
            ]
            .concat(),
        )
        .unwrap();

    Ok(expanded_input.mul(&expanded_other).unwrap())
}

pub(crate) fn addcmul(input: &Tensor, other: &Tensor, another: &Tensor) -> Result<Tensor, Error> {
    let device = Device::new_cuda(0)?;
    let input = input.to_device(&device).unwrap();
    let other = other.to_device(&device).unwrap();
    let another = another.to_device(&device).unwrap();

    Ok(input.add(&other.mul(&another).unwrap()).unwrap())
}

pub(crate) fn restricted_exp(input: &Tensor, depth: usize) -> Result<Vec<Tensor>, Error> {
    let device = Device::new_cuda(0)?;
    let input = input.to_device(&device).unwrap();

    let mut result = vec![input.clone()];
    for i in 1..depth {
        let exp = input
            .pow(&Tensor::from_vec(vec![i as f32], *&1, &device).unwrap())
            .unwrap();
        result.push(exp);
    }

    Ok(result)
}

pub(crate) fn mult_fused_restricted_exp(
    input: &Tensor,
    others: Vec<Tensor>,
) -> Result<Vec<Tensor>, Error> {
    let device = Device::new_cuda(0)?;
    let input = input.to_device(&device).unwrap();
    let depth = others.len();

    let mut result = vec![];
    for depth_index in 0..depth {
        let mut current = Tensor::from_vec(vec![1.0f32], *&1, &device).unwrap();
        for (i, other) in others.iter().enumerate() {
            current = addcmul(
                &other,
                &current,
                &(input.mul(&Tensor::try_from(1f32 / ((depth_index - i) as f32))?)).unwrap(),
            )?;
            result.push(current.clone());
        }
    }

    Ok(result)
}

pub(crate) fn mult_inner(
    inputs: &Vec<Tensor>,
    others: &Vec<Tensor>,
    depth_index: usize,
) -> Result<Tensor, Error> {
    let mut result = Tensor::from_vec(vec![0.0f32], *&1, &Device::new_cuda(0)?).unwrap();
    for i in 0..depth_index {
        let product = otimes(&inputs[i], &others[depth_index - i - 1]).unwrap();
        result = result.add(&product).unwrap();
    }

    Ok(result)
}

pub(crate) fn mult(inputs: &Vec<Tensor>, others: &Vec<Tensor>) -> Result<Vec<Tensor>, Error> {
    let depth = inputs.len();
    let mut result = vec![];

    for i in 0..depth {
        let sum = inputs[i].add(&others[i]).unwrap();
        result.push(sum);
    }

    for i in 1..depth {
        let inner = mult_inner(inputs, others, i)?;
        result[i] = result[i].add(&inner).unwrap();
    }

    Ok(result)
}

pub(crate) fn mult_partial(
    inputs: &Vec<Tensor>,
    others: &Vec<Tensor>,
    scalar_term_value: f32,
    top_terms_to_skip: usize,
) -> Result<Vec<Tensor>, Error> {
    let device = Device::new_cuda(0)?;

    let depth = inputs.len();

    let mut result = inputs.clone();

    for depth_index in (0..depth - top_terms_to_skip).rev() {
        result[depth_index] =
            Tensor::from_vec(vec![0.0f32], &*result[depth_index].shape(), &device).unwrap();
        result[depth_index] = mult_inner(&result, others, depth_index)?;
        result[depth_index] = result[depth_index]
            .add(
                &others[depth_index]
                    .mul(&Tensor::from_vec(vec![scalar_term_value], *&1, &device).unwrap())
                    .unwrap(),
            )
            .unwrap();
    }

    Ok(result)
}

fn log_coef_at_depth(depth: usize) -> f32 {
    let sign = if depth % 2 == 0 { -1.0 } else { 1.0 };
    sign / (depth + 2) as f32
}

pub(crate) fn log(input: &Vec<Tensor>) -> Result<Vec<Tensor>, Error> {
    let device = Device::new_cuda(0)?;

    let depth = input.len();
    if depth == 1 {
        return Ok(input.clone());
    }

    let mut output = vec![];
    for x in input.iter() {
        output.push(Tensor::from_vec(vec![0.0f32], &*x.shape(), &device).unwrap());
    }
    output[0] = input[0]
        .mul(&Tensor::from_vec(vec![log_coef_at_depth(depth - 2)], *&1, &device).unwrap())
        .unwrap();

    for depth_index in (0..depth - 2).rev() {
        output = mult_partial(
            &output,
            input,
            log_coef_at_depth(depth_index),
            depth_index + 1,
        )?;
    }

    output = mult_partial(&output, input, 1.0, 0)?;

    Ok(output)
}
