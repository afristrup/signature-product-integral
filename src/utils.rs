extern crate candle_core;
use candle_core::{Device, Error, Shape, Tensor};

pub(crate) fn index_select(input: &Tensor, indices: &Tensor) -> Result<Tensor, Error> {
    let device = Device::new_cuda(0)?;
    let input = input.to_device(&device)?;

    let dim = input.shape().dims()[0];
    let ndim = input.shape().dims().len();
    let n = indices.shape().dims()[1];
    assert!(
        n <= ndim,
        "n should be less than or equal to the number of dimensions of input"
    );
    let strides = (0..n)
        .map(|i| dim.pow(i as u32) as f32)
        .collect::<Vec<f32>>();
    let flattened = input.flatten_all()?;
    let _select = |index: &Tensor| -> Result<Tensor, Error> {
        let position = index.mul(&Tensor::from_vec(
            strides.clone(),
            &Shape::from_dims(&[n]),
            &index.device(),
        )?)?;
        Ok(flattened.index_select(&position, 0)?)
    };
    Ok(indices.apply(&_select)?)
}

pub(crate) fn lyndon_words(depth: usize, dim: usize) -> Result<Vec<Tensor>, Error> {
    let device = Device::new_cuda(0)?;

    let mut list_of_words = vec![];
    let mut word = vec![-1];
    while !word.is_empty() {
        let last_index = word.len() - 1;
        word[last_index] += 1;

        let m = word.len();
        list_of_words.push(
            Tensor::from_vec(
                word.clone().iter().map(|num| *num as f32).collect(),
                &Shape::from_dims(&[m]),
                &device,
            )
            .unwrap(),
        );

        while word.len() < depth {
            word.push(word[word.len() - m]);
        }
        while !word.is_empty() && word[word.len() - 1] == (dim - 1) as i32 {
            word.pop();
        }
    }
    Ok(list_of_words)
}

pub(crate) fn compress(inputs: Vec<Tensor>, indices: Vec<Tensor>) -> Result<Vec<Tensor>, Error> {
    Ok(inputs
        .iter()
        .zip(indices.iter())
        .map(|(term, index)| index_select(term, index).unwrap())
        .collect())
}

pub(crate) fn get_depth(dim: usize, depth: usize) -> (usize, usize) {
    let offset = dim.pow(depth as u32);
    let start = dim * (1 - offset) / (1 - dim);
    (offset, start)
}

pub(crate) fn term_at(
    flattened_signature: &Tensor,
    dim: usize,
    term_i: usize,
    start: usize,
    offset: usize,
) -> Result<Tensor, Error> {
    let slice = flattened_signature.slice_scatter(&flattened_signature, offset, start)?;
    Ok(slice.reshape(&[term_i + 1, dim])?)
}

pub(crate) fn unravel_signature(
    signature: &Tensor,
    dim: usize,
    depth: usize,
) -> Result<Vec<Tensor>, Error> {
    let mut unraveled = vec![];
    let (mut offset, mut start) = get_depth(dim, depth);

    for term_i in 0..depth {
        unraveled.push(term_at(signature, dim, term_i, start, offset)?);
        start += offset;
        offset *= dim;
    }

    Ok(unraveled)
}
