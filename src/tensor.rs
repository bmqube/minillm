use ndarray::{ArrayD, Axis};

#[allow(dead_code)]
pub struct Tensor {
    pub data: ArrayD<f32>,
}

#[allow(dead_code)]
impl Tensor {
    pub fn from_shape(shape: &[usize], data: Vec<f32>) -> Self {
        Self {
            data: ArrayD::from_shape_vec(shape, data).unwrap(),
        }
    }

    // Matrix multiplication - much cleaner with dynamic arrays!
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let self_shape = self.data.shape();
        let other_shape = other.data.shape();

        match (self_shape.len(), other_shape.len()) {
            // 1D x 1D: dot product
            (1, 1) => {
                let result = self
                    .data
                    .iter()
                    .zip(other.data.iter())
                    .map(|(a, b)| a * b)
                    .sum::<f32>();
                Tensor::from_shape(&[1], vec![result])
            }

            // 1D x 2D: vector-matrix multiplication
            (1, 2) => {
                let a = self
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix1>()
                    .unwrap();
                let b = other
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();
                let result = a.dot(&b);
                Tensor {
                    data: result.into_dyn(),
                }
            }

            // 1D x 3D: vector-batch matrix multiplication
            (1, 3) => {
                let batch_size = other_shape[0];
                let result_shape = [batch_size, other_shape[2]];
                let mut result_data = Vec::with_capacity(result_shape.iter().product());

                let self_1d = self
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix1>()
                    .unwrap();

                for i in 0..batch_size {
                    let other_slice = other
                        .data
                        .slice(ndarray::s![i, .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();
                    let batch_result = self_1d.dot(&other_slice);
                    result_data.extend(batch_result.iter().copied());
                }

                Tensor::from_shape(&result_shape, result_data)
            }

            // 1D x 4D: vector-4D tensor multiplication
            (1, 4) => {
                let batch_dims = &other_shape[..other_shape.len() - 2];
                let total_batches: usize = batch_dims.iter().product();
                let result_shape = [batch_dims, &[other_shape[other_shape.len() - 1]]].concat();
                let mut result_data = Vec::with_capacity(result_shape.iter().product());

                let self_1d = self
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix1>()
                    .unwrap();

                for batch_idx in 0..total_batches {
                    // Convert flat batch index to multi-dimensional indices
                    let mut temp_idx = batch_idx;
                    let mut slice_start = Vec::new();
                    for &dim_size in batch_dims.iter().rev() {
                        slice_start.insert(0, temp_idx % dim_size);
                        temp_idx /= dim_size;
                    }

                    // Extract 2D slice using simple slicing
                    let other_slice = other
                        .data
                        .slice(ndarray::s![slice_start[0], slice_start[1], .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();

                    let batch_result = self_1d.dot(&other_slice);
                    result_data.extend(batch_result.iter().copied());
                }

                Tensor::from_shape(&result_shape, result_data)
            }

            // 2D x 1D: matrix-vector multiplication
            (2, 1) => {
                let a = self
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();
                let b = other
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix1>()
                    .unwrap();
                let result = a.dot(&b);
                Tensor {
                    data: result.into_dyn(),
                }
            }

            // 2D x 2D: standard matrix multiplication
            (2, 2) => {
                let a = self
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();
                let b = other
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();
                let result = a.dot(&b);
                Tensor {
                    data: result.into_dyn(),
                }
            }

            // 2D x 3D: matrix-batch multiplication
            (2, 3) => {
                let batch_size = other_shape[0];
                let result_shape = [batch_size, self_shape[0], other_shape[2]];
                let mut result_data = Vec::with_capacity(result_shape.iter().product());

                let self_2d = self
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();

                for i in 0..batch_size {
                    let other_slice = other
                        .data
                        .slice(ndarray::s![i, .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();
                    let batch_result = self_2d.dot(&other_slice);
                    result_data.extend(batch_result.iter().copied());
                }

                Tensor::from_shape(&result_shape, result_data)
            }

            // 2D x 4D: matrix-4D tensor multiplication
            (2, 4) => {
                let batch_dims = &other_shape[..other_shape.len() - 2];
                let total_batches: usize = batch_dims.iter().product();
                let result_shape = [
                    batch_dims,
                    &[self_shape[0], other_shape[other_shape.len() - 1]],
                ]
                .concat();
                let mut result_data = Vec::with_capacity(result_shape.iter().product());

                let self_2d = self
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();

                for batch_idx in 0..total_batches {
                    // Flatten batch index to multi-dimensional indices
                    let mut temp_idx = batch_idx;
                    let mut slice_start = Vec::new();
                    for &dim_size in batch_dims.iter().rev() {
                        slice_start.insert(0, temp_idx % dim_size);
                        temp_idx /= dim_size;
                    }

                    // Create slice for this batch
                    let other_slice = other
                        .data
                        .slice(ndarray::s![slice_start[0], slice_start[1], .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();

                    let batch_result = self_2d.dot(&other_slice);
                    result_data.extend(batch_result.iter().copied());
                }

                Tensor::from_shape(&result_shape, result_data)
            }

            // 3D x 1D: batch matrix-vector multiplication
            (3, 1) => {
                let batch_size = self_shape[0];
                let result_shape = [batch_size, self_shape[1]];
                let mut result_data = Vec::with_capacity(result_shape.iter().product());

                let other_1d = other
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix1>()
                    .unwrap();

                for i in 0..batch_size {
                    let self_slice = self
                        .data
                        .slice(ndarray::s![i, .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();
                    let batch_result = self_slice.dot(&other_1d);
                    result_data.extend(batch_result.iter().copied());
                }

                Tensor::from_shape(&result_shape, result_data)
            }

            // 3D x 2D: batch matrix multiplication (most common in transformers)
            (3, 2) => {
                let batch_size = self_shape[0];
                let result_shape = [batch_size, self_shape[1], other_shape[1]];
                let mut result_data = Vec::with_capacity(result_shape.iter().product());

                let other_2d = other
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();

                for i in 0..batch_size {
                    let self_slice = self
                        .data
                        .slice(ndarray::s![i, .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();
                    let batch_result = self_slice.dot(&other_2d);
                    result_data.extend(batch_result.iter().copied());
                }

                Tensor::from_shape(&result_shape, result_data)
            }

            // 3D x 3D: batch matrix multiplication
            (3, 3) => {
                assert_eq!(self_shape[0], other_shape[0], "Batch sizes must match");
                let batch_size = self_shape[0];
                let result_shape = [batch_size, self_shape[1], other_shape[2]];
                let mut result_data = Vec::with_capacity(result_shape.iter().product());

                for i in 0..batch_size {
                    let self_slice = self
                        .data
                        .slice(ndarray::s![i, .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();
                    let other_slice = other
                        .data
                        .slice(ndarray::s![i, .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();
                    let batch_result = self_slice.dot(&other_slice);
                    result_data.extend(batch_result.iter().copied());
                }

                Tensor::from_shape(&result_shape, result_data)
            }

            // 3D x 4D: batch-4D tensor multiplication
            (3, 4) => {
                let self_batch = self_shape[0];
                let other_batch_dims = &other_shape[..other_shape.len() - 2];
                let other_total_batches: usize = other_batch_dims.iter().product();

                assert_eq!(
                    self_batch, other_total_batches,
                    "Batch dimensions must match"
                );

                let result_shape = [
                    other_batch_dims,
                    &[self_shape[1], other_shape[other_shape.len() - 1]],
                ]
                .concat();
                let mut result_data = Vec::with_capacity(result_shape.iter().product());

                for batch_idx in 0..other_total_batches {
                    let self_slice = self
                        .data
                        .slice(ndarray::s![batch_idx, .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();

                    // Convert flat batch index to multi-dimensional indices
                    let mut temp_idx = batch_idx;
                    let mut slice_start = Vec::new();
                    for &dim_size in other_batch_dims.iter().rev() {
                        slice_start.insert(0, temp_idx % dim_size);
                        temp_idx /= dim_size;
                    }

                    let other_slice = other
                        .data
                        .slice(ndarray::s![slice_start[0], slice_start[1], .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();

                    let batch_result = self_slice.dot(&other_slice);
                    result_data.extend(batch_result.iter().copied());
                }

                Tensor::from_shape(&result_shape, result_data)
            }

            // 4D x 1D: 4D tensor-vector multiplication
            (4, 1) => {
                let batch_dims = &self_shape[..self_shape.len() - 2];
                let total_batches: usize = batch_dims.iter().product();
                let result_shape = [batch_dims, &[self_shape[self_shape.len() - 2]]].concat();
                let mut result_data = Vec::with_capacity(result_shape.iter().product());

                let other_1d = other
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix1>()
                    .unwrap();

                for batch_idx in 0..total_batches {
                    // Convert flat batch index to multi-dimensional indices
                    let mut temp_idx = batch_idx;
                    let mut slice_start = Vec::new();
                    for &dim_size in batch_dims.iter().rev() {
                        slice_start.insert(0, temp_idx % dim_size);
                        temp_idx /= dim_size;
                    }

                    let self_slice = self
                        .data
                        .slice(ndarray::s![slice_start[0], slice_start[1], .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();

                    let batch_result = self_slice.dot(&other_1d);
                    result_data.extend(batch_result.iter().copied());
                }

                Tensor::from_shape(&result_shape, result_data)
            }

            // 4D x 2D: 4D tensor-matrix multiplication
            (4, 2) => {
                let batch_dims = &self_shape[..self_shape.len() - 2];
                let total_batches: usize = batch_dims.iter().product();
                let result_shape = [
                    batch_dims,
                    &[self_shape[self_shape.len() - 2], other_shape[1]],
                ]
                .concat();
                let mut result_data = Vec::with_capacity(result_shape.iter().product());

                let other_2d = other
                    .data
                    .view()
                    .into_dimensionality::<ndarray::Ix2>()
                    .unwrap();

                for batch_idx in 0..total_batches {
                    // Convert flat batch index to multi-dimensional indices
                    let mut temp_idx = batch_idx;
                    let mut slice_start = Vec::new();
                    for &dim_size in batch_dims.iter().rev() {
                        slice_start.insert(0, temp_idx % dim_size);
                        temp_idx /= dim_size;
                    }

                    let self_slice = self
                        .data
                        .slice(ndarray::s![slice_start[0], slice_start[1], .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();

                    let batch_result = self_slice.dot(&other_2d);
                    result_data.extend(batch_result.iter().copied());
                }

                Tensor::from_shape(&result_shape, result_data)
            }

            // 4D x 3D: 4D tensor-batch multiplication
            (4, 3) => {
                let self_batch_dims = &self_shape[..self_shape.len() - 2];
                let self_total_batches: usize = self_batch_dims.iter().product();
                let other_batch = other_shape[0];

                assert_eq!(
                    self_total_batches, other_batch,
                    "Batch dimensions must match"
                );

                let result_shape = [
                    self_batch_dims,
                    &[self_shape[self_shape.len() - 2], other_shape[2]],
                ]
                .concat();
                let mut result_data = Vec::with_capacity(result_shape.iter().product());

                for batch_idx in 0..self_total_batches {
                    // Convert flat batch index to multi-dimensional indices
                    let mut temp_idx = batch_idx;
                    let mut slice_start = Vec::new();
                    for &dim_size in self_batch_dims.iter().rev() {
                        slice_start.insert(0, temp_idx % dim_size);
                        temp_idx /= dim_size;
                    }

                    let self_slice = self
                        .data
                        .slice(ndarray::s![slice_start[0], slice_start[1], .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();

                    let other_slice = other
                        .data
                        .slice(ndarray::s![batch_idx, .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();

                    let batch_result = self_slice.dot(&other_slice);
                    result_data.extend(batch_result.iter().copied());
                }

                Tensor::from_shape(&result_shape, result_data)
            }

            // 4D x 4D: 4D tensor-4D tensor multiplication
            (4, 4) => {
                let self_batch_dims = &self_shape[..self_shape.len() - 2];
                let other_batch_dims = &other_shape[..other_shape.len() - 2];

                assert_eq!(
                    self_batch_dims, other_batch_dims,
                    "Batch dimensions must match"
                );

                let total_batches: usize = self_batch_dims.iter().product();
                let result_shape = [
                    self_batch_dims,
                    &[
                        self_shape[self_shape.len() - 2],
                        other_shape[other_shape.len() - 1],
                    ],
                ]
                .concat();
                let mut result_data = Vec::with_capacity(result_shape.iter().product());

                for batch_idx in 0..total_batches {
                    // Convert flat batch index to multi-dimensional indices
                    let mut temp_idx = batch_idx;
                    let mut slice_start = Vec::new();
                    for &dim_size in self_batch_dims.iter().rev() {
                        slice_start.insert(0, temp_idx % dim_size);
                        temp_idx /= dim_size;
                    }

                    let self_slice = self
                        .data
                        .slice(ndarray::s![slice_start[0], slice_start[1], .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();

                    let other_slice = other
                        .data
                        .slice(ndarray::s![slice_start[0], slice_start[1], .., ..])
                        .into_dimensionality::<ndarray::Ix2>()
                        .unwrap();

                    let batch_result = self_slice.dot(&other_slice);
                    result_data.extend(batch_result.iter().copied());
                }

                Tensor::from_shape(&result_shape, result_data)
            }

            _ => panic!(
                "Unsupported matmul combination: {:?} x {:?}",
                self_shape, other_shape
            ),
        }
    }

    // Element-wise addition with automatic broadcasting
    pub fn add(&self, other: &Tensor) -> Tensor {
        let result = &self.data + &other.data;
        Tensor { data: result }
    }

    // Layer normalization - works for any dimensionality!
    pub fn layer_norm(&self, weight: &Tensor, bias: &Tensor, eps: f32) -> Tensor {
        let result = self.data.clone();
        let shape = result.shape();
        let last_dim = shape[shape.len() - 1];

        // Get the number of elements to normalize over (last dimension)
        let total_elements = result.len();
        let num_vectors = total_elements / last_dim;

        // Reshape to 2D for easier processing: [num_vectors, last_dim]
        let mut reshaped = result.to_shape((num_vectors, last_dim)).unwrap().to_owned();

        for mut row in reshaped.axis_iter_mut(Axis(0)) {
            // Calculate mean and variance
            let mean = row.mean().unwrap();
            let variance = row.mapv(|v| (v - mean).powi(2)).mean().unwrap();
            let std = (variance + eps).sqrt();

            // Normalize
            row.mapv_inplace(|v| (v - mean) / std);

            // Apply weight and bias
            for i in 0..last_dim {
                row[i] = row[i] * weight.data[[i]] + bias.data[[i]];
            }
        }

        // Reshape back to original shape
        let final_result = reshaped.to_shape(shape).unwrap().to_owned();
        Tensor { data: final_result }
    }

    // GELU activation - applies to all elements regardless of dimensionality
    pub fn gelu(&self) -> Tensor {
        let result = self.data.mapv(|v| {
            0.5 * v
                * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (v + 0.044715 * v.powi(3))).tanh())
        });
        Tensor { data: result }
    }

    // Softmax along any dimension - works for any tensor shape!
    pub fn softmax(&self, dim: usize) -> Tensor {
        let mut result = self.data.clone();
        let shape = result.shape();

        // Calculate total size and stride for the softmax dimension
        let dim_size = shape[dim];
        let before_dim: usize = shape[..dim].iter().product();
        let after_dim: usize = shape[dim + 1..].iter().product();

        // Process each slice along the softmax dimension
        for before_idx in 0..before_dim {
            for after_idx in 0..after_dim {
                // Extract the slice along the softmax dimension
                let mut slice_data = Vec::with_capacity(dim_size);
                for dim_idx in 0..dim_size {
                    let flat_idx =
                        before_idx * dim_size * after_dim + dim_idx * after_dim + after_idx;
                    slice_data.push(result.as_slice().unwrap()[flat_idx]);
                }

                // Apply softmax
                let max_val = slice_data
                    .iter()
                    .fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                slice_data
                    .iter_mut()
                    .for_each(|x| *x = (*x - max_val).exp());
                let sum: f32 = slice_data.iter().sum();
                slice_data.iter_mut().for_each(|x| *x /= sum);

                // Write back
                for (dim_idx, &val) in slice_data.iter().enumerate() {
                    let flat_idx =
                        before_idx * dim_size * after_dim + dim_idx * after_dim + after_idx;
                    result.as_slice_mut().unwrap()[flat_idx] = val;
                }
            }
        }

        Tensor { data: result }
    }

    // Helper methods
    pub fn shape(&self) -> Vec<usize> {
        self.data.shape().to_vec()
    }

    pub fn clone(&self) -> Tensor {
        Tensor {
            data: self.data.clone(),
        }
    }
}
