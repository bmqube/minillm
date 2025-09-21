use ndarray::{Array1, Array2, Array3, Array4, ArrayD};

pub enum Tensor {
    D1(Array1<f32>),
    D2(Array2<f32>),
    D3(Array3<f32>),
    D4(Array4<f32>),
    Dyn(ArrayD<f32>),
}

impl Tensor {
    pub fn from_shape(shape: &[usize], data: Vec<f32>) -> Self {
        match shape.len() {
            1 => Tensor::D1(Array1::from(data)),
            2 => Tensor::D2(Array2::from_shape_vec((shape[0], shape[1]), data).unwrap()),
            3 => Tensor::D3(Array3::from_shape_vec((shape[0], shape[1], shape[2]), data).unwrap()),
            4 => Tensor::D4(
                Array4::from_shape_vec((shape[0], shape[1], shape[2], shape[3]), data).unwrap(),
            ),
            _ => Tensor::Dyn(ArrayD::from_shape_vec(shape, data).unwrap()),
        }
    }
}
