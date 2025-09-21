use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;

use crate::tensor::Tensor;
pub struct ModelWeights {
    pub tensors: HashMap<String, Tensor>,
}

impl ModelWeights {
    pub fn load_from_safetensors(
        path: &std::path::Path,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        println!("Loading model weights from {:?}", path);

        let buffer = fs::read(path).unwrap();
        let safetensors = SafeTensors::deserialize(&buffer).unwrap();

        println!("Processing {} tensors...", safetensors.names().len());

        let mut tensors = HashMap::new();

        for name in safetensors.names() {
            let tensor_view = safetensors.tensor(name).unwrap();
            let raw_data = tensor_view.data();
            let data: Vec<f32> = raw_data
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();

            let tensor = Tensor::from_shape(tensor_view.shape(), data);
            tensors.insert(name.to_string(), tensor);
        }

        println!("✅ Successfully loaded {} tensors", tensors.len());

        Ok(ModelWeights { tensors })
    }
}
