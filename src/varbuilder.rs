// use candle_core::{Result, Tensor};
// use candle_nn::VarBuilder;

// pub struct GPT2VarBuilder {
//     inner: VarBuilder<'static>,
// }

// impl GPT2VarBuilder {
//     pub fn new(vb: VarBuilder<'static>) -> Self {
//         Self { inner: vb }
//     }

//     pub fn pp(&self, name: &str) -> Self {
//         Self {
//             inner: self.inner.pp(name),
//         }
//     }

//     pub fn get<S: Into<candle_core::Shape>>(&self, s: S, name: &str) -> Result<Tensor> {
//         let tensor = self.inner.get(s, name)?;

//         // Transpose linear layer weights for GPT-2 compatibility
//         if name == "weight" && self.needs_transpose() {
//             tensor.t()
//         } else {
//             Ok(tensor)
//         }
//     }

//     fn needs_transpose(&self) -> bool {
//         // Check if this is a linear layer that needs transposing
//         let path = self.inner.path();
//         path.contains("c_attn")
//             || path.contains("c_proj")
//             || path.contains("mlp.c_fc")
//             || path.contains("mlp.c_proj")
//             || path.contains("lm_head")
//     }
// }
