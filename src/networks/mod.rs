mod perceptron;
pub use perceptron::*;

pub trait NeuralNetwork<const I: usize, const O: usize, F: Fn(f16) -> f16> {
    fn feed(&mut self, arr: &[f16; I]) -> [f16; O];

    const FLATTENED_SIZE: usize;
    fn flatten(&self) -> [f16; Self::FLATTENED_SIZE];
    fn unflatten(flattened: [f16; Self::FLATTENED_SIZE], activation_function: F) -> Self;
}
