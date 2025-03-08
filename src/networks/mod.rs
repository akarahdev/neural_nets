mod perceptron;
pub use perceptron::*;

pub trait NeuralNetwork<const I: usize, const O: usize> {
    fn feed(&mut self, arr: &[f16; I]) -> [f16; O];
}
