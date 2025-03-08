mod perceptron;
pub use perceptron::*;
mod static_feed_forward;
pub use static_feed_forward::*;

pub trait NeuralNetwork<const I: usize, const O: usize> {
    fn feed(&mut self, arr: &[f16; I]) -> [f16; O];
}
