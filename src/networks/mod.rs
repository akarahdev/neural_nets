mod perceptron;
pub use perceptron::*;
mod static_feed_forward;
pub use static_feed_forward::*;
mod combinators;
pub use combinators::*;

pub trait NeuralNetwork<const I: usize, const O: usize> {
    fn feed(&mut self, arr: &[f16; I]) -> [f16; O];
}

pub trait NeuralNetworkExt<const I: usize, const O: usize>
where
    Self: Sized + NeuralNetwork<I, O>,
{
    fn and_then<const O2: usize, R: NeuralNetwork<O, O2>>(
        self,
        right: R,
    ) -> AndThenNetwork<I, O, O2, Self, R> {
        AndThenNetwork { left: self, right }
    }

    fn combine<const I2: usize, const O2: usize, R: NeuralNetwork<I2, O2>>(
        self,
        right: R,
    ) -> CombinedNetwork<I, I2, O, O2, Self, R> {
        CombinedNetwork {
            n1: self,
            n2: right,
        }
    }
}

impl<const I: usize, const O: usize, T: NeuralNetwork<I, O>> NeuralNetworkExt<I, O> for T {}
