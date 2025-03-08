use super::NeuralNetwork;

pub struct AndThenNetwork<
    const I: usize,
    const M: usize,
    const O: usize,
    L: NeuralNetwork<I, M>,
    R: NeuralNetwork<M, O>,
> {
    pub(crate) left: L,
    pub(crate) right: R,
}

impl<const I: usize, const M: usize, const O: usize, L: NeuralNetwork<I, M>, R: NeuralNetwork<M, O>>
    NeuralNetwork<I, O> for AndThenNetwork<I, M, O, L, R>
{
    fn feed(&mut self, arr: &[f16; I]) -> [f16; O] {
        self.right.feed(&self.left.feed(arr))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        networks::{NeuralNetwork, NeuralNetworkExt, Perceptron},
        utils::ActFns,
    };

    #[test]
    fn and_then_doubling() {
        let mut network = Perceptron::new([2.0], 0.0, ActFns::linear()).and_then(Perceptron::new(
            [2.0],
            0.0,
            ActFns::linear(),
        ));
        assert_eq!(network.feed(&[2.0]), [8.0]);
    }
}
