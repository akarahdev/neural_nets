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

pub struct CombinedNetwork<
    const I1: usize,
    const I2: usize,
    const O1: usize,
    const O2: usize,
    N1: NeuralNetwork<I1, O1>,
    N2: NeuralNetwork<I2, O2>,
> {
    pub(crate) n1: N1,
    pub(crate) n2: N2,
}

impl<
    const I1: usize,
    const I2: usize,
    const O1: usize,
    const O2: usize,
    N1: NeuralNetwork<I1, O1>,
    N2: NeuralNetwork<I2, O2>,
> NeuralNetwork<{ I1 + I2 }, { O1 + O2 }> for CombinedNetwork<I1, I2, O1, O2, N1, N2>
{
    fn feed(&mut self, arr: &[f16; I1 + I2]) -> [f16; O1 + O2] {
        let n1o = self.n1.feed(&arr[0..I1].try_into().unwrap());
        let n2o = self.n2.feed(&arr[{ I1 }..{ I1 + I2 }].try_into().unwrap());
        let mut out = [0.0; O1 + O2];
        for (idx, item) in n1o.iter().enumerate() {
            out[idx] = *item;
        }
        for (idx, item) in n2o.iter().enumerate() {
            out[idx + I1] = *item;
        }
        out
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

    #[test]
    fn combine_doublers() {
        let perceptron = Perceptron::new([2.0], 0.0, ActFns::linear());
        let mut combined = perceptron.clone().combine(perceptron);
        assert_eq!(combined.feed(&[1.0, 2.0]), [2.0, 4.0]);
    }
}
