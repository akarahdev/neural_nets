use crate::utils::ActivationFn;

use super::NeuralNetwork;

pub struct StaticFeedForwardNetwork<
    const I: usize,
    const HN: usize,
    const IHLC: usize,
    const O: usize,
    F: ActivationFn,
> {
    first_hidden_layer: FeedForwardLayer<HN, I>,
    intermediate_hidden_layers: [FeedForwardLayer<HN, HN>; IHLC],
    output_layer: FeedForwardLayer<O, HN>,
    activation_function: F,
}

impl<const I: usize, const HN: usize, const IHLC: usize, const O: usize, F: ActivationFn>
    StaticFeedForwardNetwork<I, HN, IHLC, O, F>
{
    pub fn new(
        first_hidden_layer: FeedForwardLayer<HN, I>,
        intermediate_hidden_layers: [FeedForwardLayer<HN, HN>; IHLC],
        output_layer: FeedForwardLayer<O, HN>,
        activation_function: F,
    ) -> Self {
        StaticFeedForwardNetwork {
            first_hidden_layer,
            intermediate_hidden_layers,
            output_layer,
            activation_function,
        }
    }
}

impl<const I: usize, const HN: usize, const IHLC: usize, const O: usize, F: ActivationFn>
    NeuralNetwork<I, O> for StaticFeedForwardNetwork<I, HN, IHLC, O, F>
{
    fn feed(&mut self, arr: &[f16; I]) -> [f16; O] {
        let mut array = self.first_hidden_layer.feed(arr, self.activation_function);

        for intermediate in &self.intermediate_hidden_layers {
            array = intermediate.feed(&array, self.activation_function);
        }

        self.output_layer.feed(&array, self.activation_function)
    }
}

pub struct FeedForwardLayer<const S: usize, const PL: usize> {
    neurons: [FeedForwardNeuron<PL>; S],
}

impl<const S: usize, const PL: usize> FeedForwardLayer<S, PL> {
    pub fn new(neurons: [FeedForwardNeuron<PL>; S]) -> Self {
        FeedForwardLayer { neurons }
    }

    pub fn feed<F: ActivationFn>(&self, input: &[f16; PL], activation_fn: F) -> [f16; S] {
        let mut list: [f16; S] = [0.0; S];
        for (idx, neuron) in self.neurons.iter().enumerate() {
            list[idx] = neuron.feed(input, activation_fn);
        }
        list
    }
}

pub struct FeedForwardNeuron<const PL: usize> {
    weights: [f16; PL],
    bias: f16,
}

impl<const PL: usize> FeedForwardNeuron<PL> {
    pub fn new(weights: [f16; PL], bias: f16) -> Self {
        FeedForwardNeuron { weights, bias }
    }

    pub fn feed<F: ActivationFn>(&self, input: &[f16; PL], activation_fn: F) -> f16 {
        let mut sum = 0.0;

        for item in input.iter().zip(self.weights).take(PL) {
            sum += item.0 * item.1;
        }

        sum += self.bias;

        activation_fn.activate(sum)
    }
}

#[cfg(test)]
mod tests {
    use crate::{networks::NeuralNetwork, utils::ActFns};

    use super::{FeedForwardLayer, FeedForwardNeuron, StaticFeedForwardNetwork};

    #[test]
    pub fn feed_forward_xor() {
        let mut xor_network = StaticFeedForwardNetwork::new(
            FeedForwardLayer::new([
                FeedForwardNeuron::new([1.0, 1.0], -0.5),
                FeedForwardNeuron::new([-1.0, -1.0], 1.5),
            ]),
            [],
            FeedForwardLayer::new([FeedForwardNeuron::new([1.0, 1.0], -1.5)]),
            ActFns::binary_step(),
        );

        let output = xor_network.feed(&[1.0, 1.0]);
        assert_eq!(output[0], 0.0, "True ^ True != False");

        let output = xor_network.feed(&[0.0, 0.0]);
        assert_eq!(output[0], 0.0, "False ^ False != False");

        let output = xor_network.feed(&[1.0, 0.0]);
        assert_eq!(output[0], 1.0, "False ^ True != True");

        let output = xor_network.feed(&[0.0, 1.0]);
        assert_eq!(output[0], 1.0, "True ^ False != True");
    }
}
