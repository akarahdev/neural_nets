use serde::{Deserialize, Serialize};

use crate::utils::{ActivationFn, Array};

use super::NeuralNetwork;

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
pub struct StaticFeedForwardNetwork<
    const I: usize,
    const HN: usize,
    const IHLC: usize,
    const O: usize,
    F: ActivationFn,
> {
    first_hidden_layer: FeedForwardLayer<HN, I>,
    intermediate_hidden_layers: Array<FeedForwardLayer<HN, HN>, IHLC>,
    output_layer: FeedForwardLayer<O, HN>,
    #[serde(skip)]
    activation_function: F,
}

impl<const I: usize, const HN: usize, const IHLC: usize, const O: usize, F: ActivationFn>
    StaticFeedForwardNetwork<I, HN, IHLC, O, F>
{
    pub fn new(
        first_hidden_layer: FeedForwardLayer<HN, I>,
        intermediate_hidden_layers: impl Into<Array<FeedForwardLayer<HN, HN>, IHLC>>,
        output_layer: FeedForwardLayer<O, HN>,
        activation_function: F,
    ) -> Self {
        StaticFeedForwardNetwork {
            first_hidden_layer,
            intermediate_hidden_layers: intermediate_hidden_layers.into(),
            output_layer,
            activation_function,
        }
    }
}

impl<const I: usize, const HN: usize, const IHLC: usize, const O: usize, F: ActivationFn>
    NeuralNetwork<I, O> for StaticFeedForwardNetwork<I, HN, IHLC, O, F>
{
    fn feed(&mut self, arr: &[f32; I]) -> [f32; O] {
        let mut array = self.first_hidden_layer.feed(arr, self.activation_function);

        for intermediate in self.intermediate_hidden_layers.iter() {
            array = intermediate.feed(&array, self.activation_function);
        }

        self.output_layer.feed(&array, self.activation_function)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
pub struct FeedForwardLayer<const S: usize, const PL: usize> {
    neurons: Array<FeedForwardNeuron<PL>, S>,
}

impl<const S: usize, const PL: usize> FeedForwardLayer<S, PL> {
    pub fn new(neurons: impl Into<Array<FeedForwardNeuron<PL>, S>>) -> Self {
        FeedForwardLayer {
            neurons: neurons.into(),
        }
    }

    pub fn feed<F: ActivationFn>(&self, input: &[f32; PL], activation_fn: F) -> [f32; S] {
        let mut list: [f32; S] = [0.0; S];
        for (idx, neuron) in self.neurons.iter().enumerate() {
            list[idx] = neuron.feed(input, activation_fn);
        }
        list
    }
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Default, Serialize, Deserialize)]
pub struct FeedForwardNeuron<const PL: usize> {
    weights: Array<f32, PL>,
    bias: f32,
}

impl<const PL: usize> FeedForwardNeuron<PL> {
    pub fn new(weights: impl Into<Array<f32, PL>>, bias: f32) -> Self {
        FeedForwardNeuron {
            weights: weights.into(),
            bias,
        }
    }

    pub fn feed<F: ActivationFn>(&self, input: &[f32; PL], activation_fn: F) -> f32 {
        let mut sum = 0.0;

        for item in input.iter().zip(self.weights.iter()).take(PL) {
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

    #[test]
    pub fn save_xor_network() {
        let xor_network = StaticFeedForwardNetwork::new(
            FeedForwardLayer::new([
                FeedForwardNeuron::new([1.0, 1.0], -0.5),
                FeedForwardNeuron::new([-1.0, -1.0], 1.5),
            ]),
            [],
            FeedForwardLayer::new([FeedForwardNeuron::new([1.0, 1.0], -1.5)]),
            ActFns::binary_step(),
        );
        let string = serde_json::to_string(&xor_network).unwrap();
        eprintln!("{}", string);
        let new_xor_network = serde_json::from_str(&string).unwrap();
        assert_eq!(xor_network, new_xor_network);
    }
}
