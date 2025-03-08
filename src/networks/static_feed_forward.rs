use super::NeuralNetwork;

pub struct StaticFeedForwardNetwork<
    const I: usize,
    const HN: usize,
    const IHLC: usize,
    const O: usize,
    F: Fn(f16) -> f16,
> {
    first_hidden_layer: FeedForwardLayer<HN, I, F>,
    intermediate_hidden_layers: [FeedForwardLayer<HN, HN, F>; IHLC],
    output_layer: FeedForwardLayer<O, HN, F>,
}

impl<const I: usize, const HN: usize, const IHLC: usize, const O: usize, F: Fn(f16) -> f16>
    StaticFeedForwardNetwork<I, HN, IHLC, O, F>
{
    pub fn new(
        first_hidden_layer: FeedForwardLayer<HN, I, F>,
        intermediate_hidden_layers: [FeedForwardLayer<HN, HN, F>; IHLC],
        output_layer: FeedForwardLayer<O, HN, F>,
    ) -> Self {
        StaticFeedForwardNetwork {
            first_hidden_layer,
            intermediate_hidden_layers,
            output_layer,
        }
    }
}

impl<const I: usize, const HN: usize, const IHLC: usize, const O: usize, F: Fn(f16) -> f16>
    NeuralNetwork<I, O> for StaticFeedForwardNetwork<I, HN, IHLC, O, F>
{
    fn feed(&mut self, arr: &[f16; I]) -> [f16; O] {
        let mut array = self.first_hidden_layer.feed(arr);

        for intermediate in &self.intermediate_hidden_layers {
            array = intermediate.feed(&array);
        }

        self.output_layer.feed(&array)
    }
}

pub struct FeedForwardLayer<const S: usize, const PL: usize, F: Fn(f16) -> f16> {
    neurons: [FeedForwardNeuron<PL, F>; S],
}

impl<const S: usize, const PL: usize, F: Fn(f16) -> f16> FeedForwardLayer<S, PL, F> {
    pub fn new(neurons: [FeedForwardNeuron<PL, F>; S]) -> Self {
        FeedForwardLayer { neurons }
    }

    pub fn feed(&self, input: &[f16; PL]) -> [f16; S] {
        let mut list: [f16; S] = [0.0; S];
        for (idx, neuron) in self.neurons.iter().enumerate() {
            list[idx] = neuron.feed(input);
        }
        list
    }
}

pub struct FeedForwardNeuron<const PL: usize, F: Fn(f16) -> f16> {
    weights: [f16; PL],
    bias: f16,
    activation_function: F,
}

impl<const PL: usize, F: Fn(f16) -> f16> FeedForwardNeuron<PL, F> {
    pub fn new(weights: [f16; PL], bias: f16, activation_function: F) -> Self {
        FeedForwardNeuron {
            weights,
            bias,
            activation_function,
        }
    }

    pub fn feed(&self, input: &[f16; PL]) -> f16 {
        let mut sum = 0.0;

        for item in input.iter().zip(self.weights).take(PL) {
            sum += item.0 * item.1;
        }

        sum += self.bias;

        (self.activation_function)(sum)
    }
}

#[cfg(test)]
mod tests {
    use crate::networks::NeuralNetwork;

    use super::{FeedForwardLayer, FeedForwardNeuron, StaticFeedForwardNetwork};

    #[test]
    pub fn feed_forward_xor() {
        let step_function = |x: f16| if x >= 0.0 { 1.0 } else { 0.0 };

        let mut xor_network = StaticFeedForwardNetwork::new(
            FeedForwardLayer::new([
                FeedForwardNeuron::new([1.0, 1.0], -0.5, step_function),
                FeedForwardNeuron::new([-1.0, -1.0], 1.5, step_function),
            ]),
            [],
            FeedForwardLayer::new([FeedForwardNeuron::new([1.0, 1.0], -1.5, step_function)]),
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
