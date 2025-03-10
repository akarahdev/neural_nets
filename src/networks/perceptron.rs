use serde::{Deserialize, Serialize};

use crate::utils::{ActivationFn, Array};

use super::NeuralNetwork;

#[derive(Debug, Clone, PartialEq, Copy, PartialOrd, Serialize, Deserialize)]
pub struct Perceptron<const I: usize, F: ActivationFn> {
    weights: Array<f32, I>,
    bias: f32,
    activation_function: F,
}

impl<const I: usize, F: ActivationFn> Perceptron<I, F> {
    pub fn new(
        weights: impl Into<Array<f32, I>>,
        bias: f32,
        activation_function: F,
    ) -> Perceptron<I, F> {
        Perceptron {
            weights: weights.into(),
            bias,
            activation_function,
        }
    }
}

impl<const I: usize, F: ActivationFn> NeuralNetwork<I, 1> for Perceptron<I, F> {
    fn feed(&mut self, arr: &[f32; I]) -> [f32; 1] {
        let mut sum = 0.0;
        for item in arr.iter().enumerate() {
            sum += *item.1 * self.weights[item.0];
        }
        sum += self.bias;
        [self.activation_function.activate(sum)]
    }
}

#[cfg(test)]
mod tests {
    use crate::{networks::NeuralNetwork, utils::ActFns};

    use super::Perceptron;

    #[test]
    pub fn perceptron_sum() {
        let mut perceptron = Perceptron::new([1.0, 1.0], 0.0, ActFns::linear());
        let output = perceptron.feed(&[1.0, 2.0]);
        assert_eq!(output[0], 3.0);
    }

    #[test]
    pub fn perceptron_doubler() {
        let mut perceptron = Perceptron::new([2.0], 0.0, ActFns::linear());
        let output = perceptron.feed(&[2.0]);
        assert_eq!(output[0], 4.0);
    }

    #[test]
    pub fn perceptron_biases() {
        let mut perceptron = Perceptron::new([1.0, 1.0], 6.0, ActFns::linear());
        let output = perceptron.feed(&[1.0, 2.0]);
        assert_eq!(output[0], 9.0);
    }

    #[test]
    pub fn perceptron_and_gate() {
        let mut perceptron = Perceptron::new([1.0, 1.0], -1.5, ActFns::binary_step());

        let output = perceptron.feed(&[1.0, 0.0]);
        assert_eq!(output[0], 0.0, "True & False != False");

        let output = perceptron.feed(&[0.0, 1.0]);
        assert_eq!(output[0], 0.0, "False & True != False");

        let output = perceptron.feed(&[0.0, 0.0]);
        assert_eq!(output[0], 0.0, "False & False != False");

        let output = perceptron.feed(&[1.0, 1.0]);
        assert_eq!(output[0], 1.0, "True & True != True");
    }

    #[test]
    pub fn perceptron_or_gate() {
        let mut perceptron = Perceptron::new([1.0, 1.0], -0.5, ActFns::binary_step());

        let output = perceptron.feed(&[1.0, 0.0]);
        assert_eq!(output[0], 1.0, "True | False != True");

        let output = perceptron.feed(&[0.0, 1.0]);
        assert_eq!(output[0], 1.0, "False | True != True");

        let output = perceptron.feed(&[1.0, 1.0]);
        assert_eq!(output[0], 1.0, "True | True != True");

        let output = perceptron.feed(&[0.0, 0.0]);
        assert_eq!(output[0], 0.0, "False | False != False");
    }

    #[test]
    pub fn save_perceptron() {
        let perceptron = Perceptron::new([1.0, 1.0], -0.5, ActFns::binary_step());
        let string = serde_json::to_string(&perceptron).unwrap();
        eprintln!("{}", string);
        let new_perceptron = serde_json::from_str(&string).unwrap();
        assert_eq!(perceptron, new_perceptron);
    }
}
