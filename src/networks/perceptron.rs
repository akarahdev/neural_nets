use super::NeuralNetwork;

#[derive(Debug, Clone, PartialEq)]
pub struct Perceptron<const I: usize, F: Fn(f16) -> f16> {
    weights: [f16; I],
    bias: f16,
    activation_function: F,
}

impl<const I: usize, F: Fn(f16) -> f16> Perceptron<I, F> {
    pub fn new(weights: [f16; I], bias: f16, activation_function: F) -> Perceptron<I, F> {
        Perceptron {
            weights,
            bias,
            activation_function,
        }
    }
}

impl<const I: usize, F: Fn(f16) -> f16> NeuralNetwork<I, 1, F> for Perceptron<I, F> {
    fn feed(&mut self, arr: &[f16; I]) -> [f16; 1] {
        let mut sum = 0.0;
        for item in arr.iter().enumerate() {
            sum += *item.1 * self.weights[item.0];
        }
        sum += self.bias;
        [(self.activation_function)(sum)]
    }

    const FLATTENED_SIZE: usize = I + 1;

    fn flatten(&self) -> [f16; Self::FLATTENED_SIZE] {
        let mut arr = [0.0; Self::FLATTENED_SIZE];
        for item in self.weights.iter().enumerate() {
            arr[item.0] = *item.1;
        }
        arr[arr.len() - 1] = self.bias;
        arr
    }

    fn unflatten(flattened: [f16; Self::FLATTENED_SIZE], activation_function: F) -> Self
    where
        Self: Sized,
    {
        let weights_len = flattened.len() - 1;
        let bias = flattened[weights_len];
        let mut weights = [0.0; I];
        weights[..weights_len].copy_from_slice(&flattened[..weights_len]);
        Perceptron {
            bias,
            weights,
            activation_function,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::networks::NeuralNetwork;

    use super::Perceptron;

    #[test]
    pub fn perceptron_sum() {
        let mut perceptron = Perceptron::new([1.0, 1.0], 0.0, |x| x);
        let output = perceptron.feed(&[1.0, 2.0]);
        assert_eq!(output[0], 3.0);
    }

    #[test]
    pub fn perceptron_doubler() {
        let mut perceptron = Perceptron::new([2.0], 0.0, |x| x);
        let output = perceptron.feed(&[2.0]);
        assert_eq!(output[0], 4.0);
    }

    #[test]
    pub fn perceptron_biases() {
        let mut perceptron = Perceptron::new([1.0, 1.0], 6.0, |x| x);
        let output = perceptron.feed(&[1.0, 2.0]);
        assert_eq!(output[0], 9.0);
    }

    #[test]
    pub fn perceptron_and_gate() {
        let mut perceptron = Perceptron::new([1.0, 1.0], -1.5, |x| if x > 0.0 { 1.0 } else { 0.0 });

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
        let mut perceptron = Perceptron::new([1.0, 1.0], -0.5, |x| if x > 0.0 { 1.0 } else { 0.0 });

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
    fn perceptron_reflatten() {
        let closure = |x| if x > 0.0 { 1.0 } else { 0.0 };
        let perceptron = Perceptron::new([1.0, 1.0], -1.5, closure);

        let flattened = perceptron.flatten();
        let unflattened = Perceptron::unflatten(flattened, closure);

        assert_eq!(perceptron.weights, unflattened.weights);
        assert_eq!(perceptron.bias, unflattened.bias);
    }
}
