use std::{
    f32::consts::{E, PI},
    fmt::Debug,
};

pub trait ActivationFn: Clone + Copy + std::fmt::Debug {
    fn activate(&self, x: f32) -> f32;
}

#[derive(Debug, Clone, Copy)]
pub struct Linear;
impl ActivationFn for Linear {
    fn activate(&self, x: f32) -> f32 {
        x
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BinaryStep;
impl ActivationFn for BinaryStep {
    fn activate(&self, x: f32) -> f32 {
        if x >= 0.0 { 1.0 } else { 0.0 }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Sigmoid;
impl ActivationFn for Sigmoid {
    fn activate(&self, x: f32) -> f32 {
        1.0 / (1.0 + E.powf(-x))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Tanh;
impl ActivationFn for Tanh {
    fn activate(&self, x: f32) -> f32 {
        x.tanh()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Relu;
impl ActivationFn for Relu {
    fn activate(&self, x: f32) -> f32 {
        x.max(0.0)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct LeakyRelu {
    pub factor: f32,
}
impl ActivationFn for LeakyRelu {
    fn activate(&self, x: f32) -> f32 {
        x.max(x * -self.factor)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Elu {
    pub factor: f32,
}
impl ActivationFn for Elu {
    fn activate(&self, x: f32) -> f32 {
        if x >= 0.0 {
            x
        } else {
            self.factor * (E.powf(x) - 1.0)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Swish;
impl ActivationFn for Swish {
    fn activate(&self, x: f32) -> f32 {
        let sigmoid = Sigmoid;
        x * sigmoid.activate(x)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Gelu;
impl ActivationFn for Gelu {
    fn activate(&self, x: f32) -> f32 {
        0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.44715 * x.powi(3))).tanh())
    }
}

pub struct ActFns;

impl ActFns {
    pub fn linear() -> impl ActivationFn {
        Linear
    }

    pub fn binary_step() -> impl ActivationFn {
        BinaryStep
    }

    pub fn sigmoid() -> impl ActivationFn {
        Sigmoid
    }

    pub fn tanh() -> impl ActivationFn {
        Tanh
    }

    pub fn relu() -> impl ActivationFn {
        Relu
    }

    pub fn leaky_relu(factor: f32) -> impl ActivationFn {
        LeakyRelu { factor }
    }

    pub fn elu() -> impl ActivationFn {
        Elu { factor: 1.0 }
    }

    pub fn param_elu(factor: f32) -> impl ActivationFn {
        Elu { factor }
    }

    pub fn swish() -> impl ActivationFn {
        Swish
    }

    pub fn gelu() -> impl ActivationFn {
        Gelu
    }
}
