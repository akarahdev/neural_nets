use std::f16::consts::{E, PI};

pub struct ActFns;

impl ActFns {
    pub const fn linear() -> fn(f16) -> f16 {
        |x| x
    }

    pub const fn binary_step() -> fn(f16) -> f16 {
        |x| if x >= 0.0 { 1.0 } else { 0.0 }
    }

    pub const fn sigmoid() -> fn(f16) -> f16 {
        |x| 1.0 / (1.0 + E.powf(-x))
    }

    pub const fn tanh() -> fn(f16) -> f16 {
        |x| x.tanh()
    }

    pub const fn relu() -> fn(f16) -> f16 {
        |x| x.max(0.0)
    }

    pub const fn leaky_relu(factor: f16) -> impl Fn(f16) -> f16 {
        move |x: f16| x.max(x * -factor)
    }

    pub const fn elu() -> impl Fn(f16) -> f16 {
        move |x: f16| {
            if x >= 0.0 { x } else { E.powf(x) - 1.0 }
        }
    }

    pub const fn param_elu(a: f16) -> impl Fn(f16) -> f16 {
        move |x: f16| {
            if x >= 0.0 { x } else { a * (E.powf(x) - 1.0) }
        }
    }

    pub const fn swish() -> impl Fn(f16) -> f16 {
        move |x: f16| x * ActFns::sigmoid()(x)
    }

    pub const fn gelu() -> impl Fn(f16) -> f16 {
        move |x: f16| 0.5 * x * (1.0 + (2.0 / PI).sqrt() * (x + 0.44715 * x.powi(3)).tanh())
    }
}
