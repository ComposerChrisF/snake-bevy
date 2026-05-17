use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};


#[derive(Copy, Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum ActivationFunction {
    None,       // f(x) = x, i.e. Linear
    ModSigmoid, // f(x) = 1.0 / (1.0 + exp(-4.9 * x));                          f(1) = 0.992608459
    Sigmoid,    // f(x) = 1.0 / (1.0 + exp(-x));                                f(4) = 0.982013790037908
    ReLU,       // f(x) = if x > 0 { x } else { 0.0 };                          f(1) = 1.0
    LReLU,      // f(x) = if x > 0 { x } else ( 0.1 * x );                      f(1) = 1.0
    Tanh,       // f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)); tanh(2) = 0.964027580075817
}

impl ActivationFunction {
    pub fn linear(     x: f32) -> f32 { x }
    pub fn mod_sigmoid(x: f32) -> f32 { (1.0 / (1.0 + (-4.9 * x).exp())) * 2.0 - 1.0 }    // Pg. 112, Section 4.1
    pub fn sigmoid(    x: f32) -> f32 { (1.0 / (1.0 + (-x).exp())) * 2.0 - 1.0 }
    pub fn relu(       x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }
    pub fn lrelu(      x: f32) -> f32 { if x >= 0.0 { x } else { 0.1 * x } }
    pub fn tanh(       x: f32) -> f32 { x.tanh() }

    pub fn apply(&self, x: f32) -> f32 {
        match self {
            ActivationFunction::None       => Self::linear(x),
            ActivationFunction::ModSigmoid => Self::mod_sigmoid(x),
            ActivationFunction::Sigmoid    => Self::sigmoid(x),
            ActivationFunction::ReLU       => Self::relu(x),
            ActivationFunction::LReLU      => Self::lrelu(x),
            ActivationFunction::Tanh       => Self::tanh(x),
        }
    }

    pub fn get_neutral_value(&self) -> f32 {
        match self {
            ActivationFunction::None       => 1.0,
            ActivationFunction::ModSigmoid => 1.0,     // Mod_Sigmoid(1.0) = 0.992608459
            ActivationFunction::Sigmoid    => 4.0,     // Sigmoid(4.0) = 0.982013790037908
            ActivationFunction::ReLU       => 1.0,
            ActivationFunction::LReLU      => 1.0,
            ActivationFunction::Tanh       => 2.37,    // tanh(2.37) = 0.982674112430374
        }
    }

    pub fn choose_random() -> Self {
        match thread_rng().gen_range(0..6) {
            0 => ActivationFunction::None,
            1 => ActivationFunction::ModSigmoid,
            2 => ActivationFunction::Sigmoid,
            3 => ActivationFunction::ReLU,
            4 => ActivationFunction::LReLU,
            5 => ActivationFunction::Tanh,
            _ => panic!("Unexpected choice for choose_random()")
        }
    }
}



#[cfg(test)]
mod tests {
    use super::ActivationFunction;

    #[test]
    #[allow(clippy::approx_constant)] // -3.1415926 is a test input, not an approximation of -PI.
    fn test_funtions() {
        for (i, &x) in [-2.0, 1.0, 0.0, 123.456, -3.1415926, -0.000001, 4.0].iter().enumerate() {
            assert_eq!( x, ActivationFunction::linear( x));
            assert_eq!(-x, ActivationFunction::linear(-x));

            assert_eq!(x.abs(), ActivationFunction::relu( x.abs()));
            assert_eq!(0.0,     ActivationFunction::relu(-x.abs()));
            
            assert_eq!(x.abs(),        ActivationFunction::lrelu( x.abs()));
            assert_eq!(-0.1 * x.abs(), ActivationFunction::lrelu(-x.abs()));
            
            // f(x) = 1.0 / (1.0 + exp(-x));
            let sig = [-0.7616, 0.4621, 0.0, 1.0, -0.9172, 0.0, 0.9640];
            assert!(almost_eq(sig[i],  ActivationFunction::sigmoid( x)));
            assert!(almost_eq(-sig[i], ActivationFunction::sigmoid(-x)));

            // TODO: Add Modified Sigmoid

            let tanh = [-0.9640, 0.7616, 0.0, 1.0, -0.9963, 0.0, 0.9993];
            assert!(almost_eq(tanh[i],  ActivationFunction::tanh( x)));
            assert!(almost_eq(-tanh[i], ActivationFunction::tanh(-x)));
        }
    }

    fn almost_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < 0.0001
    }

    #[test]
    fn test_neutral_values() {
        for af in [ActivationFunction::None, ActivationFunction::ReLU, ActivationFunction::LReLU] {
            assert_eq!(1.0, af.apply(af.get_neutral_value()));
        }
        for af in [ActivationFunction::Sigmoid] {
            let v = af.apply(af.get_neutral_value());
            assert!((0.964 - v).abs() < 0.001);
        }
        for af in [ActivationFunction::Tanh] {
            let v = af.apply(af.get_neutral_value());
            assert!((0.982 - v).abs() < 0.001);
        }
    }

    #[test]
    fn test_choose() {
        let mut found = [false; 6];
        for _ in 0..1000 {
            let i = match ActivationFunction::choose_random() {
                ActivationFunction::None       => 0,
                ActivationFunction::ModSigmoid => 1,
                ActivationFunction::Sigmoid    => 2,
                ActivationFunction::ReLU       => 3,
                ActivationFunction::LReLU      => 4,
                ActivationFunction::Tanh       => 5,
            };
            found[i] = true;
        }
        assert!(found.iter().all(|&b| b));
    }
}