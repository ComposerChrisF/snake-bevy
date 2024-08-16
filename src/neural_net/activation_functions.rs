use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};


#[derive(Copy, Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub enum ActivationFunction {
    None,       // f(x) = x, i.e. Linear
    Sigmoid,    // f(x) = 1.0 / (1.0 + exp(-x));                                f(4) = 0.982013790037908
    ReLU,       // f(x) = if x > 0 { x } else { 0.0 };                          f(1) = 1.0
    LReLU,      // f(x) = if x > 0 { x } else ( 0.1 * x );                      f(1) = 1.0
    Tanh,       // f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)); tanh(2) = 0.964027580075817
}

impl ActivationFunction {
    pub fn linear( x: f32) -> f32 { x }
    pub fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
    pub fn relu(   x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }
    pub fn lrelu(  x: f32) -> f32 { if x >= 0.0 { x } else { 0.1 * x } }
    pub fn tanh(   x: f32) -> f32 { x.tanh() }

    pub fn apply(&self, x: f32) -> f32 {
        match self {
            ActivationFunction::None    => Self::linear(x),
            ActivationFunction::Sigmoid => Self::sigmoid(x),
            ActivationFunction::ReLU    => Self::relu(x),
            ActivationFunction::LReLU   => Self::lrelu(x),
            ActivationFunction::Tanh    => Self::tanh(x),
        }
    }

    pub fn get_neutral_value(&self) -> f32 {
        match self {
            ActivationFunction::None    => 1.0,
            ActivationFunction::Sigmoid => 4.0,     // Sigmoid(4.0) = 0.982013790037908
            ActivationFunction::ReLU    => 1.0,
            ActivationFunction::LReLU   => 1.0,
            ActivationFunction::Tanh    => 2.37,    // tanh(2.37) = 0.982674112430374
        }
    }

    pub fn choose_random() -> Self {
        match thread_rng().gen_range(0..5) {
            0 => ActivationFunction::None,
            1 => ActivationFunction::Sigmoid,
            2 => ActivationFunction::ReLU,
            3 => ActivationFunction::LReLU,
            4 => ActivationFunction::Tanh,
            _ => panic!("Unexpected choice for choose_random()")
        }
    }
}



#[cfg(test)]
mod tests {
    use super::ActivationFunction;

    #[test]
    fn test_funtions() {
        for (i, &x) in [-2.0, 1.0, 0.0, 123.456, -3.1415926, -0.000001, 4.0].iter().enumerate() {
            assert_eq!( x, ActivationFunction::linear( x));
            assert_eq!(-x, ActivationFunction::linear(-x));

            assert_eq!(x.abs(), ActivationFunction::relu( x.abs()));
            assert_eq!(0.0,     ActivationFunction::relu(-x.abs()));
            
            assert_eq!(x.abs(),        ActivationFunction::lrelu( x.abs()));
            assert_eq!(-0.1 * x.abs(), ActivationFunction::lrelu(-x.abs()));
            
            // f(x) = 1.0 / (1.0 + exp(-x));
            let sig = [0.1192, 0.7310, 0.5, 1.0, 0.0414, 0.5, 0.9820];
            assert!(almost_eq(sig[i],       ActivationFunction::sigmoid( x)));
            assert!(almost_eq(1.0 - sig[i], ActivationFunction::sigmoid(-x)));

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
        for af in [ActivationFunction::Sigmoid, ActivationFunction::Tanh] {
            let v = af.apply(af.get_neutral_value());
            assert!((0.982 - v).abs() < 0.001);
        }
    }

    #[test]
    fn test_choose() {
        let mut found = [false; 5];
        for _ in 0..1000 {
            let i = match ActivationFunction::choose_random() {
                ActivationFunction::None    => 0,
                ActivationFunction::Sigmoid => 1,
                ActivationFunction::ReLU    => 2,
                ActivationFunction::LReLU   => 3,
                ActivationFunction::Tanh    => 4,
            };
            found[i] = true;
        }
        assert!(found.iter().all(|&b| b));
    }
}