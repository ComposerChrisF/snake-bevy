use std::fmt::Display;


#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Layer {
    Input,
    Hidden(u16),
    Output,
    Unreachable,
}

impl Layer {
    pub fn to_number(self) -> usize {
        match self {
            Layer::Input       => 0,
            Layer::Hidden(i)   => i as usize + 1,
            Layer::Output      => u16::MAX as usize + 1,
            Layer::Unreachable => u16::MAX as usize + 2,
        }
    }

    pub fn comes_before(self, other: Layer) -> Option<bool> {
        if self == Layer::Unreachable {
            return if other == Layer::Output { Some(true) } else { None };
        } else if other == Layer::Unreachable { 
            return if self == Layer::Input { Some(true) } else { None }; 
        }
        Some(self.to_number() < other.to_number())
    }
}


impl Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Layer::Input       => write!(f, "Layer::Input"),
            Layer::Hidden(i)   => write!(f, "Layer::Hidden({i})"),
            Layer::Output      => write!(f, "Layer::Output"),
            Layer::Unreachable => write!(f, "Layer::Unreachable"),
        }
    }
}