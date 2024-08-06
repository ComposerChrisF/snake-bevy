
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Layer {
    Input,
    Hidden(u16),
    Output,
}

impl Layer {
    pub fn to_number(self) -> usize {
        match self {
            Layer::Input     => 0,
            Layer::Hidden(i) => i as usize + 1,
            Layer::Output    => u16::MAX as usize + 1,
        }
    }

    pub fn comes_before(self, other: Layer) -> bool {
        self.to_number() < other.to_number()
    }
}
