use super::nets::Net;



pub struct Population {
    pub nets: Vec<Net>,
}

impl Population {
    pub fn new() -> Self {
        Self {
            nets: Vec::<Net>::new(),
        }
    }

    pub fn create_next_generation(&self) -> Population {
        todo!()
    }
}