use std::sync::atomic::{AtomicUsize, Ordering};
use super::{activation_functions::ActivationFunction, connections::ConnectionId, layers::Layer};



static NODE_ID_NEXT: AtomicUsize = AtomicUsize::new(1);

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct NodeId(usize);

impl NodeId {
    pub fn new_unique() -> NodeId {
        NodeId(NODE_ID_NEXT.fetch_add(1, Ordering::SeqCst))
    }
}



#[derive(Debug)]
pub struct Node {
    pub activation_function: ActivationFunction,
    pub layer: Layer,
    pub id: NodeId,
    pub(super) input_connections: Vec<ConnectionId>,
    pub value: f32,
}

impl Node {
    pub fn apply_activation_function(&self, input_sum: f32) -> f32 {
        // NOTE: Traditional "bias" of a node is accomplished by having one of the input nodes 
        // always having a value of 1.0, thus the connection weight creates a bias.
        self.activation_function.apply(input_sum)
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for Node {}
impl std::hash::Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.id.0.hash(state);
    }
}
