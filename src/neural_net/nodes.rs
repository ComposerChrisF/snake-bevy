use std::{fmt, sync::atomic::{AtomicUsize, Ordering}};
use super::{activation_functions::ActivationFunction, layers::Layer, nets::{ConnectionIndex, NodeIndex}};



static NODE_ID_NEXT: AtomicUsize = AtomicUsize::new(1);

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct NodeId(usize);

impl NodeId {
    pub fn new_unique() -> NodeId {
        NodeId(NODE_ID_NEXT.fetch_add(1, Ordering::SeqCst))
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NodeId({})", self.0)
    }
}
impl fmt::Debug for NodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NodeId({})", self.0)
    }
}


#[derive(Clone, Debug)]
pub struct Node {
    pub index: NodeIndex,
    pub id: NodeId,
    pub activation_function: ActivationFunction,
    pub layer: Layer,
    pub(super) input_connections: Vec<ConnectionIndex>,
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
        self.index == other.index
    }
}
impl Eq for Node {}
impl std::hash::Hash for Node {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}
