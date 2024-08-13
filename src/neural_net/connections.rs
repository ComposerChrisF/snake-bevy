use std::{fmt, sync::atomic::{AtomicUsize, Ordering}};

use super::nets::{ConnectionIndex, NodeIndex};


static CONNECTION_ID_NEXT: AtomicUsize = AtomicUsize::new(1);

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ConnectionId(usize);

impl ConnectionId {
    pub fn new_unique() -> ConnectionId {
        ConnectionId(CONNECTION_ID_NEXT.fetch_add(1, Ordering::SeqCst))
    }
}

impl fmt::Display for ConnectionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConnectionId({})", self.0)
    }
}
impl fmt::Debug for ConnectionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConnectionId({})", self.0)
    }
}


#[derive(Clone, Debug)]
pub struct Connection {
    pub index: ConnectionIndex,
    pub id: ConnectionId,
    pub input_node:  NodeIndex,
    pub output_node: NodeIndex,
    pub weight: f32,
    pub is_enabled: bool,
}

