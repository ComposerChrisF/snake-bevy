use std::sync::atomic::{AtomicUsize, Ordering};

use super::nodes::NodeId;


static CONNECTION_ID_NEXT: AtomicUsize = AtomicUsize::new(1);

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct ConnectionId(usize);

impl ConnectionId {
    pub fn new_unique() -> ConnectionId {
        ConnectionId(CONNECTION_ID_NEXT.fetch_add(1, Ordering::SeqCst))
    }
}



#[derive(Debug)]
pub struct Connection {
    pub input_node:  NodeId,
    pub output_node: NodeId,
    pub weight: f32,
    pub is_enabled: bool,
    pub id: ConnectionId,
}

