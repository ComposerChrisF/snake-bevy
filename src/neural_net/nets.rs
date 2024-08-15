
use std::{fmt, sync::atomic::{AtomicUsize, Ordering}};
use bevy::utils::hashbrown::{HashMap, HashSet};
use log::{debug, trace};
use rand::{thread_rng, Rng, prelude::SliceRandom};

use super::{activation_functions::ActivationFunction, connections::{Connection, ConnectionId}, layers::Layer, nodes::{Node, NodeId}, populations::FitnessInfo};

fn is_none_or<T, U>(val: Option<T>, f: U) -> bool 
    where T: Sized, U: FnOnce(T) -> bool {
    match val {
        None => true,
        Some(v) => f(v),
    }
}


static NET_ID_NEXT: AtomicUsize = AtomicUsize::new(1);

/// The NetId uniquely identifies an instance of a Net.  Used for debug checks to ensure node and
/// connection indexes can only be used for the Net that generated them.
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct NetId(usize);

impl NetId {
    pub fn new_unique() -> NetId {
        NetId(NET_ID_NEXT.fetch_add(1, Ordering::SeqCst))
    }
}

impl fmt::Display for NetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NetId({})", self.0)
    }
}
impl fmt::Debug for NetId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NetId({})", self.0)
    }
}



#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct NodeIndex(NetId, usize);

impl fmt::Display for NodeIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NodeIndex({},{})", self.0, self.1)
    }
}
impl fmt::Debug for NodeIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NodeIndex({},{})", self.0, self.1)
    }
}


#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct ConnectionIndex(NetId, usize);

impl fmt::Display for ConnectionIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConnectionIndex({},{})", self.0, self.1)
    }
}
impl fmt::Debug for ConnectionIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ConnectionIndex({},{})", self.0, self.1)
    }
}




#[derive(Clone, Debug, PartialEq)]
pub struct NetParams {
    pub input_count: usize,
    pub input_names: Option<&'static[&'static str]>,
    pub output_count: usize,
    pub output_names: Option<&'static[&'static str]>,
}

impl NetParams {
    fn from_size(input_count: usize, output_count: usize) -> Self {
        NetParams {
            input_count,
            input_names: None,
            output_count,
            output_names: None,
        }
    }
}


#[derive(Clone, Debug, PartialEq)]
pub struct MutationParams {
    pub prob_mutate_activation_function_of_node: f64,
    pub prob_mutate_weight: f64,
    pub max_weight_change_magnitude: f32,
    pub prob_toggle_enabled: f64,
    pub prob_remove_connection: f64,
    pub prob_add_connection: f64,
    pub prob_remove_node: f64,
    pub prob_add_node: f64,
}



#[derive(Clone, Debug)]
pub struct Net<F> where F: FitnessInfo {
    pub id: NetId,
    pub net_params: NetParams,
    nodes: Vec<Node>,
    map_node_id_to_index: HashMap<NodeId, NodeIndex>,
    connections: Vec<Connection>,
    map_connection_id_to_index: HashMap<ConnectionId, ConnectionIndex>,
    pub fitness_info: F,
    pub is_evaluation_order_up_to_date: bool,
    node_order_list: Vec<NodeIndex>,
}

impl <F> Net<F> where F: FitnessInfo {
    pub fn get_node(&self, i: NodeIndex) -> &Node {
        assert_eq!(i.0, self.id);
        &self.nodes[i.1]
    }
    pub fn get_node_mut(&mut self, i: NodeIndex) -> &mut Node {
        assert_eq!(i.0, self.id);
        &mut self.nodes[i.1]
    }
    pub fn get_connection(&self, i: ConnectionIndex) -> &Connection {
        assert_eq!(i.0, self.id);
        &self.connections[i.1]
    }
    pub fn get_connection_mut(&mut self, i: ConnectionIndex) -> &mut Connection {
        assert_eq!(i.0, self.id);
        &mut self.connections[i.1]
    }

    fn add_node(&mut self, id: Option<NodeId>, activation_function: ActivationFunction, layer: Option<Layer>, value: f32) -> NodeIndex {
        let index = NodeIndex(self.id, self.nodes.len());
        let node = Node {
            index,
            id: id.unwrap_or_else(NodeId::new_unique),
            activation_function,
            layer: layer.unwrap_or(Layer::Hidden(1)),
            input_connections: Vec::new(),
            value,
        };
        self.map_node_id_to_index.insert(node.id, node.index);
        self.nodes.push(node);
        index
    }

    fn add_connection(&mut self, id: Option<ConnectionId>, weight: f32, is_enabled: bool, input_node: NodeIndex, output_node: NodeIndex) -> ConnectionIndex {
        assert_eq!(self.id,  input_node.0);     assert!( input_node.1 < self.nodes.len());
        assert_eq!(self.id, output_node.0);     assert!(output_node.1 < self.nodes.len());
        let index = ConnectionIndex(self.id, self.connections.len());
        let connection = Connection {
            index,
            id: id.unwrap_or_else(ConnectionId::new_unique),
            weight,
            is_enabled,
            input_node,
            output_node,
        };
        let connection_index = connection.index;
        self.map_connection_id_to_index.insert(connection.id, connection.index);
        self.connections.push(connection);
        let node_to_upate = &mut self.nodes[output_node.1];
        node_to_upate.input_connections.push(connection_index);
        index
    }

    pub fn new(net_params: NetParams) -> Self {
        let capacity = net_params.input_count + net_params.output_count;
        let mut net = Self {
            id: NetId::new_unique(),
            net_params,
            nodes: Vec::<Node>::with_capacity(capacity),
            map_node_id_to_index: HashMap::<NodeId, NodeIndex>::with_capacity(capacity),
            connections: Vec::with_capacity(capacity),
            map_connection_id_to_index: HashMap::with_capacity(capacity),
            fitness_info: F::default(),
            is_evaluation_order_up_to_date: false,
            node_order_list: Vec::with_capacity(capacity),
        };

        // NOTE: We add them specifically in this order, so that we can
        // rely on 0..input_count being the inputs, and 
        // input_count..(input_count+output_count) being the outputs!!!
        for _ in 0..net.net_params.input_count { 
            net.add_node(None, ActivationFunction::None, Some(Layer::Input), 0.0);
        }
        for _ in 0..net.net_params.output_count {
            net.add_node(None, ActivationFunction::Sigmoid, Some(Layer::Output), 0.0);
        }
        net
    }

    // NOTE: If we recursively traverse the network *once*, we can build the order that the network
    // needs to be evaluated in!  Then, to evaluate, we simply linearly replay the eval list--no
    // recursion or "node_has_been_evaluated" logic needed!
    fn build_evaluation_order(&mut self) {
        if self.is_evaluation_order_up_to_date { return; }
        let mut node_has_been_evaluated = vec![false; self.nodes.len()];
        let mut node_order_list = Vec::<NodeIndex>::with_capacity(self.nodes.len());
        let mut layer_list = HashMap::<NodeIndex, u16>::with_capacity(self.nodes.len());

        // Initialize inputs to layer 0
        for node in self.nodes.iter() {
            if node.layer == Layer::Input {
                layer_list.insert(node.index, 0);
            }
        }

        // Figure out the order to compute nodes and what layer the nodes belong to.
        for node_index in self.nodes.iter().filter_map(|n| if n.layer == Layer::Output { Some(n.index) } else { None }) {
            self.build_evaluation_order_recurse(0, &mut node_order_list, &mut node_has_been_evaluated, node_index);
            self.build_layer_order_recurse(0, &mut layer_list, node_index);
        }

        // Copy layer number to nodes.layer from layer_list[node_index], fixing the output layers,
        // which, by convention, the output layer nodes are all the highest-valued layer.
        trace!("{layer_list:?}");
        for node in self.nodes.iter_mut() {
            let layer = layer_list.get(&node.index);
            match node.layer {
                Layer::Input => assert!(is_none_or(layer, |&layer| layer == 0)),
                Layer::Output => {
                    debug!("assert: layer={layer:?}, conn_count={}", node.input_connections.len());
                    assert!(is_none_or(layer, |&layer| layer > 0 || node.input_connections.is_empty()));
                },
                _ => node.layer = if let Some(&layer) = layer {
                    //assert!(layer > 0);       // TODO: Re-enable?
                    Layer::Hidden(layer)
                } else {
                    Layer::Unreachable
                },
            }
        }
        drop(layer_list);
        drop(node_has_been_evaluated);
        self.node_order_list = node_order_list;
        self.is_evaluation_order_up_to_date = true;
    }

    // We figure out the order to compute all output nodes by recursively seeking the values 
    // of all required inputs for each output node.  Note that the last output_count nodes 
    // are the output nodes, so we only have to evalute them.  Thus, we might skip computation
    // of nodes that don't (eventually) connect to any output.
    fn build_evaluation_order_recurse(&self, recursion: usize, node_order_list: &mut Vec<NodeIndex>, node_has_been_evaluated: &mut [bool], node_index: NodeIndex) {
        if recursion > 2 * node_has_been_evaluated.len() {
            debug!("build_evaluation_order_recurse({recursion}, {node_order_list:?}, {node_has_been_evaluated:?}, {node_index}) for");
            debug!("{self:#?}");
            self.print_net_structure();
            panic!()
        }
        assert_eq!(self.id, node_index.0);
        if node_has_been_evaluated[node_index.1] { return; /* No work to do! Already evaluated! */ }
        for &connection_index in self.get_node(node_index).input_connections.iter() {
            let connection = &self.connections[connection_index.1];
            assert_eq!(node_index, connection.output_node);
            if !connection.is_enabled { continue; }     // Treat disabled connections as not being connected (i.e. do this check here rather than in evaluate()!)
            if !node_has_been_evaluated[connection.input_node.1] {
                self.build_evaluation_order_recurse(recursion + 1, node_order_list, node_has_been_evaluated, connection.input_node);
            }
        }
        node_order_list.push(node_index);
        node_has_been_evaluated[node_index.1] = true;
    }

    fn build_layer_order_recurse(&self, recursion: usize, layer_list: &mut HashMap<NodeIndex, u16>, node_index: NodeIndex) -> u16 {
        if recursion > 2 * self.nodes.len() {
            debug!("build_layer_order_recurse({recursion}, {layer_list:?}, {node_index}) for");
            debug!("{self:#?}");
            self.print_net_structure();
            panic!()
        }

        if layer_list.contains_key(&node_index) { return layer_list[&node_index]; /* No work to do! Already computed! */ }
        let mut layer = 0;
        for connection_index in self.get_node(node_index).input_connections.iter() {
            let connection = &self.connections[connection_index.1];
            assert_eq!(node_index, connection.output_node);
            layer = layer.max(1 + self.build_layer_order_recurse(recursion + 1, layer_list, connection.input_node));
        }
        layer_list.insert(node_index, layer);
        layer
    }


    pub fn evaluate(&mut self) {
        assert!(self.is_evaluation_order_up_to_date);
        //assert!(self.node_values.len() > self.nodes.len());

        // We have already computed a correct order in which to evaluate nodes, and the caller
        // has filled in the self.node_values for all input nodes, so we now visit nodes in 
        // order and evaluate them.
        for &node_index in self.node_order_list.iter() {
            let inputs_sum = self.get_node(node_index).input_connections.iter()
                .map(|connection_index| {
                    let connection = &self.connections[connection_index.1];
                    self.get_node(connection.input_node).value * connection.weight
                })
                .sum();
            { // Scope for mutable node
                let node = &mut self.nodes[node_index.1];
                node.value = node.apply_activation_function(inputs_sum);
            }
        }
    }

    fn adjust_prob(p: f64, adjuster: f64) -> f64 {
        f64::min(1.0, p * adjuster)
    }
    
    pub(super) fn cross_into_new_net(&self, other: &Self, mut_params: &MutationParams, mutation_multiplier: f64) -> Self {
        // Choose a "winning" parent, partially based on fitnesses
        let (winner, loser) = if self.fitness_info.get_fitness() >= other.fitness_info.get_fitness() { (self, other) } else { (other, self) };
        // Small chance to actually choose the "loser" as the winner:
        let (winner, loser) = if thread_rng().gen_bool(0.2) { (loser, winner) } else { (winner, loser) };

        // Initialize the child Net
        let max_node_count = self.nodes.len().max(other.nodes.len());
        let max_connection_count = self.connections.len().max(other.connections.len());
        let mut net_child = Self {
            id: NetId::new_unique(),
            net_params: winner.net_params.clone(),
            nodes: Vec::with_capacity(max_node_count),
            map_node_id_to_index: HashMap::with_capacity(max_node_count),
            connections: Vec::with_capacity(max_connection_count),
            map_connection_id_to_index: HashMap::with_capacity(max_connection_count),
            fitness_info: F::default(),
            is_evaluation_order_up_to_date: false,
            node_order_list: Vec::new(),
        };


        // Remove a node by selecting one NOT to copy!
        let mut node_id_dont_copy: Option<NodeId> = None;
        if thread_rng().gen_bool(Self::adjust_prob(mut_params.prob_remove_node, mutation_multiplier)) {
            let hidden = winner.nodes.iter().filter_map(|n| if let Layer::Hidden(_) = n.layer { Some(n.id) } else { None }).collect::<Vec<_>>();
            if let Some(&x) = hidden.choose(&mut thread_rng()) {
                node_id_dont_copy = Some(x);
            }
        }

        // Remove a connection by selecting one NOT to copy!
        let mut connection_id_dont_copy: Option<ConnectionId> = None;
        if thread_rng().gen_bool(Self::adjust_prob(mut_params.prob_remove_connection, mutation_multiplier)) {
            if let Some(x) = winner.connections.choose(&mut thread_rng()) {
                connection_id_dont_copy = Some(x.id);
            }
        }


        // Copy the common nodes randomly from either parent.  Also, copy the disjoint nodes only from the winner.
        for node_winner in winner.nodes.iter() {
            if node_id_dont_copy.is_some_and(|id| id == node_winner.id) { 
                trace!("Skipping node copy of {}", node_winner.index); 
                continue; 
            }
            let node_to_clone = match loser.map_node_id_to_index.get(&node_winner.id) {
                None => node_winner,
                Some(&node_index_loser) => if thread_rng().gen_bool(0.5) { node_winner } else { loser.get_node(node_index_loser) },
            };
            net_child.add_node(Some(node_to_clone.id), node_to_clone.activation_function, Some(node_to_clone.layer), node_to_clone.value);
        }

        // Copy the common connections randomly from either parent, BUT always set the is_enabled to the value
        // from the winner.  Also, copy the disjoint connections only from the winner.
        for connection_winner in winner.connections.iter() {
            if connection_id_dont_copy.is_some_and(|id| id == connection_winner.id) {
                trace!("Skipping connection copy of {connection_winner:#?}"); 
                continue; 
            }
            if node_id_dont_copy.is_some_and(|id| id == winner.get_node(connection_winner. input_node).id 
                                               || id == winner.get_node(connection_winner.output_node).id) { 
                trace!("Skipping copy of connection due to skipping node; connection = {connection_winner:#?}"); 
                continue; 
            }
            let (net_of_clone, connection_to_clone) = match loser.map_connection_id_to_index.get(&connection_winner.id) {
                None => (winner, connection_winner),
                Some(connection_index_loser) => if thread_rng().gen_bool(0.5) { 
                    (winner, connection_winner)
                } else { 
                    (loser, &loser.connections[connection_index_loser.1])
                },
            };
            let input_node_id =  net_of_clone.get_node(connection_to_clone. input_node).id;
            let output_node_id = net_of_clone.get_node(connection_to_clone.output_node).id;
            net_child.add_connection(
                Some(connection_to_clone.id), 
                connection_to_clone.weight, 
                connection_to_clone.is_enabled, 
                *net_child.map_node_id_to_index.get(& input_node_id).unwrap(),
                *net_child.map_node_id_to_index.get(&output_node_id).unwrap(),
            );
        }

        // Our unique cross of connections might cause Node layer values to change
        net_child.build_evaluation_order();

        // Let's verify we got everything correct
        net_child.verify_invariants();

        trace!("NET: {net_child:#?}");
        net_child.mutate_self(mut_params, mutation_multiplier);
        net_child
    }

    fn verify_invariants(&self) { 
        trace!("NET: {:#?}", self);
        //self.print_net_structure();

        let       node_count = self.nodes      .len();
        let connection_count = self.connections.len();

        // NOTE: Must keep struct invariants intact:

        // 1. All node indexes are from this net and correctly map to the same item
        assert!(self.      nodes.iter().enumerate().all(|(i, item)| item.index.0 == self.id && item.index.1 == i));
        assert!(self.connections.iter().enumerate().all(|(i, item)| item.index.0 == self.id && item.index.1 == i));

        // 2. All items are in their respective HashMaps and their indexes are correct
        assert!(self.map_node_id_to_index      .len() == self.nodes      .len());
        assert!(self.map_connection_id_to_index.len() == self.connections.len());
        assert!(self.map_node_id_to_index      .iter().all(|(&id, &index)| index.0 == self.id && self.      nodes[index.1].id == id));
        assert!(self.map_connection_id_to_index.iter().all(|(&id, &index)| index.0 == self.id && self.connections[index.1].id == id));
        assert!(self.      nodes.iter().all(|item| item.index == self.map_node_id_to_index      [&item.id]));
        assert!(self.connections.iter().all(|item| item.index == self.map_connection_id_to_index[&item.id]));

        // 3a. All node_indexes in Connection::input_node, and Connection::output_node must
        // refer to valid nodes from the same Net
        assert!(self.connections.iter().all(|c| c. input_node.0 == self.id && c. input_node.1 < self.nodes.len()));
        assert!(self.connections.iter().all(|c| c.output_node.0 == self.id && c.output_node.1 < self.nodes.len()));
        // 3b. And all Node::input_connections refer to ConnectionIds from the same Net
        assert!(self.nodes.iter().all(|n| 
            n.input_connections.iter().all(|index| index.0 == self.id && index.1 < self.connections.len())));
        // 3c. All Node::input_connections are unique (no duplicates).  We do this by collecting
        // a HashSet of ConnectionIndexes and if the len() matches the original, there were no duplicates.
        assert!(self.nodes.iter().all(|n| {
            n.input_connections.iter().copied().collect::<HashSet<ConnectionIndex>>().len() == n.input_connections.len()
        }));

        // 4. Nodes are in proper layers
        assert!(!self.is_evaluation_order_up_to_date || self.nodes.iter().all(|n| 
            n.input_connections.iter().all(|&c_index| {
                let connection = self.get_connection(c_index);
                let input_node = self.get_node(connection.input_node);
                // `input_node` should come before `n`, unless one (or both) are
                // `Unreachable` (because we haven't properly ordered those, so we can't check them).
                let come_before_option = input_node.layer.comes_before(n.layer);
                let come_before_or_is_unreachable = come_before_option.unwrap_or(true);
                if !come_before_or_is_unreachable { trace!("n={}, c_index={c_index}, input_node={}, input_node.layer={}, n.layer={}", n.index, connection.input_node, input_node.layer, n.layer)}
                come_before_or_is_unreachable
            })
        ));

        // 5. Each connection is from a lower-numbered layer to a higher-numbered layer
        assert!(!self.is_evaluation_order_up_to_date || self.connections.iter().all(|c| {
            let input_node  = self.get_node(c. input_node);
            let output_node = self.get_node(c.output_node);
            // input_node must come before output_node, unless one of
            // them is Unreachable, in which case None was returned, and we can't really check them out.
            input_node.layer.comes_before(output_node.layer).unwrap_or(true)
        }));

        // 6. Check for cycles
        // TODO: Not sure of an easy way to do this!
    }

    fn choose_index<T:Copy>(id_list: &[T]) -> T {
        let i = thread_rng().gen_range(0..id_list.len());
        id_list[i]
    }

    fn choose_index_not<T:Copy+PartialEq>(id_list: &[T], not: T) -> T {
        for _ in 0..20 {
            let i = thread_rng().gen_range(0..id_list.len());
            let id = id_list[i];
            if id != not { return id; }
        }
        panic!("Unable to find item: choose_id_cond()")
    }


    pub(super) fn mutate_self(&mut self, mut_params: &MutationParams, mutation_multiplier: f64) {
        let node_index_list   = self.nodes.iter().map(|n| n.index).collect::<Vec<_>>();
        let input_and_hidden  = self.nodes.iter().filter_map(|n| if n.layer != Layer::Output && n.layer != Layer::Unreachable { Some(n.index) } else { None }).collect::<Vec<_>>();
        let hidden_and_output = self.nodes.iter().filter_map(|n| if n.layer != Layer::Input  && n.layer != Layer::Unreachable { Some(n.index) } else { None }).collect::<Vec<_>>();

        // Change node's activation function
        if thread_rng().gen_bool(Self::adjust_prob(mut_params.prob_mutate_activation_function_of_node, mutation_multiplier)) && !node_index_list.is_empty() {
            trace!("Mutating node activation function");
            let node_mutate = self.get_node_mut(Self::choose_index(&node_index_list));
            if node_mutate.layer != Layer::Input { node_mutate.activation_function = ActivationFunction::choose_random(); }
        }

        // Change a connection's weight
        let connection_index_list = self.connections.iter().map(|c| c.index).collect::<Vec<_>>();
        if thread_rng().gen_bool(Self::adjust_prob(mut_params.prob_mutate_weight, mutation_multiplier)) && !connection_index_list.is_empty() {
            trace!("Mutating connection weight");
            let connection_mutate = self.get_connection_mut(Self::choose_index(&connection_index_list));
            connection_mutate.weight += (thread_rng().gen::<f32>() * 2.0 - 1.0) * mut_params.max_weight_change_magnitude;
        }

        // Toggle a conneciton's is_enabled
        if thread_rng().gen_bool(Self::adjust_prob(mut_params.prob_toggle_enabled, mutation_multiplier)) && !connection_index_list.is_empty() {
            trace!("Mutating connection is_enabled");
            let connection_mutate = self.get_connection_mut(Self::choose_index(&connection_index_list));
            connection_mutate.is_enabled = !connection_mutate.is_enabled;
        }

        // Add a connection
        if thread_rng().gen_bool(Self::adjust_prob(mut_params.prob_add_connection, mutation_multiplier)) && input_and_hidden.len() > 1 {
            let mut index_from = Self::choose_index(&input_and_hidden);
            let mut index_to   = Self::choose_index_not(&hidden_and_output, index_from);
            let from = self.get_node(index_from);
            let to   = self.get_node(index_to  );
            // If we chose a "from" that comes before a "to", simply swap them
            if let Layer::Hidden(l_from) = from.layer {
                if let Layer::Hidden(l_to) = to.layer { 
                    if l_from > l_to { std::mem::swap(&mut index_from, &mut index_to); }
                }
            }

            // Make sure either "from" comes before "to", or that they are both in the exact same
            // hidden layer.
            assert!({
                let l_from = self.get_node(index_from).layer;
                let l_to   = self.get_node(index_to).layer;
                let comes_before = l_from.comes_before(l_to);
                if comes_before.is_none() {
                    trace!("l_from={l_from}, l_to={l_to}");
                }
                comes_before.unwrap() || (
                    match (l_from, l_to) {
                        (Layer::Hidden(i), Layer::Hidden(j)) => i == j,
                        _ => false,
                    }
                )
            });
            let connection_index_new = self.add_connection(
                None, 
                thread_rng().gen::<f32>() * 2.0 - 1.0, 
                true, 
                index_from, 
                index_to
            );
            let connnection_new = self.get_connection(connection_index_new);
            trace!("Mutating by adding connection {connection_index_new} from={index_from} on layer {}, to={index_to} on layer {}", self.get_node(index_from).layer, self.get_node(index_to).layer);
            assert!(connection_index_new == connnection_new.index);
        }
        // NOTE!!! From this point on, layer numbers might not be accurate... we might have
        // made a new connection between two nodes in the same hidden layer

        // Add node
        if thread_rng().gen_bool(Self::adjust_prob(mut_params.prob_add_node, mutation_multiplier)) && !connection_index_list.is_empty() {
            // Choose a random Connection, and split it into two, inserting the new node inbetween 
            // and setting old.is_enabled = false
            let connection_index_old = Self::choose_index(&connection_index_list);
            let connection_old = self.get_connection_mut(connection_index_old);
            connection_old.is_enabled = false;
            let weight_connection_new_a = connection_old.weight;
            let node_index_input  = connection_old. input_node;
            let node_index_output = connection_old.output_node;
            let node_output = self.get_node(node_index_output);
            let activation_function = node_output.activation_function;

            let node_index_new = self.add_node(None, activation_function, None, 0.0);
            let connection_index_new_a = self.add_connection(None, weight_connection_new_a, true, /*from*/ node_index_input, /*to*/ node_index_new);
            let connection_index_new_b = self.add_connection(None, activation_function.get_neutral_value(), true, /*from*/ node_index_new, /*to*/ node_index_output);
            trace!("Mutating by adding node {} and connections {} and {}", node_index_new, connection_index_new_a, connection_index_new_b);
        }

        self.is_evaluation_order_up_to_date = false;
        trace!(target: "net_EXTREME", "NET: {self:#?}");
        self.build_evaluation_order();
        trace!(target: "net_EXTREME", "NET: {self:#?}");
        self.verify_invariants();
    }
    
    pub(crate) fn set_inputs(&mut self, inputs: &[f32]) {
        assert_eq!(inputs.len(), self.net_params.input_count);
        for (i, node) in self.nodes.iter_mut().enumerate().take(self.net_params.input_count) {
            assert_eq!(node.layer, Layer::Input);
            node.value = inputs[i];
        }
    }
    
    pub(crate) fn get_outputs(&self) -> Vec::<f32> {
        let mut v = Vec::<f32>::with_capacity(self.net_params.output_count);
        for (i, node) in self.nodes.iter().enumerate().skip(self.net_params.input_count).take(self.net_params.output_count) {
            assert_eq!(node.layer, Layer::Output);
            v.push(node.value);
        }
        v
    }

    pub fn print_net_structure(&self) { // FUTURE: rewrite for being logging compatible
        let mut prev = Layer::Input;
        for n in self.nodes.iter() {
            let index = n.index.1;
            let kind = match n.layer {
                Layer::Input => "I".to_string(),
                Layer::Output => "O".to_string(),
                Layer::Unreachable => "U".to_string(),
                Layer::Hidden(h) => format!("H{h}"),
            };
            print!("N{index}/{kind} : ");
            match n.layer {
                Layer::Input  => if let Some(x) = self.net_params. input_names { print!("{} : ", x[index]); },
                Layer::Output => if let Some(x) = self.net_params.output_names { print!("{} : ", x[index - self.net_params.input_count]); },
                _ => {},
            }
            for (i, c) in n.input_connections.iter().map(|&i| self.get_connection(i)).enumerate() {
                let comma = if i == 0 { "" } else { ", " };
                let index = c.index.1;
                let tf = if c.is_enabled { "t" } else { "FALSE" };
                let from = c.input_node.1;
                let to = c.output_node.1;
                print!("{comma}C{index}({tf}:N{from}->N{to})");
            }
            println!();
            if prev == Layer::Input  && n.layer != Layer::Input  { println!(); }
            if prev == Layer::Output && n.layer != Layer::Output { println!(); }
            prev = n.layer;
        }
    }
}




#[cfg(test)]
mod tests {
    use log::info;

    use super::*;

    #[test]
    fn verify_invariants_on_empty() {
        let net = Net::<f32>::new(NetParams::from_size(7, 5));
        net.verify_invariants();
    }

    #[test]
    fn test_mutations_separately() {
        let net = Net::<f32>::new(NetParams::from_size(10, 4));
        net.verify_invariants();
        let params = MutationParams {
            prob_add_connection: 0.0,
            prob_add_node: 0.0,
            prob_mutate_activation_function_of_node: 0.0,
            prob_mutate_weight: 0.0,
            max_weight_change_magnitude: 0.0,
            prob_toggle_enabled: 0.0,
            prob_remove_connection: 0.0,
            prob_remove_node: 0.0,
        };
        let mut param_add_connection = params.clone();  param_add_connection.prob_add_connection = 1.0;
        let mut param_add_node       = params.clone();  param_add_node      .prob_add_node       = 1.0;
        let mut param_toggle_enabled = params.clone();  param_toggle_enabled.prob_toggle_enabled = 1.0;
        let mut param_mutate_weight  = params.clone();  param_mutate_weight .prob_mutate_weight  = 1.0;   param_mutate_weight.max_weight_change_magnitude = 5.0;
        let mut param_mutate_af      = params.clone();  param_mutate_af     .prob_mutate_activation_function_of_node = 1.0;

        let mut net = Net::<f32>::new(NetParams::from_size(10, 4));
        net.mutate_self(&param_add_connection, 1.0);

        let mut net = Net::<f32>::new(NetParams::from_size(10, 4));
        net.mutate_self(&param_add_node, 1.0);

        let mut net = Net::<f32>::new(NetParams::from_size(10, 4));
        net.mutate_self(&param_toggle_enabled, 1.0);

        let mut net = Net::<f32>::new(NetParams::from_size(10, 4));
        net.mutate_self(&param_mutate_weight, 1.0);

        let mut net = Net::<f32>::new(NetParams::from_size(10, 4));
        net.mutate_self(&param_mutate_af, 1.0);
    }

    #[test]
    fn test_multiple_mutatations() {
        let mut net = Net::<f32>::new(NetParams::from_size(11, 2));
        net.verify_invariants();
        let params = MutationParams {
            prob_add_connection: 0.1,
            prob_add_node: 0.1,
            prob_mutate_activation_function_of_node: 0.1,
            prob_mutate_weight: 0.1,
            max_weight_change_magnitude: 1.0,
            prob_toggle_enabled: 0.1,
            prob_remove_connection: 0.0,
            prob_remove_node: 0.0,
        };
        for _ in 0..100 {
            net.mutate_self(&params, 1.0);
        }
        info!("Mutated Net = {net:#?}");
    }


    #[test]
    fn test_remove_node() {
        for _ in 0..100 {
            let mut net_a = Net::<f32>::new(NetParams::from_size(6, 6));
            let mut net_b = Net::<f32>::new(NetParams::from_size(6, 6));
            net_a.verify_invariants();
            net_b.verify_invariants();
            let params = MutationParams {
                prob_add_connection: 1.0,
                prob_add_node: 1.0,
                prob_mutate_activation_function_of_node: 0.0,
                prob_mutate_weight: 0.0,
                max_weight_change_magnitude: 1.0,
                prob_toggle_enabled: 0.0,
                prob_remove_connection: 0.0,
                prob_remove_node: 0.0,
            };
            for _ in 0..5 {
                net_a.mutate_self(&params, 1.0);
                net_b.mutate_self(&params, 1.0);
            }
            let params = MutationParams {
                prob_add_connection: 0.0,
                prob_add_node: 0.0,
                prob_mutate_activation_function_of_node: 0.0,
                prob_mutate_weight: 0.0,
                max_weight_change_magnitude: 1.0,
                prob_toggle_enabled: 0.0,
                prob_remove_connection: 0.0,
                prob_remove_node: 1.0,
            };
            let net_d = net_a.cross_into_new_net(&net_b, &params, 1.0);
            let nodes_a = net_a.nodes.len();
            let nodes_b = net_b.nodes.len();
            let nodes_d = net_d.nodes.len();
            debug!("Remove NODE: Child={nodes_d} nodes; Parent A={nodes_a}; Parent B={nodes_b}");
            debug!("Remove NODE: Child={} connections; Parent A={}; Parent B={}", net_d.connections.len(), net_a.connections.len(), net_b.connections.len());
            assert!(nodes_d < nodes_a && nodes_d < nodes_b);
        }
    }

    #[test]
    fn test_remove_connection() {
        for _ in 0..100 {
            let mut net_a = Net::<f32>::new(NetParams::from_size(7, 7));
            let mut net_b = Net::<f32>::new(NetParams::from_size(7, 7));
            net_a.verify_invariants();
            net_b.verify_invariants();
            let params = MutationParams {
                prob_add_connection: 1.0,
                prob_add_node: 1.0,
                prob_mutate_activation_function_of_node: 0.0,
                prob_mutate_weight: 0.0,
                max_weight_change_magnitude: 1.0,
                prob_toggle_enabled: 0.0,
                prob_remove_connection: 0.0,
                prob_remove_node: 0.0,
            };
            for _ in 0..5 {
                net_a.mutate_self(&params, 1.0);
                net_b.mutate_self(&params, 1.0);
            }
            let params = MutationParams {
                prob_add_connection: 0.0,
                prob_add_node: 0.0,
                prob_mutate_activation_function_of_node: 0.0,
                prob_mutate_weight: 0.0,
                max_weight_change_magnitude: 1.0,
                prob_toggle_enabled: 0.0,
                prob_remove_connection: 1.0,
                prob_remove_node: 0.0,
            };
            let net_c = net_a.cross_into_new_net(&net_b, &params, 1.0);
            let connections_a = net_a.connections.len();
            let connections_b = net_b.connections.len();
            let connections_c = net_c.connections.len();
            debug!("Remove CONNECTION: Child={connections_c} connections; Parent A={connections_a}; Parent B={connections_b}");
            assert!(connections_a == 0 || (connections_c < connections_a && connections_c < connections_b));
        }
    }


    #[test]
    fn test_unconnected_hidden_node() {
        let mut net_a = Net::<f32>::new(NetParams::from_size(1, 1));
        assert!(net_a.nodes[0].layer == Layer::Input);
        let ni_input = NodeIndex(net_a.id, 0);
        assert!(net_a.nodes[1].layer == Layer::Output);
        let ni_output = NodeIndex(net_a.id, 1);
        let ni_ha = net_a.add_node(None, ActivationFunction::LReLU, None, 0.0);
        let ci_x = net_a.add_connection(None, 1.0, false, ni_ha, ni_output);
        let ni_hb = net_a.add_node(None, ActivationFunction::LReLU, None, 0.0);
        let ci_y = net_a.add_connection(None, 1.0, true, ni_ha, ni_hb);
        net_a.verify_invariants();
        net_a.build_evaluation_order();
        net_a.verify_invariants();
    }    
}