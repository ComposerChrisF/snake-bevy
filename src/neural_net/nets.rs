
use std::sync::atomic::{AtomicUsize, Ordering};
use bevy::utils::hashbrown::HashMap;
use rand::{thread_rng, Rng, prelude::SliceRandom};

use super::{activation_functions::ActivationFunction, connections::{Connection, ConnectionId}, layers::Layer, nodes::{Node, NodeId}};



static NET_ID_NEXT: AtomicUsize = AtomicUsize::new(1);

/// The NetId uniquely identifies an instance of a Net.  Used for debug checks to ensure node and
/// connection indexes can only be used for the Net that generated them.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct NetId(usize);

impl NetId {
    pub fn new_unique() -> NetId {
        NetId(NET_ID_NEXT.fetch_add(1, Ordering::SeqCst))
    }
}



#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct NodeIndex(NetId, usize);

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct ConnectionIndex(NetId, usize);


pub struct MutationParams {
    prob_mutate_activation_function_of_node: f64,
    prob_mutate_weight: f64,
    max_weight_change_magnitude: f32,
    prob_toggle_enabled: f64,
    prob_remove_connection: f64,
    prob_add_connection: f64,
    prob_remove_node: f64,
    prob_add_node: f64,
}



#[derive(Debug)]
pub struct Net {
    id: NetId,
    nodes: Vec<Node>,
    map_node_id_to_index: HashMap<NodeId, NodeIndex>,
    connections: Vec<Connection>,
    map_connection_id_to_index: HashMap<ConnectionId, ConnectionIndex>,
    fitness: f32,
    input_count: usize,
    output_count: usize,
    is_evaluation_order_up_to_date: bool,
    node_order_list: Vec<NodeIndex>,
}

impl Net {
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

    pub fn new(input_count: usize, output_count: usize) -> Self {
        let mut net = Self {
            id: NetId::new_unique(),
            nodes: Vec::<Node>::with_capacity(input_count * output_count),
            map_node_id_to_index: HashMap::<NodeId, NodeIndex>::with_capacity(input_count + output_count),
            connections: Vec::with_capacity(input_count * output_count),
            map_connection_id_to_index: HashMap::with_capacity(input_count * output_count),
            fitness: f32::MIN,
            input_count,
            output_count,
            is_evaluation_order_up_to_date: false,
            node_order_list: Vec::with_capacity(input_count * output_count),
        };

        for _ in 0..input_count { 
            net.add_node(None, ActivationFunction::None, Some(Layer::Input), 0.0);
        }
        for _ in 0..output_count {
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
            self.build_evaluation_order_recurse(&mut node_order_list, &mut node_has_been_evaluated, node_index);
            self.build_layer_order_recurse(&mut layer_list, node_index);
        }

        // Copy layer number to nodes.layer from layer_list[node_index], fixing the output layers,
        // which, by convention, the output layer nodes are all the highest-valued layer.
        for node in self.nodes.iter_mut() {
            let layer = layer_list[&node.index];
            match node.layer {
                Layer::Input => assert_eq!(layer, 0),
                Layer::Output => assert!(layer > 0),
                _ => node.layer = Layer::Hidden(layer),
            }
        }
        drop(layer_list);
        drop(node_has_been_evaluated);
        self.node_order_list = node_order_list;
        //self.node_values = vec![0_f32; self.nodes.len()];
    }

    // We figure out the order to compute all output nodes by recursively seeking the values 
    // of all required inputs for each output node.  Note that the last output_count nodes 
    // are the output nodes, so we only have to evalute them.  Thus, we might skip computation
    // of nodes that don't (eventually) connect to any output.
    fn build_evaluation_order_recurse(&self, node_order_list: &mut Vec<NodeIndex>, node_has_been_evaluated: &mut [bool], node_index: NodeIndex) {
        assert_eq!(self.id, node_index.0);
        if node_has_been_evaluated[node_index.1] { return; /* No work to do! Already evaluated! */ }
        for &connection_index in self.get_node(node_index).input_connections.iter() {
            let connection = &self.connections[connection_index.1];
            assert_eq!(node_index, connection.output_node);
            if !connection.is_enabled { continue; }     // Treat disabled connections as not being connected (i.e. do this check here rather than in evaluate()!)
            if !node_has_been_evaluated[connection.input_node.1] {
                self.build_evaluation_order_recurse(node_order_list, node_has_been_evaluated, connection.input_node);
            }
        }
        node_order_list.push(node_index);
        node_has_been_evaluated[node_index.1] = true;
    }

    fn build_layer_order_recurse(&self, layer_list: &mut HashMap<NodeIndex, u16>, node_index: NodeIndex) -> u16 {
        if layer_list.contains_key(&node_index) { return layer_list[&node_index]; /* No work to do! Already computed! */ }
        let mut layer = 0;
        for connection_index in self.get_node(node_index).input_connections.iter() {
            let connection = &self.connections[connection_index.1];
            assert_eq!(node_index, connection.output_node);
            if !layer_list.contains_key(&connection.input_node) {
                layer = layer.max(1 + self.build_layer_order_recurse(layer_list, connection.input_node));
            }
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

    
    pub(super) fn cross_into_new_net(&self, other: &Self, mut_params: &MutationParams) -> Self {
        // Choose a "winning" parent, partially based on fitnesses
        let (winner, loser) = match thread_rng().gen_range(0..4) {
            0 => if thread_rng().gen_bool(0.5)    { (self, other) } else { (other, self) }, // 25% of the time, choose randomly
            _ => if self.fitness >= other.fitness { (self, other) } else { (other, self) }, // 75% of the time, choose highest fitness
            // FUTURE: Choose proportionally based on relative fitnesses
        };

        // Initialize the child Net
        let max_node_count = self.nodes.len().max(other.nodes.len());
        let max_connection_count = self.connections.len().max(other.connections.len());
        let mut net_child = Self {
            id: NetId::new_unique(),
            nodes: Vec::with_capacity(max_node_count),
            map_node_id_to_index: HashMap::with_capacity(max_node_count),
            connections: Vec::with_capacity(max_connection_count),
            map_connection_id_to_index: HashMap::with_capacity(max_connection_count),
            fitness: 0.0,
            input_count: winner.input_count,
            output_count: winner.output_count,
            is_evaluation_order_up_to_date: false,
            node_order_list: Vec::new(),
        };


        // TODO: Remove a node by selecting one NOT to copy!
        let mut node_id_dont_copy: Option<NodeId> = None;
        if thread_rng().gen_bool(mut_params.prob_remove_node) {
            let hidden = winner.nodes.iter().filter_map(|n| if let Layer::Hidden(_) = n.layer { Some(n.id) } else { None }).collect::<Vec<_>>();
            let &x = hidden.choose(&mut thread_rng()).unwrap();
            node_id_dont_copy = Some(x);
        }

        // TODO: Remove a connection by selecting one NOT to copy!
        let mut connection_id_dont_copy: Option<ConnectionId> = None;
        if thread_rng().gen_bool(mut_params.prob_remove_connection) {
            let x = winner.connections.choose(&mut thread_rng()).unwrap().id;
            connection_id_dont_copy = Some(x);
        }


        // Copy the common nodes randomly from either parent.  Also, copy the disjoint nodes only from the winner.
        for node_winner in winner.nodes.iter() {
            if node_id_dont_copy.is_some_and(|id| id == node_winner.id) { continue; }
            let node_to_clone = match loser.map_node_id_to_index.get(&node_winner.id) {
                None => node_winner,
                Some(&node_index_loser) => if thread_rng().gen_bool(0.5) { node_winner } else { loser.get_node(node_index_loser) },
            };
            net_child.add_node(Some(node_to_clone.id), node_to_clone.activation_function, Some(node_to_clone.layer), node_to_clone.value);
        }

        // Copy the common connections randomly from either parent, BUT always set the is_enabled to the value
        // from the winner.  Also, copy the disjoint connections only from the winner.
        for connection_winner in winner.connections.iter() {
            if connection_id_dont_copy.is_some_and(|id| id == connection_winner.id) { continue; }
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

        // Let's verify we got everything correct
        self.verify_invariants();

        net_child.mutate_self(mut_params);
        net_child
    }

    fn verify_invariants(&self) {
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

        // 4. Nodes are in proper layers
        assert!(self.nodes.iter().all(|n| 
            n.input_connections.iter().all(|&c_index| {
                let connection = self.get_connection(c_index);
                let input_node = self.get_node(connection.input_node);
                input_node.layer.comes_before(n.layer)
            })
        ));

        // 5. Each connection is from a lower-numbered layer to a higher-numbered layer
        assert!(self.connections.iter().all(|c| {
            let input_node  = self.get_node(c. input_node);
            let output_node = self.get_node(c.output_node);
            input_node.layer.comes_before(output_node.layer)
        }));
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


    fn mutate_self(&mut self, mut_params: &MutationParams) {
        let node_index_list   = self.nodes.iter().map(|n| n.index).collect::<Vec<_>>();
        let input_and_hidden  = self.nodes.iter().filter_map(|n| if n.layer != Layer::Output { Some(n.index) } else { None }).collect::<Vec<_>>();
        let hidden_and_output = self.nodes.iter().filter_map(|n| if n.layer != Layer::Input  { Some(n.index) } else { None }).collect::<Vec<_>>();

        // Change node's activation function
        if thread_rng().gen_bool(mut_params.prob_mutate_activation_function_of_node) {
            let node_mutate = self.get_node_mut(Self::choose_index(&node_index_list));
            if node_mutate.layer != Layer::Input { node_mutate.activation_function = ActivationFunction::choose_random(); }
        }

        // Change a connection's weight
        let connection_index_list = self.connections.iter().map(|c| c.index).collect::<Vec<_>>();
        if thread_rng().gen_bool(mut_params.prob_mutate_weight) {
            let connection_mutate = self.get_connection_mut(Self::choose_index(&connection_index_list));
            connection_mutate.weight += (thread_rng().gen::<f32>() * 2.0 - 1.0) * mut_params.max_weight_change_magnitude;
        }

        // Toggle a conneciton's is_enabled
        if thread_rng().gen_bool(mut_params.prob_toggle_enabled) {
            let connection_mutate = self.get_connection_mut(Self::choose_index(&connection_index_list));
            connection_mutate.is_enabled = !connection_mutate.is_enabled;
        }

        // Add a connection
        if thread_rng().gen_bool(mut_params.prob_add_connection) {
            let mut index_from = Self::choose_index(&input_and_hidden);
            let mut index_to   = Self::choose_index_not(&hidden_and_output, index_from);
            let from = self.get_node(index_from);
            let to   = self.get_node(index_to  );
            // If we chose a "from" that comes before a "to", simply swap them
            if let Layer::Hidden(l_from) = from.layer {
                if let Layer::Hidden(l_to) = to.layer { 
                    if to.layer.comes_before(from.layer) { std::mem::swap(&mut index_from, &mut index_to); }
                }
            }
            // Make sure either "from" comes before "to", or that they are both in the exact same
            // hidden layer.
            assert!({
                let l_from = self.get_node(index_from).layer;
                let l_to   = self.get_node(index_to).layer;
                l_from.comes_before(l_to) || (
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
            assert!(connection_index_new == connnection_new.index);
            // Since we added a new connection, we must also add this connection to the 
            // output node's input collection.
            let to = self.get_node_mut(index_to);
            to.input_connections.push(connection_index_new);
        }
        // NOTE!!! From this point on, layer numbers might not be accurate... we might have
        // made a new connection between two nodes in the same hidden layer

        // Add node
        if thread_rng().gen_bool(mut_params.prob_add_node) {
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
        }

        self.is_evaluation_order_up_to_date = false;
        self.build_evaluation_order();
        self.verify_invariants();
    }
}
