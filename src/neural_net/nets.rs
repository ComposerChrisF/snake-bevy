
use bevy::utils::hashbrown::{HashMap, HashSet};
use rand::{thread_rng, Rng};

use super::{activation_functions::ActivationFunction, connections::{Connection, ConnectionId}, layers::Layer, nodes::{Node, NodeId}, populations::Population};


#[derive(Debug)]
pub struct Net {
    nodes: HashMap<NodeId, Node>,
    connections: HashMap<ConnectionId, Connection>,
    fitness: f32,
    input_count: usize,
    output_count: usize,
    is_evaluation_order_up_to_date: bool,
    node_order_list: Vec<NodeId>,
}

impl Net {
    pub fn new(population: &mut Population, input_count: usize, output_count: usize) -> Self {
        let mut nodes = HashMap::<NodeId, Node>::with_capacity(input_count + output_count);
        for i in 0..input_count { 
            let node = Node {
                activation_function: ActivationFunction::Sigmoid,
                layer: Layer::Input,
                id: NodeId::new_unique(),
                input_connections: Vec::new(),
                value: 0.0,
            };
            nodes.insert(node.id, node);
        }
        for i in 0..output_count {
            let node = Node {
                activation_function: ActivationFunction::Sigmoid,
                layer: Layer::Output,
                id: NodeId::new_unique(),
                input_connections: Vec::new(),
                value: 0.0,
            };
            nodes.insert(node.id, node);
        }

        Self {
            nodes,
            connections: HashMap::with_capacity(input_count * output_count),
            fitness: f32::MIN,
            input_count,
            output_count,
            is_evaluation_order_up_to_date: false,
            node_order_list: Vec::new(),
        }
    }

    // NOTE: If we recursively traverse the network *once*, we can build the order that the network
    // needs to be evaluated in!  Then, to evaluate, we simply linearly replay the eval list--no
    // recursion or "node_has_been_evaluated" logic needed!
    fn build_evaluation_order(&mut self) {
        if self.is_evaluation_order_up_to_date { return; }
        let mut node_has_been_evaluated = HashSet::<NodeId>::with_capacity(self.nodes.len());
        let mut node_order_list = Vec::<NodeId>::with_capacity(self.nodes.len());
        let mut layer_list = HashMap::<NodeId, u16>::with_capacity(self.nodes.len());

        // Initialize inputs to layer 0
        for (id, node) in self.nodes.iter() {
            if node.layer == Layer::Input {
                layer_list.insert(node.id, 0);
            }
        }

        // Figure out the order to compute nodes and what layer the nodes belong to.
        for &node_id in self.nodes.iter().filter_map(|(id, n)| if n.layer == Layer::Output { Some(id) } else { None }) {
            self.build_evaluation_order_recurse(&mut node_order_list, &mut node_has_been_evaluated, node_id);
            self.build_layer_order_recurse(&mut layer_list, node_id);
        }

        // Copy layer number to nodes.layer from layer_list[node_index], fixing the output layers,
        // which, by convention, the output layer nodes are all the highest-valued layer.
        for (&node_id, node) in self.nodes.iter_mut() {
            let layer = layer_list[&node.id];
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
    fn build_evaluation_order_recurse(&self, node_order_list: &mut Vec<NodeId>, node_has_been_evaluated: &mut HashSet<NodeId>, node_id: NodeId) {
        if node_has_been_evaluated.contains(&node_id) { return; /* No work to do! Already evaluated! */ }
        for connection_id in self.nodes.get(&node_id).unwrap().input_connections.iter() {
            let connection = self.connections.get(connection_id).unwrap();
            assert_eq!(node_id, connection.output_node);
            if !connection.is_enabled { continue; }     // Treat disabled connections as not being connected (i.e. do this check here rather than in evaluate()!)
            if !node_has_been_evaluated.contains(&connection.input_node) {
                self.build_evaluation_order_recurse(node_order_list, node_has_been_evaluated, connection.input_node);
            }
        }
        node_order_list.push(node_id);
        node_has_been_evaluated.insert(node_id);
    }

    fn build_layer_order_recurse(&self, layer_list: &mut HashMap<NodeId, u16>, node_id: NodeId) -> u16 {
        if layer_list.contains_key(&node_id) { return layer_list[&node_id]; /* No work to do! Already computed! */ }
        let mut layer = 0;
        for connection_id in self.nodes.get(&node_id).unwrap().input_connections.iter() {
            let connection = self.connections.get(connection_id).unwrap();
            assert_eq!(node_id, connection.output_node);
            if !layer_list.contains_key(&connection.input_node) {
                layer = layer.max(1 + self.build_layer_order_recurse(layer_list, connection.input_node));
            }
        }
        layer_list.insert(node_id, layer);
        layer
    }


    pub fn evaluate(&mut self) {
        assert!(self.is_evaluation_order_up_to_date);
        //assert!(self.node_values.len() > self.nodes.len());

        // We have already computed a correct order in which to evaluate nodes, and the caller
        // has filled in the self.node_values for all input nodes, so we now visit nodes in 
        // order and evaluate them.
        for &node_id in self.node_order_list.iter() {
            let inputs_sum = self.nodes.get(&node_id).unwrap().input_connections.iter()
                .map(|connection_id| {
                    let connection = self.connections.get(connection_id).unwrap();
                    self.nodes.get(&connection.input_node).unwrap().value * connection.weight
                })
                .sum();
            {
                let node = self.nodes.get_mut(&node_id).unwrap();
                node.value = node.apply_activation_function(inputs_sum);
            }
        }
    }

    
    pub fn cross_into_new_net(&self, other: &Self) -> Self {
        // Choose a "winning" parent, partially based on fitnesses
        let (winner, loser) = match thread_rng().gen_range(0..4) {
            0 => if thread_rng().gen_bool(0.5)    { (self, other) } else { (other, self) }, // 25% of the time, choose randomly
            _ => if self.fitness >= other.fitness { (self, other) } else { (other, self) }, // 75% of the time, choose highest fitness
            // FUTURE: Choose proportionally based on relative fitnesses
        };

        // Copy the common nodes randomly from either parent.  Also, copy the disjoint nodes only from the winner.
        let nodes_child: HashMap<NodeId, Node> = winner.nodes.iter().map(|(&node_id_winner, node_winner)| {
            let node_to_clone = match loser.nodes.get(&node_id_winner) {
                None => node_winner,
                Some(node_loser) => if thread_rng().gen_bool(0.5) { node_winner } else { node_loser },
            };
            let node_child = Node { 
                activation_function: node_to_clone.activation_function, 
                layer: node_to_clone.layer, 
                id: node_to_clone.id, 
                input_connections: Vec::new(),  
                value: node_to_clone.value, 
            };
            (node_child.id, node_child)
        }).collect();

        // Copy the common connections randomly from either parent, BUT always set the is_enabled to the value
        // from the winner.  Also, copy the disjoint connections only from the winner.
        let connections_child: HashMap<ConnectionId, Connection> = winner.connections.iter().map(|(&connection_id_winner, connection_winner)| {
            let connection_to_clone = match loser.connections.get(&connection_id_winner) {
                None => connection_winner,
                Some(connection_loser) => if thread_rng().gen_bool(0.5) { connection_winner } else { connection_loser },
            };
            let connection_child = Connection {
                id: connection_to_clone.id,
                weight: connection_to_clone.weight,
                is_enabled: connection_winner.is_enabled,
                // Replace input_node and output_node, which point to a parent node, with a reference 
                // to the corresponding child node.
                input_node:  connection_to_clone.input_node,
                output_node: connection_to_clone.output_node,
            };
            // Make sure we are using nodes from the child's Net, not one of the parents'
            (connection_child.id, connection_child)
        }).collect();

        let net_child = Self {
            nodes: nodes_child,
            connections: connections_child,
            fitness: 0.0,
            input_count: winner.input_count,
            output_count: winner.output_count,
            is_evaluation_order_up_to_date: false,
            node_order_list: Vec::new(),
        };

        // Let's verify we got everything correct
        self.verify_invariants();

        net_child
    }

    fn verify_invariants(&self) {
        let node_count = self.nodes.len();
        let connection_count = self.connections.len();

        // NOTE: Must keep struct invariants intact:

        // 1. All hash keys map to correct entity
        assert!(self.nodes.iter().all(|(&id, n)| n.id == id));
        assert!(self.connections.iter().all(|(&id, c)| c.id == id));

        // 2a. All node_ids in Connection::input_node, and Connection::output_node must
        // refer to valid nodes from the same Net::nodes collection.
        assert!(self.connections.values().all(|c| self.nodes.contains_key(&c.input_node)));
        assert!(self.connections.values().all(|c| self.nodes.contains_key(&c.output_node)));
        // 2b. And all Node::input_connections refer to ConnectionIds found in the same 
        // Net's Net::connections collection.
        assert!(self.nodes.values().all(|n| 
            n.input_connections.iter().all(|c| self.connections.contains_key(c)))); 

        // 5. Nodes are in proper layers
        assert!(self.nodes.values().all(|n| 
            n.input_connections.iter().all(|c| {
                let connection = self.connections.get(c).unwrap();
                let input_node = self.nodes.get(&connection.input_node).unwrap();
                input_node.layer.comes_before(n.layer)
            })
        ));

        // 6. Each connection is from a lower-numbered layer to a higher-numbered layer
        assert!(self.connections.values().all(|c| {
            let input_node  = self.nodes.get(&c.input_node ).unwrap();
            let output_node = self.nodes.get(&c.output_node).unwrap();
            input_node.layer.comes_before(output_node.layer)
        }));
    }

    fn choose_id<T:Copy>(id_list: &[T]) -> T {
        let i = thread_rng().gen_range(0..id_list.len());
        id_list[i]
    }

    fn choose_id_not<T:Copy+PartialEq>(id_list: &[T], not: T) -> T {
        for _ in 0..20 {
            let i = thread_rng().gen_range(0..id_list.len());
            let id = id_list[i];
            if id != not { return id; }
        }
        panic!("Unable to find item: choose_id_cond()")
    }


    pub fn mutate_self(&mut self,
        population: &mut Population,  
        prob_mutate_activation_function_of_node: f64,   // done
        prob_mutate_weight: f64,                        // done
        max_weight_change_magnitude: f32,               // done
        prob_toggle_enabled: f64,                       // done
        prob_remove_connection: f64,                    // done
        prob_add_connection: f64,                       // done
        prob_remove_node: f64,                          // TODO!!!
        prob_add_node: f64,                             // TODO!!!
    ) {
        let node_id_list      = self.nodes.keys().copied().collect::<Vec<NodeId>>();
        let input_and_hidden  = self.nodes.keys().filter_map(|id| if self.nodes.get(id).unwrap().layer != Layer::Output { Some(*id) } else { None }).collect::<Vec<NodeId>>();
        let hidden_and_output = self.nodes.keys().filter_map(|id| if self.nodes.get(id).unwrap().layer != Layer::Input  { Some(*id) } else { None }).collect::<Vec<NodeId>>();

        // Change node's activation function
        if thread_rng().gen_bool(prob_mutate_activation_function_of_node) {
            let node_mutate = self.nodes.get_mut(&Self::choose_id(&node_id_list)).unwrap();
            if node_mutate.layer != Layer::Input { node_mutate.activation_function = ActivationFunction::choose_random(); }
        }

        // Change a connection's weight
        let connection_id_list = self.connections.keys().copied().collect::<Vec<ConnectionId>>();
        if thread_rng().gen_bool(prob_mutate_weight) {
            let connection_mutate = self.connections.get_mut(&Self::choose_id(&connection_id_list)).unwrap();
            connection_mutate.weight += (thread_rng().gen::<f32>() * 2.0 - 1.0) * max_weight_change_magnitude;
        }

        // Toggle a conneciton's is_enabled
        if thread_rng().gen_bool(prob_toggle_enabled) {
            let connection_mutate = self.connections.get_mut(&Self::choose_id(&connection_id_list)).unwrap();
            connection_mutate.is_enabled = !connection_mutate.is_enabled;
        }

        // Remove a connection
        if thread_rng().gen_bool(prob_remove_connection) {
            let id_connection_remove = Self::choose_id(&connection_id_list);
            let connection_remove = self.connections.remove(&id_connection_remove).unwrap();
            // Since we're removing a connection, the connection's output node will have a list 
            // of incoming connections--we must remove this connection from that collection.
            let node_output = self.nodes.get_mut(&connection_remove.output_node).unwrap();
            node_output.input_connections.retain(|&id_c| id_c != id_connection_remove);
            // NOTE: We do NOT update connection_id_list!!!
        }

        // Add a connection
        if thread_rng().gen_bool(prob_add_connection) {
            let mut id_from = Self::choose_id(&input_and_hidden);
            let mut id_to   = Self::choose_id_not(&hidden_and_output, id_from);
            let from = self.nodes.get(&id_from).unwrap();
            let to   = self.nodes.get(&id_to  ).unwrap();
            // If we chose a "from" that comes before a "to", simply swap them
            if let Layer::Hidden(l_from) = from.layer {
                if let Layer::Hidden(l_to) = to.layer { 
                    if to.layer.comes_before(from.layer) { std::mem::swap(&mut id_from, &mut id_to); }
                }
            }
            // Make sure either "from" comes before "to", or that they are both in the exact same
            // hidden layer.
            assert!({
                let l_from = self.nodes.get(&id_from).unwrap().layer;
                let l_to   = self.nodes.get(&id_to).unwrap().layer;
                l_from.comes_before(l_to) || (
                    match (l_from, l_to) {
                        (Layer::Hidden(i), Layer::Hidden(j)) => i == j,
                        _ => false,
                    }
                )
            });
            let connnection_new = Connection {
                id: ConnectionId::new_unique(),
                is_enabled: true,
                weight: thread_rng().gen::<f32>() * 2.0 - 1.0,
                input_node: id_from,
                output_node: id_to,
            };
            assert!(!self.connections.contains_key(&connnection_new.id));
            // Since we added a new connection, we must also add this connection to the 
            // output node's input collection.
            let to = self.nodes.get_mut(&id_to).unwrap();
            to.input_connections.push(connnection_new.id);
            self.connections.insert(connnection_new.id, connnection_new);
        }
        // NOTE!!! From this point on, layer numbers might not be accurate... we might have
        // made a new connection between two nodes in the same hidden layer

        // Add node
        if thread_rng().gen_bool(prob_add_node) {
            // Choose a random Connection, and split it into two, inserting the new node inbetween 
            // and setting old.is_enabled = false
            let connection_old = self.connections.get_mut(&Self::choose_id(&connection_id_list)).unwrap();
            connection_old.is_enabled = false;
            let [node_input, node_output] = self.nodes.get_many_mut([&connection_old.input_node, &connection_old.output_node]).unwrap();
            let activation_function = node_output.activation_function;
            let mut node_new = Node {
                activation_function,
                id: NodeId::new_unique(),
                input_connections: Vec::new(),
                layer: Layer::Hidden(1),    // This will be correctly computed later
                value: 0.0,
            };
            let connection_new_a = Connection {
                id: ConnectionId::new_unique(),
                input_node: node_input.id,
                output_node: node_new.id,
                is_enabled: true,
                weight: connection_old.weight,
            };
            let connection_new_b = Connection {
                id: ConnectionId::new_unique(),
                input_node: node_new.id,
                output_node: node_output.id,
                is_enabled: true,
                weight: activation_function.get_neutral_value(),
            };
            node_new.input_connections.push(connection_new_a.id);
            node_output.input_connections.push(connection_new_b.id);
            self.connections.insert(connection_new_a.id, connection_new_a);
            self.connections.insert(connection_new_b.id, connection_new_b);
            self.nodes.insert(node_new.id, node_new);
        }

        // Remove node
        if thread_rng().gen_bool(prob_remove_node) {
            let hidden = self.nodes.keys().filter_map(|id| if let Layer::Hidden(_) = self.nodes.get(id).unwrap().layer { Some(*id) } else { None }).collect::<Vec<NodeId>>();
            let id_node_remove = Self::choose_id(&hidden);
            self.nodes.retain(|&id, n| id != id_node_remove);
            // Now find nodes who have an input_connection from this removed node--remove that
            // input_connection.  Note we don't have to worry about output_connection, since 
            // that points to the node we've already removed.
            for n in self.nodes.values_mut() {
                n.input_connections.retain(|id_c| self.connections.get(id_c).unwrap().input_node != id_node_remove);
            }
            // Finally, remove any connections mentioning the removed node
            self.connections.retain(|&id, c| c.input_node != id_node_remove && c.output_node != id_node_remove);
        }

        self.is_evaluation_order_up_to_date = false;
        self.build_evaluation_order();
        self.verify_invariants();
    }
}
