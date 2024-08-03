#![allow(dead_code)]
#![allow(unused_variables)]

use std::ops::Range;

use bevy::utils::hashbrown::{HashMap, HashSet};
use rand::{thread_rng, Rng};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ActivationFunction {
    None,       // f(x) = x, i.e. Linear
    Sigmoid,    // f(x) = 1.0 / (1.0 + exp(-x))
    ReLU,       // f(x) = if x > 0 { x } else { 0.0 }
    LReLU,      // f(x) = if x > 0 { x } else ( 0.1 * x )
    Tanh,       // f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
}

impl ActivationFunction {
    pub fn linear( x: f32) -> f32 { x }
    pub fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }
    pub fn relu(   x: f32) -> f32 { if x > 0.0 { x } else { 0.0 } }
    pub fn lrelu(  x: f32) -> f32 { if x >= 0.0 { x } else { 0.1 * x } }
    pub fn tanh(   x: f32) -> f32 { x.tanh() }

    pub fn apply(&self, x: f32) -> f32 {
        match self {
            ActivationFunction::None    => Self::linear(x),
            ActivationFunction::Sigmoid => Self::sigmoid(x),
            ActivationFunction::ReLU    => Self::relu(x),
            ActivationFunction::LReLU   => Self::lrelu(x),
            ActivationFunction::Tanh    => Self::tanh(x),
        }
    }
}


#[derive(Clone, Debug)]
pub struct Node {
    pub activation_function: ActivationFunction,
    pub layer: usize,
    pub innovation_id: usize,
    node_index: usize,
    input_connections: Vec<usize>,
}

impl Node {
    fn apply_activation_function(&self, input_sum: f32) -> f32 {
        // NOTE: node bias is accomplished by having one of the input nodes always having a value
        // of 1.0, thus the connection weight creates a bias.
        self.activation_function.apply(input_sum)
    }
}

#[derive(Clone, Debug)]
pub struct Connection {
    pub input_index: usize,
    pub output_index: usize,
    pub weight: f32,
    pub is_enabled: bool,
    pub innovation_id: usize,
}

#[derive(Clone, Debug)]
pub struct Net {
    nodes: Vec<Node>,
    connections: Vec<Connection>,
    fitness: f32,
    input_count: usize,
    output_count: usize,
    is_evaluation_order_up_to_date: bool,
    node_order_list: Vec<usize>,
    node_values: Vec<f32>,  // For evaluate()
}

impl Net {
    pub fn new(population: &mut Population, input_count: usize, output_count: usize) -> Self {
        let mut nodes = Vec::<Node>::with_capacity(input_count + output_count);
        for i in 0..input_count { 
            nodes.push(Node {
                activation_function: ActivationFunction::Sigmoid,
                layer: 0,
                innovation_id: population.new_gene_id(),
                node_index: i,
                input_connections: Vec::<usize>::new(),
            });
        }
        for i in 0..output_count {
            nodes.push(Node {
                activation_function: ActivationFunction::Sigmoid,
                layer: 1,
                innovation_id: population.new_gene_id(),
                node_index: i + input_count,
                input_connections: Vec::<usize>::new(),
            });
        }

        Self {
            nodes,
            connections: Vec::<Connection>::with_capacity(input_count * output_count),
            fitness: f32::MIN,
            input_count,
            output_count,
            is_evaluation_order_up_to_date: false,
            node_order_list: Vec::<usize>::new(),
            node_values: Vec::<f32>::new(),
        }
    }

    fn input_range( &self) -> Range<usize> { 0..self.input_count }
    fn hidden_range(&self) -> Range<usize> { self.input_count..(self.nodes.len() - self.output_count) }
    fn output_range(&self) -> Range<usize> { (self.nodes.len() - self.output_count)..self.nodes.len() }
    fn input_and_hidden_range( &self) -> Range<usize> { 0..(self.nodes.len() - self.output_count) }
    fn hidden_and_output_range(&self) -> Range<usize> { self.input_count..self.nodes.len() }

    // NOTE: If we recursively traverse the network *once*, we can build the order that the network
    // needs to be evaluated in!  Then, to evaluate, we simply linearly replay the eval list--no
    // recursion or "has_node_been_evaluated" logic needed!
    fn build_evaluation_order(&mut self) {
        if self.is_evaluation_order_up_to_date { return; }
        let mut has_node_been_evaluated = vec![false; self.nodes.len()];
        let mut node_order_list = Vec::<usize>::with_capacity(self.nodes.len());
        let mut layer_list: Vec<Option<usize>> = vec![None; self.nodes.len()];

        // Initialize inputs to layer 0
        let inputs = self.input_range();
        for n in self.nodes[inputs].iter_mut() {
            layer_list[n.node_index] = Some(0);
        }

        // Figure out the order to compute nodes and what layer the nodes belong to.
        for node in self.nodes[self.output_range()].iter() {
            self.build_evaluation_order_recurse(&mut node_order_list, &mut has_node_been_evaluated, node);
            self.build_layer_order_recurse(&mut layer_list, node);
        }

        // Copy layer number to nodes.layer from layer_list[node_index], fixing the output layers,
        // which, by convention, the output layer nodes are all the highest-valued layer.
        let layer_output = layer_list[self.output_range()].iter().filter_map(|&l| l).max().unwrap();
        for (node_index, &layer) in layer_list.iter().enumerate() {
            match layer {
                None => self.nodes[node_index].layer = 1,
                Some(layer) => self.nodes[node_index].layer = if self.output_range().contains(&node_index) { layer_output } else { layer }, 
            }
        }

        self.node_order_list = node_order_list;
        self.node_values = vec![0_f32; self.nodes.len()];
    }

    // We figure out the order to compute all output nodes by recursively seeking the values 
    // of all required inputs for each output node.  Note that the last output_count nodes 
    // are the output nodes, so we only have to evalute them.  Thus, we might skip computation
    // of nodes that don't (eventually) connect to any output.
    fn build_evaluation_order_recurse(&self, node_order_list: &mut Vec<usize>, has_node_been_evaluated: &mut [bool], node: &Node) {
        if has_node_been_evaluated[node.node_index] { return; /* No work to do! Already evaluated! */ }
        for &connection_index in node.input_connections.iter() {
            let connection = &self.connections[connection_index];
            assert_eq!(node.node_index, connection.output_index);
            if !connection.is_enabled { continue; }     // Treat disabled connections as not being connected (i.e. do this check here rather than in evaluate()!)
            if !has_node_been_evaluated[connection.input_index] {
                self.build_evaluation_order_recurse(node_order_list, has_node_been_evaluated, &self.nodes[connection.input_index]);
            }
        }
        node_order_list.push(node.node_index);
        has_node_been_evaluated[node.node_index] = true;
    }

    fn build_layer_order_recurse(&self, layer_list: &mut [Option<usize>], node: &Node) -> usize {
        if let Some(layer) = layer_list[node.node_index] { return layer; /* No work to do! Already computed! */ }
        let mut layer = 0;
        for &connection_index in node.input_connections.iter() {
            let connection = &self.connections[connection_index];
            assert_eq!(node.node_index, connection.output_index);
            if layer_list[connection.input_index].is_none() {
                layer = layer.max(1 + self.build_layer_order_recurse(layer_list, &self.nodes[connection.input_index]));
            }
        }
        layer_list[node.node_index] = Some(layer);
        layer
    }


    pub fn evaluate(&mut self) {
        assert!(self.is_evaluation_order_up_to_date);
        assert!(self.node_values.len() > self.nodes.len());

        // We have already computed a correct order in which to evaluate nodes, and the caller
        // has filled in the self.node_values for all input nodes, so we now visit nodes in 
        // order and evaluate them.
        for &node_index in self.node_order_list.iter() {
            let node = &self.nodes[node_index];

            let inputs_sum = node.input_connections.iter()
                .map(|&connection_index| &self.connections[connection_index])
                .map(|connection| self.node_values[connection.input_index] * connection.weight)
                .sum();

            self.node_values[node_index] = node.apply_activation_function(inputs_sum);
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
        let nodes_winner: HashMap<usize, &Node> = winner.nodes.iter().map(|n| (n.innovation_id, n)).collect();
        let nodes_loser:  HashMap<usize, &Node> = loser .nodes.iter().map(|n| (n.innovation_id, n)).collect();
        let mut map_indexes_winner_to_child = HashMap::<usize, usize>::new();
        let mut map_indexes_loser_to_child  = HashMap::<usize, usize>::new();
        let nodes_child: Vec<Node> = nodes_winner.values().enumerate().map(|(i, &node_winner)| {
            match nodes_loser.get(&node_winner.innovation_id) {
                None => {
                        map_indexes_winner_to_child.insert(node_winner.node_index, i);
                        node_winner
                    },
                Some(node_loser) => if thread_rng().gen_bool(0.5) { 
                        map_indexes_winner_to_child.insert(node_winner.node_index, i);
                        node_winner 
                    } else { 
                        map_indexes_loser_to_child.insert(node_loser.node_index, i);
                        node_loser 
                    },
            }.clone()
        }).collect();

        // Copy the common connections randomly from either parent, BUT always set the is_enabled to the value
        // from the winner.  Also, copy the disjoint connections only from the winner.
        let connections_winner: HashMap<usize, &Connection> = winner.connections.iter().map(|c| (c.innovation_id, c)).collect();
        let connections_loser:  HashMap<usize, &Connection> = loser .connections.iter().map(|c| (c.innovation_id, c)).collect();
        let connections_child: Vec<Connection> = connections_winner.values().map(|&connection_winner| {
            let (connection_child, input_index, output_index) = match connections_loser.get(&connection_winner.innovation_id) {
                None => (connection_winner, map_indexes_winner_to_child[&connection_winner.input_index], map_indexes_winner_to_child[&connection_winner.output_index]),
                Some(&connection_loser) => {
                    if thread_rng().gen_bool(0.5) { 
                        (connection_winner, map_indexes_winner_to_child[&connection_winner.input_index], map_indexes_winner_to_child[&connection_winner.output_index]) 
                    } else { 
                        (connection_loser,  map_indexes_loser_to_child[ &connection_loser .input_index], map_indexes_loser_to_child[ &connection_loser .output_index]) 
                    }
                },
            };
            let mut connection_child = connection_child.clone();
            connection_child.input_index  = input_index;
            connection_child.output_index = output_index;
            connection_child.is_enabled = connection_winner.is_enabled;
            connection_child
        }).collect();

        let net_child = Self {
            nodes: nodes_child,
            connections: connections_child,
            fitness: 0.0,
            input_count: winner.input_count,
            output_count: winner.output_count,
            is_evaluation_order_up_to_date: false,
            node_order_list: Vec::new(),
            node_values: Vec::new(),
        };

        // Let's verify we got everything correct
        self.verify_invariants();

        net_child
    }

    fn verify_invariants(&self) {
        let node_count = self.nodes.len();
        let connection_count = self.connections.len();

        // NOTE: Must keep struct invariants intact:
        // 1. All node_index values in (Node::node_index, Node::input_connections[],
        //    Connection::input_index, and Connection::output_index) must refer to valid nodes.
        assert!(self.nodes.iter().all(|n| n.node_index < node_count));
        assert!(self.nodes.iter().all(|n| n.input_connections.iter().all(|&i| i < node_count)));
        assert!(self.connections.iter().all(|c| c.input_index < node_count));
        assert!(self.connections.iter().all(|c| c.output_index < node_count));

        // 2. All innovation_id values in Net::nodes must be unique
        assert_eq!({
            let hash_set: HashSet<usize> = self.nodes.iter().map(|n| n.innovation_id).collect();
            hash_set.len()
        }, node_count);

        // 3. All innovation_id values in Net::connections must be unique
        assert_eq!({
            let hash_set: HashSet<usize> = self.connections.iter().map(|c| c.innovation_id).collect();
            hash_set.len()
        }, connection_count);

        // 4. All node/Node::input_connections[] pairs correspond to a Connection in self.connections
        assert!(self.nodes.iter().all(|n| 
            n.input_connections.iter().all(|&i| 
                self.connections.iter().any(|c| c.input_index == i && c.output_index == n.node_index)
        )));

        // 5. Nodes are in proper layers
        assert!(self.nodes.iter().all(|n| n.input_connections.iter().all(|&i| self.nodes[i].layer < n.layer)));

        // 6a. Each connection is from a lower-numbered layer to a higher-numbered layer
        assert!(self.connections.iter().all(|c| self.nodes[c.input_index].layer < self.nodes[c.output_index].layer));
        // 6b. Each connection is from a lower-numbered index to a higher-numbered index
        assert!(self.connections.iter().all(|c| c.input_index < c.output_index));

        // 7. For the new net, is_evaluation_order_up_to_date is false
        assert!(!self.is_evaluation_order_up_to_date);
    }

    pub fn mutate_self(&mut self, 
        prob_mutate_activation_function_of_node: f64,
        prob_add_connection: f64, 
        prob_remove_conneciton: f64, 
        prob_toggle_enabled: f64, 
        prob_change_weight: f64,
        max_weight_change_magnitude: f64,
        prob_remove_node: f64,
        prob_add_node: f64, 
    ) {
        if thread_rng().gen_bool(prob_mutate_activation_function_of_node) {
            let i_node_mutate = thread_rng().gen_range(self.hidden_and_output_range());
            self.nodes[i_node_mutate].activation_function = match thread_rng().gen_range(0..5) {
                0 => ActivationFunction::None,
                1 => ActivationFunction::Sigmoid,
                2 => ActivationFunction::ReLU,
                3 => ActivationFunction::LReLU,
                4 => ActivationFunction::Tanh,
                _ => panic!(),
            };
        }

        if thread_rng().gen_bool(prob_add_connection) {
            for _ in [0..10] {
                let i_node_from = thread_rng().gen_range(self.input_and_hidden_range());
                let i_node_to = thread_rng().gen_range((i_node_from + 1)..self.nodes.len());
                
            }
        }


        // Add or remove a connection (update all node.input_connections vector as needed)
        // Enable or diable a connection
        // Add or remove a hidden layer node
        // If modifications made, make sure is_evaluation_order_up_to_date is false

        self.verify_invariants();
    }
}

pub struct Population {
    pub nets: Vec<Net>,
    gene_id_max: usize,
    innovation_id_max: usize,
}

impl Population {
    pub fn new() -> Self {
        Self {
            nets: Vec::<Net>::new(),
            gene_id_max: 1,
            innovation_id_max: 1,
        }
    }

    pub fn new_gene_id(&mut self) -> usize {
        self.gene_id_max += 1;
        self.gene_id_max
    }

    pub fn new_innovation_id_max(&mut self) -> usize {
        self.innovation_id_max += 1;
        self.innovation_id_max
    }

    pub fn create_next_generation(&self) -> Population {
        todo!()
    }
}