#![allow(dead_code)]
#![allow(unused_variables)]

use std::{cell::Cell, rc::Rc};

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

#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct NodeId(usize);


#[derive(Debug)]
pub struct Node {
    pub activation_function: ActivationFunction,
    pub layer: Cell<Layer>,
    pub id: NodeId,
    input_connections: Vec<Rc<Connection>>,
    pub value: Cell<f32>,
}

impl Node {
    fn apply_activation_function(&self, input_sum: f32) -> f32 {
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


#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
pub struct ConnectionId(usize);

#[derive(Debug)]
pub struct Connection {
    pub input_node: Rc<Node>,
    pub output_node: Rc<Node>,
    pub weight: f32,
    pub is_enabled: bool,
    pub id: ConnectionId,
}

#[derive(Debug)]
pub struct Net {
    nodes: Vec<Rc<Node>>,
    connections: Vec<Rc<Connection>>,
    fitness: f32,
    input_count: usize,
    output_count: usize,
    is_evaluation_order_up_to_date: bool,
    node_order_list: Vec<Rc<Node>>,
    map_node_id_to_node: HashMap<NodeId, Rc<Node>>,
    map_connection_id_to_connection: HashMap<ConnectionId, Rc<Connection>>,
    //node_values: Vec<f32>,  // For evaluate()
}

impl Net {
    pub fn new(population: &mut Population, input_count: usize, output_count: usize) -> Self {
        let mut nodes = Vec::<Rc<Node>>::with_capacity(input_count + output_count);
        let mut map_node_id_to_node = HashMap::<NodeId, Rc<Node>>::with_capacity(input_count + output_count);
        for i in 0..input_count { 
            let node = Rc::new(Node {
                activation_function: ActivationFunction::Sigmoid,
                layer: Cell::new(Layer::Input),
                id: population.new_node_id(),
                input_connections: Vec::<Rc<Connection>>::new(),
                value: Cell::new(0.0),
            });
            nodes.push(node.clone());
            map_node_id_to_node.insert(node.id, node);
        }
        for i in 0..output_count {
            let node = Rc::new(Node {
                activation_function: ActivationFunction::Sigmoid,
                layer: Cell::new(Layer::Output),
                id: population.new_node_id(),
                input_connections: Vec::<Rc<Connection>>::new(),
                value: Cell::new(0.0),
            });
            nodes.push(node.clone());
            map_node_id_to_node.insert(node.id, node);
        }

        Self {
            nodes,
            connections: Vec::<Rc<Connection>>::with_capacity(input_count * output_count),
            fitness: f32::MIN,
            input_count,
            output_count,
            is_evaluation_order_up_to_date: false,
            node_order_list: Vec::<Rc<Node>>::new(),
            //node_values: Vec::<f32>::new(),
            map_node_id_to_node,
            map_connection_id_to_connection: HashMap::new(),
        }
    }

    //fn input_range( &self) -> Range<usize> { 0..self.input_count }
    //fn hidden_range(&self) -> Range<usize> { self.input_count..(self.nodes.len() - self.output_count) }
    //fn output_range(&self) -> Range<usize> { (self.nodes.len() - self.output_count)..self.nodes.len() }
    //fn input_and_hidden_range( &self) -> Range<usize> { 0..(self.nodes.len() - self.output_count) }
    //fn hidden_and_output_range(&self) -> Range<usize> { self.input_count..self.nodes.len() }

    // NOTE: If we recursively traverse the network *once*, we can build the order that the network
    // needs to be evaluated in!  Then, to evaluate, we simply linearly replay the eval list--no
    // recursion or "node_has_been_evaluated" logic needed!
    fn build_evaluation_order(&mut self) {
        if self.is_evaluation_order_up_to_date { return; }
        let mut node_has_been_evaluated = HashSet::<Rc<Node>>::with_capacity(self.nodes.len());
        let mut node_order_list = Vec::<Rc<Node>>::with_capacity(self.nodes.len());
        let mut layer_list = HashMap::<Rc<Node>, u16>::with_capacity(self.nodes.len());

        // Initialize inputs to layer 0
        for n in self.nodes.iter() {
            if n.layer.get() == Layer::Input {
                layer_list.insert(n.clone(), 0);
            }
        }

        // Figure out the order to compute nodes and what layer the nodes belong to.
        for node in self.nodes.iter().filter(|n| n.layer.get() == Layer::Output) {
            Self::build_evaluation_order_recurse(&mut node_order_list, &mut node_has_been_evaluated, node);
            Self::build_layer_order_recurse(&mut layer_list, node);
        }

        // Copy layer number to nodes.layer from layer_list[node_index], fixing the output layers,
        // which, by convention, the output layer nodes are all the highest-valued layer.
        for node in self.nodes.iter_mut() {
            let layer = layer_list[node];
            match node.layer.get() {
                Layer::Input => assert_eq!(layer, 0),
                Layer::Output => assert!(layer > 0),
                _ => node.layer.set(Layer::Hidden(layer)),
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
    fn build_evaluation_order_recurse(node_order_list: &mut Vec<Rc<Node>>, node_has_been_evaluated: &mut HashSet<Rc<Node>>, node: &Rc<Node>) {
        if node_has_been_evaluated.contains(node) { return; /* No work to do! Already evaluated! */ }
        for connection in node.input_connections.iter() {
            assert_eq!(node.id, connection.output_node.id);
            if !connection.is_enabled { continue; }     // Treat disabled connections as not being connected (i.e. do this check here rather than in evaluate()!)
            if !node_has_been_evaluated.contains(&connection.input_node) {
                Self::build_evaluation_order_recurse(node_order_list, node_has_been_evaluated, &connection.input_node);
            }
        }
        node_order_list.push(node.clone());
        node_has_been_evaluated.insert(node.clone());
    }

    fn build_layer_order_recurse(layer_list: &mut HashMap<Rc<Node>, u16>, node: &Rc<Node>) -> u16 {
        if layer_list.contains_key(node) { return layer_list[node]; /* No work to do! Already computed! */ }
        let mut layer = 0;
        for connection in node.input_connections.iter() {
            assert_eq!(node.id, connection.output_node.id);
            if !layer_list.contains_key(&connection.input_node) {
                layer = layer.max(1 + Self::build_layer_order_recurse(layer_list, &connection.input_node));
            }
        }
        layer_list.insert(node.clone(), layer);
        layer
    }


    pub fn evaluate(&mut self) {
        assert!(self.is_evaluation_order_up_to_date);
        //assert!(self.node_values.len() > self.nodes.len());

        // We have already computed a correct order in which to evaluate nodes, and the caller
        // has filled in the self.node_values for all input nodes, so we now visit nodes in 
        // order and evaluate them.
        for node in self.node_order_list.iter() {
            let inputs_sum = node.input_connections.iter()
                .map(|connection| connection.input_node.value.get() * connection.weight)
                .sum();

            node.value.set(node.apply_activation_function(inputs_sum));
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
        let nodes_child: Vec<Rc<Node>> = winner.nodes.iter().map(|node_winner| {
            let node_to_clone = match loser.map_node_id_to_node.get(&node_winner.id) {
                None => node_winner,
                Some(node_loser) => if thread_rng().gen_bool(0.5) { node_winner } else { node_loser },
            };
            let node_child = Rc::new(Node { 
                activation_function: node_to_clone.activation_function, 
                layer: node_to_clone.layer.clone(), 
                id: node_to_clone.id, 
                input_connections: Vec::new(),  
                value: node_to_clone.value.clone(), 
            });
            node_child
        }).collect();
        let map_node_id_to_node: HashMap<NodeId, Rc<Node>> = nodes_child.iter().map(|node| (node.id, node.clone())).collect();

        // Copy the common connections randomly from either parent, BUT always set the is_enabled to the value
        // from the winner.  Also, copy the disjoint connections only from the winner.
        let connections_child: Vec<Rc<Connection>> = winner.connections.iter().map(|connection_winner| {
            let connection_to_clone = match loser.map_connection_id_to_connection.get(&connection_winner.id) {
                None => connection_winner,
                Some(connection_loser) => if thread_rng().gen_bool(0.5) { connection_winner } else { connection_loser },
            };
            let connection_child = Rc::new(Connection {
                id: connection_to_clone.id,
                weight: connection_to_clone.weight,
                is_enabled: connection_winner.is_enabled,
                // Replace input_node and output_node, which point to a parent node, with a reference 
                // to the corresponding child node.
                input_node:  map_node_id_to_node.get(&connection_to_clone.input_node.id ).unwrap().clone(),
                output_node: map_node_id_to_node.get(&connection_to_clone.output_node.id).unwrap().clone(),
            });
            // Make sure we are using nodes from the child's Net, not one of the parents'
            assert!(!Rc::ptr_eq(&connection_child.input_node,  &connection_to_clone.input_node ));
            assert!(!Rc::ptr_eq(&connection_child.output_node, &connection_to_clone.output_node));
            connection_child
        }).collect();
        let map_connection_id_to_connection: HashMap<ConnectionId, Rc<Connection>> = 
            connections_child.iter()
                .map(|connection| (connection.id, connection.clone()))
                .collect();

        let net_child = Self {
            nodes: nodes_child,
            connections: connections_child,
            fitness: 0.0,
            input_count: winner.input_count,
            output_count: winner.output_count,
            is_evaluation_order_up_to_date: false,
            node_order_list: Vec::new(),
            //node_values: Vec::new(),
            map_node_id_to_node,
            map_connection_id_to_connection,
        };

        // Let's verify we got everything correct
        self.verify_invariants();

        net_child
    }

    fn verify_invariants(&self) {
        let node_count = self.nodes.len();
        let connection_count = self.connections.len();

        // NOTE: Must keep struct invariants intact:

        // 1. All Rc<Node> references in Connection::input_node, and Connection::output_node must
        // refer to valid nodes from the same Net::nodes collection...
        assert!(self.connections.iter().all(|c| self.nodes.iter().any(|n| Rc::ptr_eq(n, &c.input_node))));
        assert!(self.connections.iter().all(|c| self.nodes.iter().any(|n| Rc::ptr_eq(n, &c.output_node))));
        // ...and all Node::input_connections refer to Rc<Connection> found in the same 
        // Net's Net::connections collection.
        assert!(self.nodes.iter().all(|n| 
            n.input_connections.iter().all(|c1| 
                self.connections.iter().any(|c2| Rc::ptr_eq(c1, c2)))));


        // 2. All innovation_id values in Net::nodes must be unique
        assert_eq!({
            let hash_set: HashSet<NodeId> = self.nodes.iter().map(|n| n.id).collect();
            hash_set.len()
        }, node_count);

        // 3. All innovation_id values in Net::connections must be unique
        assert_eq!({
            let hash_set: HashSet<ConnectionId> = self.connections.iter().map(|c| c.id).collect();
            hash_set.len()
        }, connection_count);

        // 5. Nodes are in proper layers
        assert!(self.nodes.iter().all(|n| 
            n.input_connections.iter().all(|c| c.input_node.layer.get().comes_before(n.layer.get()))));

        // 6. Each connection is from a lower-numbered layer to a higher-numbered layer
        assert!(self.connections.iter().all(|c| c.input_node.layer.get().comes_before(c.output_node.layer.get())));

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
            let i_node_mutate = thread_rng().gen_range(0..self.nodes.len());
            let node_mutate = self.nodes[i_node_mutate].clone();
            if node_mutate.layer.get() != Layer::Input {
                //node_mutate.activation_function = match thread_rng().gen_range(0..5) {
                //    0 => ActivationFunction::None,
                //    1 => ActivationFunction::Sigmoid,
                //    2 => ActivationFunction::ReLU,
                //    3 => ActivationFunction::LReLU,
                //    4 => ActivationFunction::Tanh,
                //    _ => panic!(),
                //};
                todo!();
            }
        }


        if thread_rng().gen_bool(prob_add_connection) {
            for _ in 0..10 {
                // Select a "from" node that is NOT an Output node
                let node_from = self.nodes[thread_rng().gen_range(0..self.nodes.len())].clone();
                if node_from.layer.get() == Layer::Output { continue; }

                // Select a "to" node that is NOT an Input node
                let node_to = self.nodes[thread_rng().gen_range(0..self.nodes.len())].clone();
                if node_to.layer.get() == Layer::Input { continue; }
                
                // Make sure
                if node_from.layer.get().to_number() > node_to.layer.get().to_number() { continue; }
                
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

    pub fn new_node_id(&mut self) -> NodeId {
        self.gene_id_max += 1;
        NodeId(self.gene_id_max)
    }

    pub fn new_connection_id(&mut self) -> ConnectionId {
        self.innovation_id_max += 1;
        ConnectionId(self.innovation_id_max)
    }

    pub fn create_next_generation(&self) -> Population {
        todo!()
    }
}