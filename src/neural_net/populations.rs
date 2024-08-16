use std::cmp::Ordering;

use bevy::utils::hashbrown::HashSet;
use rand::{thread_rng, Rng};

use crate::neural_net::nets::NetId;

use super::nets::{MutationParams, Net, NetParams};


#[derive(Clone, Debug, PartialEq)]
pub struct PopulationParams {
    pub population_size: usize,
    pub mutation_params: MutationParams,
    pub net_params: NetParams,
}

pub trait FitnessInfo : Clone + Default + std::fmt::Debug {
    fn get_fitness(&self) -> f32;
    fn set_fitness(&mut self, new: f32);
}

impl FitnessInfo for f32 {
    fn get_fitness(&self) -> f32 { *self }
    fn set_fitness(&mut self, new: f32) { *self = new; }
}


pub struct Population<Fit> where Fit: FitnessInfo {
    pub nets: Vec<Net<Fit>>,
    pub population_params: PopulationParams,
}

impl <Fit> Population<Fit> where Fit: FitnessInfo {
    pub fn new(meta: PopulationParams) -> Self {
        Self {
            nets: Vec::<Net<Fit>>::new(),
            population_params: meta,
        }
    }

    pub fn run_one_generation(&mut self, mutation_multipier: f64, fitness_of_net: impl FnMut(&mut Net<Fit>) -> Fit) {
        self.create_initial_population();
        self.evaluate_population(fitness_of_net);
        self.create_next_generation(mutation_multipier);

    }

    pub fn create_initial_population(&mut self) {
        while self.nets.len() < self.population_params.population_size {
            let mut net = Net::new(self.population_params.net_params.clone());
            net.mutate_self(&self.population_params.mutation_params, 1.0);
            assert!(net.is_evaluation_order_up_to_date);
            self.nets.push(net);
        }
    }

    pub fn evaluate_population(&mut self, mut f: impl FnMut(&mut Net<Fit>) -> Fit) {
        for net in self.nets.iter_mut() {
            net.fitness_info = f(net);
        }
    }

    pub fn create_next_generation(&mut self, mutation_multiplier: f64) {
        // Sort population by fitness
        self.nets.sort_by(|a,b| Ordering::reverse(a.fitness_info.get_fitness().partial_cmp(&b.fitness_info.get_fitness()).unwrap()));
        assert!(self.nets[0].fitness_info.get_fitness() >= self.nets[self.nets.len() - 1].fitness_info.get_fitness());
        assert!(self.nets[0].fitness_info.get_fitness() >= self.nets[1].fitness_info.get_fitness());
        let mut nets_already_chosen = HashSet::<NetId>::with_capacity(self.nets.len());
        //for i in 0..self.nets.len() {
        //    let net = &self.nets[i];
        //    println!("i={i}, id={}, fitness={}", net.id, net.fitness);
        //}

        // Forward propigate most fit nets
        let mut nets_new = Vec::<Net<Fit>>::with_capacity(self.nets.len());
        for i in 0..4 {
            nets_new.push(self.nets[i].clone());
            nets_already_chosen.insert(self.nets[i].id);
        }

        // Choose 25% of population randomly from current population, biased by their fitness
        // ranking.
        let percent_25 = (self.population_params.population_size as f32 * 0.25).round() as usize;
        let target = 4 + percent_25;
        let mut rechosen_count = 0_usize;
        while nets_new.len() < target {
            let net_chosen = &self.nets[self.choose()];
            let is_already_chosen = nets_already_chosen.contains(&net_chosen.id);
            let new_net = net_chosen.clone();
            if is_already_chosen { rechosen_count += 1; continue; } // new_net.mutate_self(&self.population_params.mutation_params, mutation_multiplier * 2.0); }
            nets_new.push(new_net);
            nets_already_chosen.insert(net_chosen.id);
        }
        //println!("Rechosen: {rechosen_count} out of {target}");

        // Fill out population by randomly choosing nets to cross proportionally by fitness
        while nets_new.len() < self.population_params.population_size {
            let net_chosen_a = &self.nets[self.choose()];
            let net_chosen_b = &self.nets[self.choose()];
            if std::ptr::addr_eq(net_chosen_a, net_chosen_b) { continue; }  // Skip if same
            let net_new = net_chosen_a.cross_into_new_net(net_chosen_b, &self.population_params.mutation_params, mutation_multiplier);
            nets_new.push(net_new);
        }
        self.nets = nets_new;
    }

    fn choose(&self) -> usize {
        let rand = thread_rng().gen::<f32>();
        let sq = rand * rand;   // more likely to choose values close to 0.0 than 1.0
        let index = (sq * self.nets.len() as f32).round() as usize;
        index.clamp(0, self.nets.len() - 1)
    }
}
