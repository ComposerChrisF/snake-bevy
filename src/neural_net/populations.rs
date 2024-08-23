
use bevy::utils::hashbrown::HashMap;
use rand::{thread_rng, Rng};

use crate::neural_net::species::SpeciesIndex;

use super::{nets::{MutationParams, Net, NetParams}, species::{AllSpecies, SpeciesMetaParams}};



#[derive(Clone, Debug, PartialEq)]
pub struct PopulationParams {
    pub population_size: usize,
    pub min_required_members_to_forward_best: usize,
    pub frac_chance_to_cross_globally: f64,
    pub mutation_params: MutationParams,
    pub net_params: NetParams,
    pub species_param: SpeciesMetaParams,
}

pub trait FitnessInfo : Clone + Default + std::fmt::Debug {
    fn get_fitness(&self) -> f32;
    fn set_fitness(&mut self, new: f32);
    fn get_species_weighted_fitness(&self) -> f32;
    fn set_species_weighted_fitness(&mut self, new: f32);
}

struct NetInfoPerSpecies<Fit> where Fit: FitnessInfo {
    net_index: usize,
    fitness_info: Fit,
    order: usize,
}


pub struct Population<Fit> where Fit: FitnessInfo {
    pub nets: Vec<Net<Fit>>,
    pub population_params: PopulationParams,
    pub species: AllSpecies<Fit>,
}

impl <Fit> Population<Fit> where Fit: FitnessInfo {
    pub fn new(meta: PopulationParams) -> Self {
        let species_param = meta.species_param.clone();
        Self {
            nets: Vec::<Net<Fit>>::new(),
            population_params: meta,
            species: AllSpecies::new(species_param),
        }
    }

    pub fn run_one_generation(&mut self, mutation_multipier: f64, fitness_of_net: impl FnMut(&mut Net<Fit>, f32) -> Fit) {
        self.create_initial_population();
        self.evaluate_population(fitness_of_net);
        self.create_next_generation(mutation_multipier);

    }

    pub fn create_initial_population(&mut self) {
        let parent_species_population_size = self.population_params.population_size as f32;
        while self.nets.len() < self.population_params.population_size {
            let mut net = Net::new(self.population_params.net_params.clone());
            net.mutate_self(&self.population_params.mutation_params, 1.0, parent_species_population_size);
            assert!(net.is_evaluation_order_up_to_date);
            self.nets.push(net);
        }
        self.species.assign_nets_to_species(&mut self.nets);
    }

    pub fn evaluate_population(&mut self, mut f: impl FnMut(&mut Net<Fit>, f32) -> Fit) {
        for net in self.nets.iter_mut() {
            let net_count_in_same_species = self.species.get(net.species_index.unwrap()).stats.current_count as f32;
            assert!(net_count_in_same_species >= 1.0);
            let fitness_info = f(net, net_count_in_same_species);
            assert!((fitness_info.get_fitness() / net_count_in_same_species - fitness_info.get_species_weighted_fitness()).abs() < 0.001);
            net.fitness_info = fitness_info;
        }
        self.species.recompute_species_stats_for_new_generation(&mut self.nets);    // self.nets is now sorted!
    }

    pub fn create_next_generation(&mut self, mutation_multiplier: f64) {
        // Allocate room for next genaration
        let mut nets_new = Vec::<Net<Fit>>::with_capacity(self.nets.len());

        // Create separate sorted, per-species list of net information
        let mut map_species_index_to_net_infos = HashMap::<SpeciesIndex, Vec<NetInfoPerSpecies<Fit>>>::new();
        for (net_index, n) in self.nets.iter().enumerate() {
            match map_species_index_to_net_infos.get_mut(&n.species_index.unwrap()) {
                None => { 
                    let net_info = NetInfoPerSpecies {
                        net_index,
                        fitness_info: n.fitness_info.clone(),
                        order: 0,
                    };
                    map_species_index_to_net_infos.insert(n.species_index.unwrap(), vec![net_info]); 
                }
                Some(net_infos) => {
                    let net_info = NetInfoPerSpecies {
                        net_index,
                        fitness_info: n.fitness_info.clone(),
                        order: net_infos.len(),
                    };
                    net_infos.push(net_info); 
                }
            }
        }

        // Copy best nets of each species (if more than 5 members) into next generation
        for (_, net_info_list) in map_species_index_to_net_infos.iter_mut()
            .filter(|(_, info_list)| info_list.len() >= self.population_params.min_required_members_to_forward_best) {
            let net_index = net_info_list[0].net_index;
            nets_new.push(self.nets[net_index].clone());
        }

        // Remove all nets in the bottom frac_eliminated of their species
        for (&species_index, net_info_list) in map_species_index_to_net_infos.iter_mut() {
            let should_eliminate_species = false; // TODO: Reintroduce?:  self.species.get(species_index).stats.generations_stagnant > self.population_params.species_param.gen_without_new_max;
            let len = net_info_list.len();
            let should_keep_most_of_species = len < 10;   
            let trunc = if should_eliminate_species { 0 } else if should_keep_most_of_species { (len - 1).max(0) } else {
                let trunc = len as f32 * (1.0 - self.population_params.species_param.frac_eliminated);
                let trunc = trunc.round() as usize;
                assert!(trunc < len || len == 1);
                trunc
            };
            //println!("Keeping {trunc} out of {len} for {};  should_eliminate_species={should_eliminate_species}, gens_w/o_max={}", self.species.get(species_index).id, self.species.get(species_index).stats.generations_stagnant);
            net_info_list.truncate(trunc);
        }


        // Sum all remaining net's sw_fitness, globally and per-species
        let mut sum_global = 0.0;
        let mut map_species_index_to_species_sum = HashMap::<SpeciesIndex, f32>::new();
        for (&species_index, net_info_list) in map_species_index_to_net_infos.iter_mut() {
            let sum_this_species = net_info_list.iter().fold(0.0, |acc, n| acc + n.fitness_info.get_species_weighted_fitness());
            map_species_index_to_species_sum.insert(species_index, sum_this_species);
            sum_global += sum_this_species;
        }

        // Based on each species' stat.frac_reproduction, add a number of new nets based on this species' old ones
        let num_to_fill = (self.population_params.population_size - nets_new.len()) as f32;
        let mut sum_fracs = 0.0;
        for (&species_index, net_info_list) in map_species_index_to_net_infos.iter() {
            if net_info_list.is_empty() { continue; }
            let species = self.species.get(species_index);
            let frac = map_species_index_to_species_sum[&species_index] / sum_global;
            let num_of_this_species_to_add = frac * num_to_fill;
            sum_fracs += frac;
            let target = nets_new.len() + num_of_this_species_to_add.round() as usize;
            if net_info_list.len() == 1 {
                let net = &self.nets[net_info_list[0].net_index];
                while nets_new.len() < target {
                    let mut net_new = net.clone();
                    net_new.mutate_self(&self.population_params.mutation_params, mutation_multiplier, 1.0);
                    nets_new.push(net_new);
                }
            } else {
                while nets_new.len() < target {
                    let net_chosen_a = self.choose_from(net_info_list, self.population_params.frac_chance_to_cross_globally);
                    let net_chosen_b = self.choose_from(net_info_list, self.population_params.frac_chance_to_cross_globally);
                    if std::ptr::addr_eq(net_chosen_a, net_chosen_b) { continue; }  // Skip if same
                    let net_new = net_chosen_a.cross_into_new_net(net_chosen_b, &self.population_params.mutation_params, mutation_multiplier, species.stats.current_count as f32);
                    nets_new.push(net_new);
                }
            }
        }

        // Fill out population by randomly choosing nets to cross proportionally by fitness
        if nets_new.len() < self.population_params.population_size - 1 { println!("NEED MORE POPULATION: cur={}, target={}, sum_fracs={sum_fracs}", nets_new.len(), self.population_params.population_size); }
        while nets_new.len() < self.population_params.population_size {
            let net_chosen_a = self.choose();
            let net_chosen_b = self.choose();
            if std::ptr::addr_eq(net_chosen_a, net_chosen_b) { continue; }  // Skip if same
            let net_new = net_chosen_a.cross_into_new_net(net_chosen_b, &self.population_params.mutation_params, mutation_multiplier, self.population_params.population_size as f32);
            nets_new.push(net_new);
        }
        self.nets = nets_new;
        self.species.assign_nets_to_species(&mut self.nets);
    }

    fn choose_from(&self, net_info_list: &[NetInfoPerSpecies<Fit>], frac_chance_to_cross_globally: f64) -> &Net<Fit> {
        if thread_rng().gen_bool(frac_chance_to_cross_globally) {
            return self.choose();
        }
        let rand = thread_rng().gen::<f32>();
        let sq = rand * rand;   // more likely to choose values close to 0.0 than 1.0
        let index = (sq * net_info_list.len() as f32).round() as usize;
        let index = index.clamp(0, net_info_list.len() - 1);
        let info = &net_info_list[index];
        &self.nets[info.net_index]
    }

    fn choose(&self) -> &Net<Fit> {
        let rand = thread_rng().gen::<f32>();
        let sq = rand * rand;   // more likely to choose values close to 0.0 than 1.0
        let index = (sq * self.nets.len() as f32).round() as usize;
        let index = index.clamp(0, self.nets.len() - 1);
        &self.nets[index]
    }
}
