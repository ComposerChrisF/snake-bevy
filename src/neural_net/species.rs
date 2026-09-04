use std::sync::atomic::AtomicUsize;

use serde::Serialize;

use super::{nets::Net, populations::FitnessInfo};

static SPECIES_ID_NEXT: AtomicUsize = AtomicUsize::new(1);

/// The NetId uniquely identifies an instance of a Net.  Used for debug checks to ensure node and
/// connection indexes can only be used for the Net that generated them.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Serialize, serde::Deserialize)]
pub struct SpeciesId(pub usize);

impl SpeciesId {
    pub fn new_unique() -> SpeciesId {
        SpeciesId(SPECIES_ID_NEXT.fetch_add(1, std::sync::atomic::Ordering::SeqCst))
    }
}

impl std::fmt::Display for SpeciesId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SpeciesId({})", self.0)
    }
}
impl std::fmt::Debug for SpeciesId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SpeciesId({})", self.0)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct SpeciesIndex(usize);
impl SpeciesIndex {
    pub fn get_ordinal(&self) -> usize {
        self.0
    }
}

impl std::fmt::Display for SpeciesIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SpeciesIndex({})", self.0)
    }
}
impl std::fmt::Debug for SpeciesIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SpeciesIndex({})", self.0)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpeciesMetaParams {
    pub c1_excess: f32,
    pub c2_disjoint: f32,
    pub c3_weights: f32,
    pub threshold: f32,
    pub threshold_factor: f32,
    pub frac_eliminated: f32,
    pub gen_without_new_max: usize, // when this many generations, typically 15-20, pass without new best result only top two species are kept
}

impl SpeciesMetaParams {
    pub fn compute_compatibility_distance<Fit>(
        &self,
        net1: &Net<Fit>,
        net2: &Net<Fit>,
    ) -> (bool, f32)
    where
        Fit: FitnessInfo,
    {
        let n_net1 = net1.nodes.len()
            - (net1.net_params.input_count + net1.net_params.output_count)
            + net1.connections.len();
        let n_net2 = net2.nodes.len()
            - (net2.net_params.input_count + net2.net_params.output_count)
            + net2.connections.len();
        let n = (n_net1.max(n_net2) + 1) as f32;
        let n = if n <= 5.0 { 1.0 } else { n };
        let (excess_count, disjoint_count) = net1.count_excess_disjoint(net2);
        let w = net1.sum_weights_distance_for_common_connections(net2);
        let excess_count = excess_count as f32;
        let disjoint_count = disjoint_count as f32;

        let compatibility_distance = (self.c1_excess * excess_count) / n
            + (self.c2_disjoint * disjoint_count) / n
            + self.c3_weights * w;
        if self.threshold_factor == 100.0 {
            println!("threshold={} vs. distance={compatibility_distance} (excess={excess_count}, disjoint={disjoint_count}, w={w}, n={n}, n_net1={n_net1}, n_net2={n_net2})", self.threshold * self.threshold_factor);
        }
        let is_same_species = compatibility_distance <= self.threshold * self.threshold_factor;
        (is_same_species, compatibility_distance)
    }
}

pub struct AllSpecies<Fit>
where
    Fit: super::populations::FitnessInfo,
{
    pub meta: SpeciesMetaParams,
    pub species_list: Vec<SingleSpecies<Fit>>,
    pub extinct_species_list: Vec<SingleSpecies<Fit>>,
}

impl<Fit> AllSpecies<Fit>
where
    Fit: FitnessInfo,
{
    pub fn new(meta: SpeciesMetaParams) -> Self {
        AllSpecies {
            meta,
            species_list: Vec::new(),
            extinct_species_list: Vec::new(),
        }
    }

    pub fn get(&self, index: SpeciesIndex) -> &SingleSpecies<Fit> {
        &self.species_list[index.0]
    }
    pub fn get_mut(&mut self, index: SpeciesIndex) -> &mut SingleSpecies<Fit> {
        &mut self.species_list[index.0]
    }

    pub fn assign_nets_to_species(&mut self, nets: &mut [Net<Fit>]) {
        let old_species_list = &mut self.species_list;
        let mut new_species_list = Vec::<SingleSpecies<Fit>>::new();

        // Assume all old species are extinct; we'll mark them non-extinct as we add them to new_species_list
        for s in old_species_list.iter_mut() {
            s.is_extinct = true;
            s.stats.generations_stagnant += 1;
        }

        // Assign each net to a species, re-using existing ones where possible.
        'all_nets: for net in nets.iter_mut() {
            // Is the net already in the new_species_list?
            for s in new_species_list.iter_mut() {
                let (is_in_species, _) = self
                    .meta
                    .compute_compatibility_distance(&s.representative, net);
                if is_in_species {
                    s.stats.current_count += 1;
                    net.species_index = Some(s.index);
                    continue 'all_nets;
                }
            }
            // Is the net in the old_species_list?  If so, copy the species over and update things.
            for s_old in old_species_list.iter_mut().filter(
                |s| s.is_extinct, /* i.e. not already copied to new-species_list */
            ) {
                let (is_in_species, _) = self
                    .meta
                    .compute_compatibility_distance(&s_old.representative, net);
                if is_in_species {
                    // This species needs to be added to new_species list, in which case it will have a new SpeciesIndex!
                    s_old.is_extinct = false; // Mark that we're copying this to new_species_list (before we clone!)
                    let mut s_new = s_old.clone();
                    s_new.stats.current_count = 1;
                    s_new.stats.current_sw_fitness_sum = 0.0;
                    s_new.index = SpeciesIndex(new_species_list.len());
                    net.species_index = Some(s_new.index);
                    new_species_list.push(s_new);
                    continue 'all_nets;
                }
            }
            // If we get here, then we need to create a new species
            let s_new = SingleSpecies::<Fit> {
                id: SpeciesId::new_unique(),
                index: SpeciesIndex(new_species_list.len()),
                representative: net.clone(),
                is_extinct: false,
                stats: SpeciesStats {
                    max_count: 1,
                    max_single_sw_fitness: f32::MIN,
                    max_ave_sw_fitness: f32::MIN,
                    current_count: 1,
                    current_sw_fitness_sum: 0.0,
                    current_sw_fitness_ave: 0.0,
                    current_max_sw_fitness: f32::MIN,
                    generations_stagnant: 0,
                },
            };
            net.species_index = Some(s_new.index);
            new_species_list.push(s_new);
        }

        // Move species still marked is_exinct from old_species_list to self.extinct_species_list for historical and statistcal analysis later
        for s in old_species_list.drain(0..) {
            if s.is_extinct {
                self.extinct_species_list.push(s);
            }
        }

        // Replace species_list with the new one
        self.species_list = new_species_list;

        if self.species_list.len() < 4 {
            let should_print = self.meta.threshold_factor != 0.01;
            self.meta.threshold_factor = (self.meta.threshold_factor * 0.9).max(0.01);
            if should_print {
                println!(
                    "NEW THRESHOLD (decrease): {:.2}; species={}",
                    self.meta.threshold_factor,
                    self.species_list.len()
                );
            }
        } else if self.species_list.len() > 50 {
            let should_print = self.meta.threshold_factor != 100.0;
            self.meta.threshold_factor = (self.meta.threshold_factor * 1.111_111).min(100.0);
            if should_print {
                println!(
                    "NEW THRESHOLD (increase): {:.2}; species={}",
                    self.meta.threshold_factor,
                    self.species_list.len()
                );
            }
        }
    }

    pub fn recompute_species_stats_for_new_generation(&mut self, nets: &mut [Net<Fit>]) {
        //// We build a new species list, based on used entries from the old one, and assign all
        //// nets to a species from this new list, updating all SpeciesIndexes.
        //self.assign_nets_to_species(nets);
        //// All nets now are net.species_index.is_some(), and SpeciesStats.current_count is correct.
        //
        //// Weight fitness by species count to update each net's species-weighted fitness
        //for n in nets.iter_mut() {
        //    let fitness_unweighted = n.fitness_info.get_fitness();
        //    assert!(fitness_unweighted != crate::nn_plays_snake::FITNESS_SENTINAL);
        //    let species_count = self.get(n.species_index.unwrap()).stats.current_count;
        //    assert!(species_count > 0);
        //    n.fitness_info.set_species_weighted_fitness(fitness_unweighted / species_count as f32);
        //}

        // Sort population by species-weighted fitness
        nets.sort_by(|a, b| {
            std::cmp::Ordering::reverse(
                a.fitness_info
                    .get_species_weighted_fitness()
                    .partial_cmp(&b.fitness_info.get_species_weighted_fitness())
                    .unwrap(),
            )
        });
        assert!(
            nets[0].fitness_info.get_species_weighted_fitness()
                >= nets[nets.len() - 1]
                    .fitness_info
                    .get_species_weighted_fitness()
        );
        assert!(
            nets[0].fitness_info.get_species_weighted_fitness()
                >= nets[1].fitness_info.get_species_weighted_fitness()
        );

        // Update current_sw_fitness_sums (only SpeciesStats.current_count is up-to-date), current_max_sw_fitness, and max_single_sw_fitness
        for n in nets.iter() {
            assert!(n.species_index.is_some()); // Ensure self.assign_nets_to_species() already filled this in!
            let species_index = n.species_index.unwrap();
            let species = self.get_mut(species_index);
            let stats = &mut species.stats;
            let sw_fitness = n.fitness_info.get_species_weighted_fitness();
            stats.current_sw_fitness_sum += sw_fitness;
            if stats.current_max_sw_fitness < sw_fitness {
                stats.current_max_sw_fitness = sw_fitness;
                if stats.max_single_sw_fitness < sw_fitness {
                    stats.max_single_sw_fitness = sw_fitness;
                    stats.generations_stagnant = 0;
                }
            }
        }

        // Update max_count, current_sw_fitness_ave
        for s in self.species_list.iter_mut() {
            let stats = &mut s.stats;
            if stats.max_count < stats.current_count {
                stats.max_count = stats.current_count;
            }
            stats.current_sw_fitness_ave =
                stats.current_sw_fitness_sum / stats.current_count as f32;
            if stats.max_ave_sw_fitness < stats.current_sw_fitness_ave {
                stats.max_ave_sw_fitness = stats.current_sw_fitness_ave;
                stats.generations_stagnant = 0; // Remove this line to make generations_stagnant only dependent on max single sw_fitness
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpeciesStats {
    pub max_count: usize,
    pub max_single_sw_fitness: f32,
    pub max_ave_sw_fitness: f32,
    pub current_count: usize,
    pub current_sw_fitness_sum: f32,
    pub current_sw_fitness_ave: f32,
    pub current_max_sw_fitness: f32,
    pub generations_stagnant: usize,
}

#[derive(Clone, Debug)]
pub struct SingleSpecies<Fit>
where
    Fit: FitnessInfo,
{
    pub id: SpeciesId,
    pub index: SpeciesIndex,
    pub representative: Net<Fit>,
    pub stats: SpeciesStats,
    is_extinct: bool, // Used to cull non-used species
}
