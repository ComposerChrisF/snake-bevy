// use std::sync::atomic::AtomicUsize;

use super::{nets::Net, populations::FitnessInfo};



// static SPECIES_ID_NEXT: AtomicUsize = AtomicUsize::new(1);
// 
// /// The NetId uniquely identifies an instance of a Net.  Used for debug checks to ensure node and
// /// connection indexes can only be used for the Net that generated them.
// #[derive(Copy, Clone, PartialEq, Eq, Hash)]
// pub struct SpeciesId(pub usize);
// 
// impl SpeciesId {
//     pub fn new_unique() -> SpeciesId {
//         SpeciesId(SPECIES_ID_NEXT.fetch_add(1, core::sync::atomic::Ordering::SeqCst))
//     }
// }
// 
// impl std::fmt::Display for SpeciesId {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "SpeciesId({})", self.0)
//     }
// }
// impl std::fmt::Debug for SpeciesId {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "SpeciesId({})", self.0)
//     }
// }




#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct SpeciesIndex(usize);
impl SpeciesIndex {
    pub fn get_ordinal(&self) -> usize { self.0 }
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
    pub frac_eliminated: f32,
    pub gen_without_new_max: usize, // when this many generations, typically 15-20, pass without new best result only top two species are kept
}


pub struct AllSpecies<Fit> where Fit: super::populations::FitnessInfo  {
    pub meta: SpeciesMetaParams,
    pub species: Vec<SingleSpecies<Fit>>,
}

impl <Fit> AllSpecies<Fit> where Fit: FitnessInfo {
    pub fn get(&self, index: SpeciesIndex) -> &SingleSpecies<Fit> {
        &self.species[index.0]
    }
    pub fn get_mut(&mut self, index: SpeciesIndex) -> &mut SingleSpecies<Fit> {
        &mut self.species[index.0]
    }

    pub fn compute_compatibility_distance(&self, net1: &Net<Fit>, net2: &Net<Fit>) -> (bool, f32) {
        let n = net1.connections.len().max(net2.connections.len());
        let n = if n < 10 /*20*/ { 1.0 } else { n as f32 };
        let (excess_count, disjoint_count) = net1.count_excess_disjoint(net2);
        let w = net1.sum_weights_distance_for_common_connections(net2);
        let excess_count = excess_count as f32;
        let disjoint_count = disjoint_count as f32;

        let compatibility_distance = (self.meta.c1_excess * excess_count) / n 
            + (self.meta.c2_disjoint * disjoint_count) / n 
            + self.meta.c3_weights * w;
        let is_same_species = compatibility_distance <= self.meta.threshold;
        (is_same_species, compatibility_distance)
    }

    pub fn find_or_add_species(&mut self, net: &Net<Fit>) -> SpeciesIndex {
        for s in self.species.iter() {
            let (is_in_species, _) = self.compute_compatibility_distance(&s.representative, net);
            if is_in_species { return s.index }
        }
        let s = SingleSpecies::<Fit> {
            index: SpeciesIndex(self.species.len()),
            representative: net.clone(),
        };
        let index = s.index;
        self.species.push(s);
        index
    }
}



#[derive(Clone, Debug)]
pub struct SingleSpecies<Fit> where Fit: FitnessInfo  {
    pub index: SpeciesIndex,
    pub representative: Net<Fit>,
}

