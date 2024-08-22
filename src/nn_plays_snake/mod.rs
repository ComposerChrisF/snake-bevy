use core::fmt;
use std::fs::File;
use std::io::Write;

use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

use crate::neural_net::nets::{Net, NetParams};
use crate::neural_net::populations::{FitnessInfo, PopulationParams};
use crate::neural_net::species::SpeciesMetaParams;
use crate::snake_game::{Direction, GameState, SnakeGame};
use crate::neural_net::{populations::Population, nets::MutationParams};

// TODO list:
// - Create Net viewer
// - Allow switching between playbacks, nets, user driving the game.
// - Allow easy selection/changing of playback and net.
// - Assume "stash/" if path specified can't be found.
// - Older, not relevant unless we allow removal of nodes/connections:
//      - Prune Layer::Unreachable nodes!
//      - Mark nodes not (eventually) reaching back to Inputs as Layer::Unreachable
//      - OR: Figure out how to correctly assign Hidden(#) to current Unreachables!
// - Refactor NeuralNet and SnakeGame into crates separate from snake_bevy
// - Add originating NetId into ConnectionId (and NodeId)?  So we can trace geneology?
// - Mark Nets with a GUID for easy long-term identification
// - Add multi-threading for running generations
// - Add "low-level reactions" to snake.  I.e. if a crash would happen based on snake's output, severely penalize
//      it's fitness score, but override the reaction and choose a valid direction (perhaps highest ranked valid
//      direction?)
//      - Alternate: consider NSEW input that turn to 1.0 when that direction would imminently cause death.
//      - Perhaps there are other "hard-coded AI" logic ideas worth pursuing.  (e.g. "choice enters closed off area,
//          so penalize score." or it's alternate: NSEW inputs that turn 1.0 if that directions closes off an area.)
// - Consider changing inputs to NSEW distance to obstacle, but also with "lifetime" of obstacle (e.g. walls are 
//      forever), but snake body depends on how close to tail it is?
// + Research and implement NEAT techniques for speciation/diversity, rather than my ad hoc stuff.
//      - From NEAT paper:
//      - pop = 150 (DPNV used 1000), c1_excess = 1.0, c2_disjoint = 1.0, c3_weights = 0.4 (DPNV used 3.0),
//          threshold = 3.0 (DPNV used 4.0 because of larger c_weights), gen_w/o_max = 15
//      - Best net from each species (if member_count > 5) copied to next generation.
//      - 80% chance net having weights mutated (90% uniformly perturbed, 10% chance assigned a new random value)
//      - 75% chance inherited gene was disabled if it was disabled in either parent
//      - 25% of offspring are result of mutation without crossover.
//      - Inter-species mating rate was 0.001.
//      - In small populations, probability of adding new node was 0.03, and new link mutation was 0.05.
//      - In larger populations, adding new link was 0.30
//      - Used modified Sigmoid(x) = 1/(1+e^(4.9x)) at all nodes

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct MyFitnessInfo {
    fitness: f32,
    apples:  f32,
    visited: f32,
    moves:   f32,

    #[serde(skip_serializing, skip_deserializing)]
    fitness_weighted_by_species: f32,
    //net_id: Option<NetId>,
}
pub const FITNESS_SENTINAL: f32 = -1_000_001.0;

impl Default for MyFitnessInfo {
    fn default() -> Self {
        MyFitnessInfo {
            fitness: FITNESS_SENTINAL,
            apples:  0.0,
            visited: 0.0,
            moves:   0.0,
            fitness_weighted_by_species: FITNESS_SENTINAL,
            //net_id: None,
        }
    }
}

impl FitnessInfo for MyFitnessInfo {
    fn get_fitness(&self) -> f32 { self.fitness }
    fn set_fitness(&mut self, new: f32) { self.fitness = new; self.fitness_weighted_by_species = new; }
    
    fn get_species_weighted_fitness(&self) -> f32 { self.fitness_weighted_by_species }
    fn set_species_weighted_fitness(&mut self, new: f32) { self.fitness_weighted_by_species = new }
}

impl fmt::Display for MyFitnessInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1} (apples:{:.1}, visited:{:.1}, moves={:.1}, wt={:.1})", self.fitness, self.apples, self.visited, self.moves, self.fitness_weighted_by_species)
    }
}

impl fmt::Debug for MyFitnessInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1} (apples:{:.1}, visited:{:.1}, moves={:.1}, wt={:.1})", self.fitness, self.apples, self.visited, self.moves, self.fitness_weighted_by_species)
    }
}
impl std::ops::Mul<f32> for MyFitnessInfo {
    type Output = MyFitnessInfo;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::Output {
            fitness: rhs * self.fitness,
            visited: rhs * self.visited,
            apples:  rhs * self.apples,
            moves:   rhs * self.moves,
            fitness_weighted_by_species: rhs * self.fitness_weighted_by_species,
        }
    }
}

impl std::ops::Add for MyFitnessInfo {
    type Output = MyFitnessInfo;

    fn add(self, rhs: Self) -> Self::Output {
        Self::Output {
            fitness: self.fitness + rhs.fitness,
            visited: self.visited + rhs.visited,
            apples:  self.apples  + rhs.apples,
            moves:   self.moves   + rhs.moves,
            fitness_weighted_by_species: self.fitness_weighted_by_species + rhs.fitness_weighted_by_species,
        }
    }
}

impl std::ops::AddAssign<&Self> for MyFitnessInfo {
    fn add_assign(&mut self, rhs: &Self) {
        self.fitness += rhs.fitness;
        self.visited += rhs.visited;
        self.apples  += rhs.apples;
        self.moves   += rhs.moves;
        self.fitness_weighted_by_species += rhs.fitness_weighted_by_species;
    }
}


#[derive(Copy, Clone, PartialEq, Debug)]
pub enum EraFitness {
    Normal = 0,
    FavorVisits,
    FavorMoves,
}


#[derive(Copy, Clone, Debug)]
pub struct EraInfo {
    pub generations: usize,
    pub eras: usize,
    pub is_era_boundary: bool,
    pub is_end_special_fitness: bool,
    pub fitness_kind: EraFitness,
}


#[allow(clippy::identity_op)]
pub const NUM_INPUTS: usize = 
    4 /*NSEW dist to wall*/ +
    4 /*NSEW dist to snake*/ +
    2 /*x,y head - x,y apple*/ +
    1 /*snake length*/ +
    1 /*1.0 (bias)*/ +
    0;
pub const INPUT_NAMES: [&str; NUM_INPUTS] = [
    "WallN", "WallE", "WallS", "WallW",
    "SnakeN", "SnakeE", "SnakeS", "SnakeW",
    "AppleDistX", "AppleDistY",
    "SnakeLen",
    "1.0",
];
pub const NUM_OUTPUTS: usize = 4;
pub const OUTPUT_NAMES: [&str; NUM_OUTPUTS] = [
    "MoveN", "MoveE", "MoveS", "MoveW",
];


#[derive(Clone,Debug)]
pub struct MyMetaParams {
    pub max_generations: usize, // 100_000
    pub games_per_net: usize, // 10
    pub generations_between_events: usize, // 25
    pub meta: PopulationParams,
}

pub struct StashInfo {
    pub net: Net<MyFitnessInfo>,
    pub generation: usize,
}

pub struct NnPlaysSnake {
    game: SnakeGame,
    my_meta: MyMetaParams,
    population: Population<MyFitnessInfo>,
    max_info: MyFitnessInfo,
    stashed_nets: Vec<StashInfo>,
}


impl Default for NnPlaysSnake {
    fn default() -> Self { Self::new() }
}

pub const ERA_SIZE: usize = 200;
pub const ERA_FIRST_PORTION_SIZE: usize = 100;

impl NnPlaysSnake {
    pub fn new_params() -> NetParams {
        NetParams {
            input_count: NUM_INPUTS,
            input_names: Some(&INPUT_NAMES),
            output_count: NUM_OUTPUTS,
            output_names: Some(&OUTPUT_NAMES),
        }
    }

    pub fn new() -> Self {
        let my_meta = MyMetaParams {
            max_generations: 100_000,
            games_per_net: 10,
            generations_between_events: 100,
            meta: PopulationParams {
                population_size: 1_000, // 150,   // was 1_000 or 10_000
                min_required_members_to_forward_best: 5,
                frac_chance_to_cross_globally: 0.001,       // 0.1% chance to mate cross-species
                net_params: Self::new_params(),
                mutation_params: MutationParams {
                    prob_add_connection: 0.05,
                    prob_add_node: 0.03,
                    prob_mutate_activation_function_of_node: 0.0,   // 0.02,
                    prob_mutate_weight: 0.80,
                    prob_reset_weight_when_mutating: 0.10,
                    max_weight_change_frac: 0.10,   // +/- 10% of current value
                    prob_toggle_enabled: 0.025,
                    prob_remove_connection: 0.0, // 0.01,
                    prob_remove_node: 0.0, // 0.025,
                },
                species_param: SpeciesMetaParams { 
                    c1_excess: 1.0, 
                    c2_disjoint: 1.0, 
                    c3_weights: 0.4, 
                    threshold: 3.0, 
                    frac_eliminated: 0.50, 
                    gen_without_new_max: 15,
                 },
            },
        };
        Self {
            game: SnakeGame::new(),
            my_meta: my_meta.clone(),
            population: Population::new(my_meta.meta),
            max_info: MyFitnessInfo::default(),
            stashed_nets: Vec::new(),
        }
    }

    fn compute_era_fitness(eras: usize, gens_since_max: usize) -> EraFitness {
        if (gens_since_max % ERA_SIZE) >= ERA_FIRST_PORTION_SIZE { return EraFitness::Normal; }
        match eras % 3 {
            0 => EraFitness::Normal,
            1 => EraFitness::FavorVisits,
            2 => EraFitness::FavorMoves,
            _ => panic!()
        }
    }

    pub fn eras_since_last_max(&self, generation: usize) -> EraInfo {
        let generation_of_max = if self.stashed_nets.is_empty() { 0 } else { self.stashed_nets[self.stashed_nets.len() - 1].generation };
        let gens_since_max = generation - generation_of_max;
        let eras = gens_since_max / ERA_SIZE;
        EraInfo {
            generations: gens_since_max,
            eras,
            is_era_boundary: (gens_since_max % ERA_SIZE) == 0,
            is_end_special_fitness: (gens_since_max % ERA_SIZE) == ERA_FIRST_PORTION_SIZE,
            fitness_kind: Self::compute_era_fitness(eras, gens_since_max),
        }
    }

    pub fn run_x_generations(&mut self) {
        let mut stash_population_last = 0;
        for generation in 0..self.my_meta.max_generations {
            let era_info = self.eras_since_last_max(generation);
            if era_info.eras > 0 {
                if era_info.is_era_boundary {
                    println!("***** NEW ERA ****************************************** {:?}:{}", era_info.fitness_kind, era_info.eras);
                    self.pick_and_apply_era_event(&era_info);
                } else if era_info.is_end_special_fitness {
                    println!("----- End Special Fitness ----- {:?}:{}", era_info.fitness_kind, era_info.eras);
                }
            }
            self.run_one_generation(generation, &era_info, self.my_meta.games_per_net);
            let count_in_stash = self.population.nets.iter().filter(|n| self.stashed_nets.iter().any(|b| n.id == b.net.id)).count();
            if count_in_stash != stash_population_last || (generation % 10) == 0 {
                stash_population_last = count_in_stash;
                let n = &self.population.nets[0];
                let net_id = n.id;
                let species_count = self.population.species.species_list.len();
                let (cur, max) = self.population.species.species_list.iter().fold((0, 0), |acc, s| (acc.0 + s.stats.current_count, acc.1 + s.stats.max_count));
                let cur = cur as f32 / species_count as f32;
                let max = max as f32 / species_count as f32;
                let pop = self.population.nets.len();
                let stash_len = self.stashed_nets.len();
                println!("Best for gen {generation}: {net_id}: fitness={}; {count_in_stash} ({:.1}%,{stash_len}) - species={species_count}({cur:.1},{max:.1})/pop={pop}", n.fitness_info, 100.0 * count_in_stash as f32 / stash_len as f32);
            }
        }
    }

    pub fn run_one_generation(&mut self, generation: usize, era_info: &EraInfo, games_played_for_fitness: usize) {
        let multiplier = 1.0 + era_info.eras as f64;
        let pop  = &mut self.population;
        let game = &mut self.game;
        let mut global_max_fitness_info = self.max_info;
        pop.run_one_generation(multiplier, |net, net_count_in_same_species| {
            // If we've already computed this Net's fitness, just use that, unless...
            if net.fitness_info.fitness != FITNESS_SENTINAL { 
                // ...unless it's an era boundary, in which case the fitness function might
                // change, so let's re-evaluate then.
                if era_info.is_era_boundary {
                    net.fitness_info.fitness = FITNESS_SENTINAL;
                } else {
                    net.fitness_info.fitness_weighted_by_species = net.fitness_info.fitness / net_count_in_same_species;    // Update, since population changed since last time!!
                    return net.fitness_info;
                }
            }
            let mut max_single_game_fitness_info = MyFitnessInfo::default();
            let mut min_single_game_fitness_info = MyFitnessInfo { fitness: f32::MAX, fitness_weighted_by_species: f32::MAX, ..Default::default() };
            let mut max_playback = game.playback.clone();
            let mut sum_fitnesses_info = MyFitnessInfo { fitness: 0.0, fitness_weighted_by_species: 0.0, ..Default::default() };
            for _ in 0..games_played_for_fitness {
                let single_game_fitness_info = Self::run_one_game(net, game, era_info, net_count_in_same_species);
                assert!(single_game_fitness_info.fitness != crate::nn_plays_snake::FITNESS_SENTINAL);
                assert!(single_game_fitness_info.fitness_weighted_by_species != crate::nn_plays_snake::FITNESS_SENTINAL);
                if max_single_game_fitness_info.fitness_weighted_by_species < single_game_fitness_info.fitness_weighted_by_species { 
                    max_single_game_fitness_info = single_game_fitness_info;
                    max_playback = game.playback.clone();
                }
                if min_single_game_fitness_info.fitness_weighted_by_species > single_game_fitness_info.fitness_weighted_by_species {
                    min_single_game_fitness_info = single_game_fitness_info;
                }
                sum_fitnesses_info += &single_game_fitness_info;
            }
            assert!(max_single_game_fitness_info.fitness != crate::nn_plays_snake::FITNESS_SENTINAL);
            assert!(min_single_game_fitness_info.fitness != f32::MAX);
            let ave_fitness_info = sum_fitnesses_info * (1.0 / games_played_for_fitness as f32);
            let mut final_net_fitness_info = max_single_game_fitness_info * 0.25 + ave_fitness_info * 0.50 + min_single_game_fitness_info * 0.25;
            final_net_fitness_info.fitness_weighted_by_species = final_net_fitness_info.fitness / net_count_in_same_species;    // Set explicitly to avoid rounding errors in sums
            if generation != 0 && global_max_fitness_info.fitness_weighted_by_species < final_net_fitness_info.fitness_weighted_by_species {
                println!("New Max  gen={generation}: {}: fitness={final_net_fitness_info}; max={max_single_game_fitness_info}    multiplier={multiplier}", net.id);
                global_max_fitness_info = final_net_fitness_info;
                self.stashed_nets.push(StashInfo { 
                    net: net.clone(), 
                    generation,
                });
                // Write out current playback and net to JSON files (for further inspection and the ability to load them in later)
                match serde_json::to_string_pretty(&net) {
                    Err(e) => { println!("ERROR serializing Net to JSON: {e:#?}"); panic!() }
                    Ok(s) => {
                        let gen = generation;
                        let apples = final_net_fitness_info.apples;
                        let apples_max = max_single_game_fitness_info.apples;
                        let sw_fitness = final_net_fitness_info.fitness_weighted_by_species;
                        let date = chrono::Local::now().format("%Y%m%d");
                        let filename = format!("stash/Net-{date}-Fit{sw_fitness:.0}-Apples{apples:.2}({apples_max:.0})-Gen{gen}.json");
                        let mut file = File::create(filename).unwrap();
                        file.write_all(s.as_bytes()).unwrap();
                    }
                }
                match serde_json::to_string_pretty(&max_playback) {
                    Err(e) => { println!("ERROR serializing Playback to JSON: {e:#?}"); panic!() }
                    Ok(s) => {
                        let gen = generation;
                        let apples = final_net_fitness_info.apples;
                        let apples_max = max_single_game_fitness_info.apples;
                        let sw_fitness = final_net_fitness_info.fitness_weighted_by_species;
                        let date = chrono::Local::now().format("%Y%m%d");
                        let filename = format!("stash/Net-{date}-Fit{sw_fitness:.0}-Apples{apples:.2}({apples_max:.0})-Gen{gen}-Playback.json");
                        let mut file = File::create(filename).unwrap();
                        file.write_all(s.as_bytes()).unwrap();
                    }
                }
            }
            final_net_fitness_info
        });
        self.max_info = global_max_fitness_info;
    }

    pub fn run_one_game(net: &mut Net<MyFitnessInfo>, game: &mut SnakeGame, era_info: &EraInfo, net_count_in_same_species: f32) -> MyFitnessInfo {
        game.restart(None, None, None);
        let mut moves = 0_usize;
        while game.state == GameState::Running {
            Self::collect_and_apply_inputs(net, game);
            net.evaluate();
            let dir = Self::interpret_outputs(net);
            let apples_before = game.apples_eaten;
            game.move_snake(dir, None);
            if apples_before != game.apples_eaten { game.clear_visited(); }
            moves += 1;
            // Bail early if nothing is happening for too long
            if moves > 500 + game.points_visited + apples_before * (1 + SnakeGame::GROW_INCREMENT) { break; }
        }
        // Fitness now includes # unique squares visited, where what's considered unique
        // gets reset every apple (so points_visited is monotonically increasing).
        let apples  = game.apples_eaten;
        let visited = game.points_visited;
        let fitness = Self::compute_fitness(era_info, apples, visited, moves);
        MyFitnessInfo { 
            fitness,
            apples:  apples  as f32,
            visited: visited as f32,
            moves:   moves   as f32,
            fitness_weighted_by_species: fitness / net_count_in_same_species,
        }
    }


    // TODO: Consider keeping separate set of MAX values for each EraFitness value.
    fn compute_fitness(era_info: &EraInfo, apples: usize, visited: usize, moves: usize) -> f32 {
        let apples  = apples  as f32;   // Typical max is 9
        let visited = visited as f32;   // Typical max is 1000
        let moves   = moves   as f32;   // Typical max is 1300
        let excess_moves = moves - visited;
        match era_info.fitness_kind {
            EraFitness::Normal => {
                // The "normal" fitness function
                let okay_for_1st_apple = if apples == 0.0 { -1.0 } else { 1.0 };
                10_000.0 * apples
                -    1.0 * visited * okay_for_1st_apple
                -   10.0 * (excess_moves / (apples + 1.0))
            }
            EraFitness::FavorVisits => {
                // Favor visiting new spaces
                10_000.0 * apples
                +  40.0 * visited
                -  40.0 * excess_moves
            }
            EraFitness::FavorMoves => {
                // Favor moves
                10_000.0 * apples
                +   30.0 * moves
            }
        }
    }
    
    pub fn interpret_outputs(net: &Net<MyFitnessInfo>) -> Direction {
        let outputs = net.get_outputs();
        let mut i_max = 0;
        let mut v_max = f32::MIN;
        for (i, &v) in outputs.iter().enumerate() {
            if v > v_max {
                v_max = v;
                i_max = i;
            }
        }
        match i_max {
            0 => Direction::North,
            1 => Direction::East,
            2 => Direction::South,
            3 => Direction::West,
            _ => panic!(),
        }
    }

    pub fn collect_and_apply_inputs(net: &mut Net<MyFitnessInfo>, game: &SnakeGame) {
        let (wall_dist, snake_dist) = game.wall_and_body_distances();
        //println!("wall_dist={wall_dist:?}; snake_dist={snake_dist:?}");
        let pt_snake_head = game.snake.head_location;
        let pt_apple = game.apple.location;
        let snake_length = game.snake.length();

        // Normalized inputs
        let inputs: [f32; NUM_INPUTS] = [
            wall_dist[0] as f32 / 40.0,
            wall_dist[1] as f32 / 40.0,
            wall_dist[2] as f32 / 40.0,
            wall_dist[3] as f32 / 40.0,
            snake_dist[0] as f32 / 40.0,
            snake_dist[1] as f32 / 40.0,
            snake_dist[2] as f32 / 40.0,
            snake_dist[3] as f32 / 40.0,
            (pt_snake_head.x - pt_apple.x) as f32 / 35.0,   // Max distance = RMS(30,40) = 35.36
            (pt_snake_head.y - pt_apple.y) as f32 / 35.0,
            snake_length as f32 / 1200.0, 
            1.0
        ];
        net.set_inputs(&inputs);
    }
    

    // EVENTS
    fn pick_and_apply_era_event(&mut self, era_info: &EraInfo) {
        if era_info.eras > 0 && era_info.is_era_boundary && era_info.fitness_kind == EraFitness::Normal {
            self.event_resurrect_maxes();
        }
        
        
        match era_info.eras {
            4 => self.event_cataclism_remove_fewest_visited(),
            5 => self.event_cataclism_remove_fewest_apples(),
            8 => self.event_resurrect_maxes(),
            _ => {},
        }
    }

    fn event_cataclism_remove_fewest_visited(&mut self) {
        println!("XXXXXX CATACLISM: Remove fewest visited XXXXXXXXXXXXXXXXXXXXXXXX");
        let visited_max = self.population.nets.iter().map(|n| n.fitness_info.visited).reduce(|acc, v| if acc < v { v } else { acc }).unwrap();
        let visited_ave = self.population.nets.iter().map(|n| n.fitness_info.visited).sum::<f32>() / self.population.nets.len() as f32;
        let visited_benchmark = if thread_rng().gen_bool(0.5) { visited_max / 2.0 } else { visited_ave };
        self.population.nets.retain(|n| n.fitness_info.visited > visited_benchmark );
    }
    
    fn event_cataclism_remove_fewest_apples(&mut self) {
        println!("XXXXXX CATACLISM: Remove fewest apples XXXXXXXXXXXXXXXXXXXXXXXX");
        let apples_max = self.population.nets.iter().map(|n| n.fitness_info.apples).reduce(|acc, v| if acc < v { v } else { acc }).unwrap();
        let apples_ave = self.population.nets.iter().map(|n| n.fitness_info.apples).sum::<f32>() / self.population.nets.len() as f32;
        let apples_benchmark = if thread_rng().gen_bool(0.5) { apples_max / 2.0 } else { apples_ave };
        self.population.nets.retain(|n| n.fitness_info.apples > apples_benchmark );
    }

    fn event_resurrect_maxes(&mut self) {
        println!("@@@@ RESURECTION!!! @@@@@@@@@@@@@@@@@");
        for sn in self.stashed_nets.iter() {
            let mut net = sn.net.clone();
            // We need to recompute fitness for our new environment (set of species, e.g. will have 
            // changed), so set fitness values to the sentinal value.
            net.fitness_info.fitness = FITNESS_SENTINAL;
            net.fitness_info.fitness_weighted_by_species = FITNESS_SENTINAL;
            self.population.nets.push(net);
        }
    }
}