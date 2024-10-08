use core::fmt;
use std::fs::File;
use std::io::Write;

use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

use crate::neural_net::nets::{Net, NetParams};
use crate::neural_net::populations::{FitnessInfo, PopulationParams};
use crate::snake_game::{Direction, GameState, SnakeGame};
use crate::neural_net::{populations::Population, nets::MutationParams};

// TODO list:
// x Support save of Nets
// - Support load of Nets
// - Create separate Net viewer
// x Support save of game playback
// - Support load of game playback
// - Create separate Playback viewer
// - Combine net viewer with playback viewer (animate net during playback!)
// - Prune Layer::Unreachable nodes!
// - Mark nodes not (eventually) reaching back to Inputs as Layer::Unreachable
// - OR: Figure out how to correctly assign Hidden(#) to current Unreachables!
// - Refactor NeuralNet and SnakeGame into crates separate from snake_bevy
// - Add originating NetId into ConnectionId (and NodeId)?  So we can trace geneology?
// - Mark Nets with a GUID for easy long-term identification
// - When population stagnates (e.g. 100 generations without new highest fitness):
//      x Always stash newest best fitness]
//      x Increase mutations
//      - If population has already been rebooted x times, then seed next generation from the 
//          stash instead of usual best from prev generation; reset reboot counter
//      - Stash top 5% or so, and reboot population
//      !!! CONSIDER: Using NEAT approach to retaining genetically distinct Nets in population?
//      - CONSIDER: Using different fitness functions to create diversity, e.g.:
//          - Instead of 75% max + 25% ave, use
//              - only max
//              - only ave
//              - only min
//          - Add severe penalty for Hidden node count or moves or moves beyond unique ones
//          - Vary mutations rate: multiplier of 1.0, 2.0, 5.0, 0.2 for a while (100 generations?)
//          - Vary population size
//      - CONSIDER: Instead of varying fitness function, per se, how about periodic cataclisms or
//          bonuses that affect the whole population, e.g.:
//          - Cataclism: Nets with fewest visited are removed from population
//          - Cataclism: Nets with most/fewest nodes are removed from population
//          - Cataclism: Lowest 50% of population based on new (temporary, only for this cataclism) fitness rule
//          - Cataclism: Keep only nets with most apples
//          - Bonus: Explode (2x, 4x?) the population for one round by randomly mating pairs
//          - Bouns: Resurection of stashed best Nets, but with their fitness re-evaluated.
//          - Bouns: Resurection of stashed best Nets, but with all of their weights tweaked.
// - Add multi-threading for running generations
// - Every 10 generations, display stats: ave(fitness, apples, visited, move), current best(fitness,etc)
// x Look at command line to determine to run the game or to run simulation; at least until refactored into multiple crates and apps!

#[derive(Copy, Clone, Serialize, Deserialize)]
pub struct MyFitnessInfo {
    fitness: f32,
    apples:  f32,
    visited: f32,
    moves:   f32,
    //net_id: Option<NetId>,
}

impl Default for MyFitnessInfo {
    fn default() -> Self {
        MyFitnessInfo {
            fitness: f32::MIN,
            apples:  0.0,
            visited: 0.0,
            moves:   0.0,
            //net_id: None,
        }
    }
}

impl FitnessInfo for MyFitnessInfo {
    fn get_fitness(&self) -> f32 { self.fitness }
    fn set_fitness(&mut self, new: f32) { self.fitness = new; }
}

impl fmt::Display for MyFitnessInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1} (apples:{:.1}, visited:{:.1}, moves={:.1})", self.fitness, self.apples, self.visited, self.moves)
    }
}

impl fmt::Debug for MyFitnessInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:.1} (apples:{:.1}, visited:{:.1}, moves={:.1})", self.fitness, self.apples, self.visited, self.moves)
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
        }
    }
}

impl std::ops::AddAssign<&Self> for MyFitnessInfo {
    fn add_assign(&mut self, rhs: &Self) {
        self.fitness += rhs.fitness;
        self.visited += rhs.visited;
        self.apples  += rhs.apples;
        self.moves   += rhs.moves;
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
    pub fn new() -> Self {
        let my_meta = MyMetaParams {
            max_generations: 100_000,
            games_per_net: 2,
            generations_between_events: 25,
            meta: PopulationParams {
                population_size: 10_000,
                net_params: NetParams {
                    input_count: NUM_INPUTS,
                    input_names: Some(&INPUT_NAMES),
                    output_count: NUM_OUTPUTS,
                    output_names: Some(&OUTPUT_NAMES),
                },
                mutation_params: MutationParams {
                    prob_add_connection: 0.05,
                    prob_add_node: 0.03,
                    prob_mutate_activation_function_of_node: 0.02,
                    prob_mutate_weight: 0.80,
                    prob_reset_weight_when_mutating: 0.10,
                    max_weight_change_frac: 0.10,   // +/- 10% of current value
                    prob_toggle_enabled: 0.025,
                    prob_remove_connection: 0.0, // 0.01,
                    prob_remove_node: 0.0, // 0.025,
                },
            },
        };
        Self {
            game: SnakeGame::new(None),
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
                    self.pick_and_apply_event(&era_info);
                } else if era_info.is_end_special_fitness {
                    println!("----- End Special Fitness ----- {:?}:{}", era_info.fitness_kind, era_info.eras);
                }
            }
            self.run_one_generation(generation, &era_info, self.my_meta.games_per_net);
            let count_in_stash = self.population.nets.iter().filter(|n| self.stashed_nets.iter().any(|b| n.id == b.net.id)).count();
            if count_in_stash != stash_population_last || (generation % 10) == 0 {
                stash_population_last = count_in_stash;
                let n = &self.population.nets[0];
                println!("Best for gen {generation}: {}: fitness={}; {count_in_stash} ({:.1}%)", n.id, n.fitness_info, 100.0 * count_in_stash as f32 / self.stashed_nets.len() as f32);
            }
        }
    }

    pub fn run_one_generation(&mut self, generation: usize, era_info: &EraInfo, games_played_for_fitness: usize) {
        let multiplier = 1.0 + era_info.eras as f64;
        let pop  = &mut self.population;
        let game = &mut self.game;
        let mut global_max_fitness_info = self.max_info;
        pop.run_one_generation(multiplier, |net| {
            // If we've already computed this Net's fitness, just use that, unless...
            if net.fitness_info.fitness != f32::MIN { 
                // ...unless it's an era boundary, in which case the fitness function might
                // change, so let's re-evaluate then.
                if era_info.is_era_boundary {
                    net.fitness_info.fitness = f32::MIN;
                } else {
                    return net.fitness_info;
                }
            }
            let mut max_single_game_fitness_info = MyFitnessInfo::default();
            let mut sum_fitnesses_info = MyFitnessInfo { fitness: 0.0, ..Default::default() };
            for _ in 0..games_played_for_fitness {
                let single_game_fitness_info = Self::run_one_game(net, game, era_info);
                if max_single_game_fitness_info.fitness < single_game_fitness_info.fitness { 
                    max_single_game_fitness_info = single_game_fitness_info; 
                }
                sum_fitnesses_info += &single_game_fitness_info;
            }
            let ave_fitness_info = sum_fitnesses_info * (1.0 / games_played_for_fitness as f32);
            let final_net_fitness_info = max_single_game_fitness_info * 0.75 + ave_fitness_info * 0.25;
            net.fitness_info = final_net_fitness_info;
            if generation != 0 && global_max_fitness_info.fitness < final_net_fitness_info.fitness {
                println!("New Max  gen={generation}: {}: fitness={final_net_fitness_info}; max={max_single_game_fitness_info}    multiplier={multiplier}", net.id);
                global_max_fitness_info = final_net_fitness_info;
                self.stashed_nets.push(StashInfo { 
                    net: net.clone(), 
                    generation,
                });
                match serde_json::to_string_pretty(&net) {
                    Err(e) => { println!("ERROR serializing Net to JSON: {e:#?}"); panic!() }
                    Ok(s) => {
                        let gen = generation;
                        let apples = final_net_fitness_info.apples;
                        let fitness = final_net_fitness_info.fitness;
                        let date = chrono::Local::now().format("%Y%m%d");
                        let filename = format!("stash/Net-{date}-Gen{gen}-Apples{apples}-Fit{fitness:.0}.json");
                        let mut file = File::create(filename).unwrap();
                        file.write_all(s.as_bytes()).unwrap();
                    }
                }
                match serde_json::to_string_pretty(&game.playback) {
                    Err(e) => { println!("ERROR serializing Playback to JSON: {e:#?}"); panic!() }
                    Ok(s) => {
                        let gen = generation;
                        let apples = final_net_fitness_info.apples;
                        let fitness = final_net_fitness_info.fitness;
                        let date = chrono::Local::now().format("%Y%m%d");
                        let filename = format!("stash/Net-{date}-Gen{gen}-Apples{apples}-Fit{fitness:.0}-Playback.json");
                        let mut file = File::create(filename).unwrap();
                        file.write_all(s.as_bytes()).unwrap();
                    }
                }
            }
            final_net_fitness_info
        });
        self.max_info = global_max_fitness_info;
    }

    pub fn run_one_game(net: &mut Net<MyFitnessInfo>, game: &mut SnakeGame, era_info: &EraInfo) -> MyFitnessInfo {
        game.restart(None);
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
        MyFitnessInfo { 
            fitness: Self::compute_fitness(era_info, apples, visited, moves),
            apples:  apples  as f32,
            visited: visited as f32,
            moves:   moves   as f32,
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
                10_000.0 * apples
                +    1.0 * visited
                -    0.1 * (excess_moves / (apples + 1.0))
            }
            EraFitness::FavorVisits => {
                // Favor visiting new spaces
                1_000.0 * apples
                +  40.0 * visited
                -   1.0 * excess_moves
            }
            EraFitness::FavorMoves => {
                // Favor moves
                1_000.0 * apples
                +  30.0 * moves
            }
        }
    }
    
    fn interpret_outputs(net: &Net<MyFitnessInfo>) -> Direction {
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
            3 => Direction::East,
            _ => panic!(),
        }
    }

    fn collect_and_apply_inputs(net: &mut Net<MyFitnessInfo>, game: &SnakeGame) {
        let (wall_dist, snake_dist) = game.wall_and_body_distances();
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
    fn pick_and_apply_event(&mut self, era_info: &EraInfo) {
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
            self.population.nets.push(sn.net.clone());
        }
    }
}