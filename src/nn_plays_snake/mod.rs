use core::fmt;

use crate::neural_net::nets::{Net, NetParams};
use crate::neural_net::populations::{FitnessInfo, PopulationParams};
use crate::snake_game::{Direction, GameState, SnakeGame};
use crate::neural_net::{populations::Population, nets::MutationParams};

// TODO list:
// - Support load/save of Nets
// - Create separate Net viewer
// - Support load/save of game playback
// - Create separate Playback viewer
// - Combine net viewer with playback viewer (animate net during playback!)
// - Prune Layer::Unreachable nodes!
// - Mark nodes not (eventually) reaching back to Inputs as Layer::Unreachable
// - OR: Figure out how to correctly assign Hidden(#) to current Unreachables!
// - Write out Nets (and Playback) when "New Max" above 550 encountered (pass in command line?)
// - Refactor NeuralNet and SnakeGame into crates separate from snake_bevy
// - Add originating NetId into ConnectionId (and NodeId)?  So we can trace geneology?
// - Mark Nets with a GUID for easy long-term identification
// - When population stagnates (e.g. 100 generations without new highest fitness):
//      - Always stash newest best fitness when above, say, 100 generations
//      - Increase mutations
//      - If population has already been rebooted x times, then seed next generation from the 
//          stash instead of usual best from prev generation; reset reboot counter
//      - Stash top 5% or so, and reboot population
//      - CONSIDER: Using NEAT approach to retaining genetically distinct Nets in population?
//      - CONSIDER: Using different fitness functions to create diversity, e.g.:
//          - Instead of 75% max + 25% ave, use
//              - only max
//              - only ave
//              - only min
//          - Change weighing of apples vs. uniqely visited squares vs. movement
//              - e.g. for a while (100 generations?) make visiting unique sqares most important, then apples, then moves
//          - Add severe penalty for Hidden node count or moves or moves beyond unique ones
//          - Also, choose to keep (or not to keep) fitness once computed
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


#[derive(Copy, Clone)]
pub struct MyFitnessInfo {
    fitness: f32,
    visited: f32,
    apples:  f32,
    moves:   f32,
    //net_id: Option<NetId>,
}

impl Default for MyFitnessInfo {
    fn default() -> Self {
        MyFitnessInfo {
            fitness: f32::MIN,
            visited: 0.0,
            apples:  0.0,
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

impl NnPlaysSnake {
    pub fn new() -> Self {
        let my_meta = MyMetaParams {
            max_generations: 100_000,
            games_per_net: 10,
            generations_between_events: 25,
            meta: PopulationParams {
                population_size: 1_000,
                net_params: NetParams {
                    input_count: NUM_INPUTS,
                    input_names: Some(&INPUT_NAMES),
                    output_count: NUM_OUTPUTS,
                    output_names: Some(&OUTPUT_NAMES),
                },
                mutation_params: MutationParams {
                    prob_add_connection: 0.05,
                    prob_add_node: 0.05,
                    prob_mutate_activation_function_of_node: 0.05,
                    prob_mutate_weight: 0.10,
                    max_weight_change_magnitude: 1.0,
                    prob_toggle_enabled: 0.025,
                    prob_remove_connection: 0.01,
                    prob_remove_node: 0.025,
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

    pub fn gen_count_since_last_max(&self, generation: usize) -> usize {
        if self.stashed_nets.is_empty() { 0 } else { generation - self.stashed_nets[self.stashed_nets.len() - 1].generation }
    }

    pub fn run_x_generations(&mut self) {
        let mut stash_population_last = 0;
        for generation in 0..self.my_meta.max_generations {
            self.run_one_generation(generation, self.my_meta.games_per_net);
            let count_in_stash = self.population.nets.iter().filter(|n| self.stashed_nets.iter().any(|b| n.id == b.net.id)).count();
            if count_in_stash != stash_population_last || (generation % 10) == 0 {
                stash_population_last = count_in_stash;
                let n = &self.population.nets[0];
                println!("Best for gen {generation}: {}: fitness={}; population from stash: {count_in_stash} ({:.1}%)", n.id, n.fitness_info, 100.0 * count_in_stash as f32 / self.stashed_nets.len() as f32);
            }
            if (generation % self.my_meta.generations_between_events) == 0 {
                self.pick_and_apply_event();
            }
        }
    }

    pub fn run_one_generation(&mut self, generation: usize, games_played_for_fitness: usize) {
        let multiplier = if self.gen_count_since_last_max(generation) > 100 { 2.0 } else { 1.0 };
        let pop  = &mut self.population;
        let game = &mut self.game;
        let mut global_max_fitness_info = self.max_info;
        pop.run_one_generation(multiplier, |net| {
            if net.fitness_info.fitness != f32::MIN { return net.fitness_info; }  // If we've already computed this Net's fitness, just use that
            let mut max_single_game_fitness_info = MyFitnessInfo::default();
            let mut sum_fitnesses_info = MyFitnessInfo { fitness: 0.0, ..Default::default() };
            for _ in 0..games_played_for_fitness {
                let single_game_fitness_info = Self::run_one_game(net, game);
                if max_single_game_fitness_info.fitness < single_game_fitness_info.fitness { 
                    max_single_game_fitness_info = single_game_fitness_info; 
                }
                sum_fitnesses_info += &single_game_fitness_info;
            }
            let ave_fitness_info = sum_fitnesses_info * (1.0 / games_played_for_fitness as f32);
            let final_net_fitness_info = max_single_game_fitness_info * 0.75 + ave_fitness_info * 0.25;
            net.fitness_info = final_net_fitness_info;
            if global_max_fitness_info.fitness < final_net_fitness_info.fitness {
                println!("New Max gen={generation}:  {}: fitness={final_net_fitness_info}; max={max_single_game_fitness_info}    multiplier={multiplier}", net.id);
                global_max_fitness_info = final_net_fitness_info;
                self.stashed_nets.push(StashInfo { 
                    net: net.clone(), 
                    generation,
                });
            }
            final_net_fitness_info
        });
        self.max_info = global_max_fitness_info;
    }

    pub fn run_one_game(net: &mut Net<MyFitnessInfo>, game: &mut SnakeGame) -> MyFitnessInfo {
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
            if moves > 500 + game.points_visited { break; }
        }
        // Fitness now includes # unique squares visited, where what's considered unique
        // gets reset every apple (so points_visited is monotonically increasing).
        let apples = game.apples_eaten;
        let adjustment = if apples < 2 { -1.0 } else { 1.0 }; 
        let visited = game.points_visited;
        let fitness = (10_000 * apples) as f32 
            - adjustment * 0.001 * ((moves - visited) as f32 / (apples + 1) as f32)
            + visited as f32;
        MyFitnessInfo { 
            fitness,
            visited: visited as f32,
            apples:  apples  as f32,
            moves:   moves   as f32,
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

        let inputs: [f32; NUM_INPUTS] = [
            wall_dist[0] as f32,
            wall_dist[1] as f32,
            wall_dist[2] as f32,
            wall_dist[3] as f32,
            snake_dist[0] as f32,
            snake_dist[1] as f32,
            snake_dist[2] as f32,
            snake_dist[3] as f32,
            (pt_snake_head.x - pt_apple.x) as f32 ,
            (pt_snake_head.y - pt_apple.y) as f32 ,
            snake_length as f32, 
            1.0
        ];
        net.set_inputs(&inputs);
    }
    
    fn pick_and_apply_event(&mut self) {
        
    }

    //fn event_cataclism_remove_fewest_visited(&mut self) {
    //    for n in self.population.nets.iter() {
    //        
    //    }
    //}
}