use crate::neural_net::nets::Net;
use crate::snake_game::{Direction, GameState, SnakeGame};
use crate::neural_net::{populations::Population, nets::MutationParams};

// TODO list:
// - Support load/save of Nets
// - Create separate Net viewer
// - Support load/save of game playback
// - Create separate Playback viewer
// - Combine net viewer with playback viewer (animate net during playback!)
// x Track unique cells visited, reset each apple eaten, adds to score in minor way
//      New Max: NetId(196774): fitness=2431, moves=1418, apples=0, visited=1013
//      New Max: NetId(273216): fitness=2471, moves=1486, apples=0, visited=985
//      New Max: NetId(309134): fitness=2545, moves=1523, apples=0, visited=1022
//      New Max: NetId(94449): fitness=1980, moves=990, apples=0, visited=990
//      Best for gen 104: NetId(94449): fitness=1750.75
// - Prune Layer::Unreachable nodes!
// - Mark nodes not (eventually) reaching back to Inputs as Layer::Unreachable
// - OR: Figure out how to correctly assign Hidden(#) to current Unreachables!
// - Write out Nets (and Playback) when "New Max" above 550 encountered (pass in command line?)
// - Add multi-threading for running generations
// x !!! Instead of running a single game per Net, run X games per Net and take a weighted average (75% * top  + 25% * average)
// - Refactor NeuralNet and SnakeGame into crates separate from snake_bevy
// - !!! Write custom Net formatter (to do like my own manual deciphering of network)--make printing nicer!
// + !!! Fix the stack-overflow.  Must be one of the two recursive functions--debug by counting recursions and print net when too deep!
//      - Two bugs: 
//          x C18 repeated twice on N21 (so assert connections not in node.input_nodes more than once!)
//          - Loop was created: C17(T:N21->N20) and C18(T:N20->N21), both N20 and N21 started as H4
// - Add originating NetId into ConnectionId (and NodeId)?  So we can trace geneology?
// - Mark Nets with a GUID for easy long-term identification


struct MaxInfo {
    fitness: f32,
    //moves: usize,
    //apples: usize,
    //net_id: Option<NetId>,
}

impl Default for MaxInfo {
    fn default() -> Self {
        MaxInfo {
            fitness: f32::MIN,
            //moves: 0,
            //apples: 0,
            //net_id: None,
        }
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

pub struct NnPlaysSnake {
    game: SnakeGame,
    population: Population,
    max_info: MaxInfo,
}


impl Default for NnPlaysSnake {
    fn default() -> Self { Self::new() }
}

impl NnPlaysSnake {
    pub fn new() -> Self {
        let input_count = NUM_INPUTS;
        let output_count = NUM_OUTPUTS; // Move N, S, E, or W
        let population_size = 1000;
        let mutation_params = MutationParams {
            prob_add_connection: 0.05,
            prob_add_node: 0.05,
            prob_mutate_activation_function_of_node: 0.05,
            prob_mutate_weight: 0.10,
            max_weight_change_magnitude: 1.0,
            prob_toggle_enabled: 0.025,
            prob_remove_connection: 0.01,
            prob_remove_node: 0.025,
        };
        Self {
            game: SnakeGame::new(None),
            population: Population::new(input_count, Some(&INPUT_NAMES), output_count, Some(&OUTPUT_NAMES), population_size, mutation_params),
            max_info: MaxInfo::default(),
        }
    }

    pub fn run_x_generations(&mut self, x: usize, games_per_net: usize) {
        for generation in 0..x {
            self.run_one_generation(games_per_net);
            if (generation % 2) == 0 {
                let n = &self.population.nets[0];
                println!("Best for gen {generation}: {}: fitness={}", n.id, n.fitness);
            }
        }
    }

    pub fn run_one_generation(&mut self, games_per_net: usize) {
        let pop = &mut self.population;
        let game = &mut self.game;
        let mut global_max_fitness = self.max_info.fitness;
        pop.run_one_generation(|net| {
            let mut max_for_net = f32::MIN;
            let mut sum = 0_f32;
            for _ in 0..games_per_net {
                let (moves, fitness) = Self::run_one_game(net, game);
                if global_max_fitness < fitness {
                    let apples = game.apples_eaten;
                    println!("New Max: {}: fitness={fitness}, moves={moves}, apples={apples}, visited={}", net.id, game.points_visited);
                    global_max_fitness = fitness;
                }
                if max_for_net < fitness { max_for_net = fitness; }
                sum += fitness;
            }
            max_for_net * 0.75 + (sum / games_per_net as f32) * 0.25
        });
        self.max_info.fitness = global_max_fitness;
    }

    pub fn run_one_game(net: &mut Net, game: &mut SnakeGame) -> (usize, f32) {
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
        let fitness = (10_000 * apples) as f32 
            - adjustment * 0.001 * ((2 * moves) as f32 / game.snake.length() as f32)
            + game.points_visited as f32;
        (moves, fitness)
    }
    
    fn interpret_outputs(net: &Net) -> Direction {
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

    fn collect_and_apply_inputs(net: &mut Net, game: &SnakeGame) {
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
}