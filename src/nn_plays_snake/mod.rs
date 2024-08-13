use crate::neural_net::nets::{Net, NetId};
use crate::snake_game::{Direction, GameState, SnakeGame};
use crate::neural_net::{populations::Population, nets::MutationParams};

// TODO list:
// - Support load/save of Nets
// - Create separate Net viewer
// - Support load/save of game playback
// - Create separate Playback viewer
// - Combine net viewer with playback viewer (animate net during playback!)
// - Track unique cells visited, reset each apple eaten, adds to score in minor way


struct MaxInfo {
    moves: usize,
    apples: usize,
    fitness: f32,
    net_id: Option<NetId>,
}

impl Default for MaxInfo {
    fn default() -> Self {
        MaxInfo {
            moves: 0,
            apples: 0,
            fitness: f32::MIN,
            net_id: None,
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
        let output_count = 4; // Move N, S, E, or W
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
            population: Population::new(input_count, output_count, population_size, mutation_params),
            max_info: MaxInfo::default(),
        }
    }

    pub fn run_x_generations(&mut self, x: usize) {
        for generation in 0..x {
            self.run_one_generation();
            if (generation & 1023) == 0 {
                let n = &self.population.nets[0];
                println!("Best for gen {generation}: {}: fitness={}", n.id, n.fitness);
            }
        }
    }

    pub fn run_one_generation(&mut self) {
        self.population.run_one_generation(|net| {
            self.game.restart(None);
            let mut moves = 0_usize;
            while self.game.state == GameState::Running {
                Self::collect_and_apply_inputs(net, &self.game);
                net.evaluate();
                let dir = Self::interpret_outputs(net);
                self.game.move_snake(dir, None);
                moves += 1;
                if moves > 500 + self.game.snake.length() * 2 { break; }
            }
            // TODO: Fitness should include # unique squares visited since last apple for the 
            // first few apples, at least. Note that this might require changes to the snake_game.
            let apples = self.game.apples_eaten;
            let adjustment = if apples < 2 { -1.0 } else { 1.0 }; 
            let fitness = (1000 * apples) as f32 - adjustment * ((2 * moves) as f32 / self.game.snake.length() as f32);
            if self.max_info.fitness < fitness {
                println!("New Max: {}: moves={moves}, apples={apples}, fitness={fitness}", net.id);
                self.max_info.net_id = Some(net.id);
                self.max_info.moves = moves;
                self.max_info.apples = apples;
                self.max_info.fitness = fitness;
            }
            fitness
        });
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