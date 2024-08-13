use std::cmp::Ordering;

use rand::{thread_rng, Rng};

use super::nets::{MutationParams, Net};



pub struct Population {
    pub nets: Vec<Net>,
    input_count: usize,
    output_count: usize,
    population_size: usize,
    mutation_params: MutationParams,
}

impl Population {
    pub fn new(input_count: usize, output_count: usize, population_size: usize, mutation_params: MutationParams) -> Self {
        Self {
            nets: Vec::<Net>::new(),
            input_count,
            output_count,
            population_size,
            mutation_params,
        }
    }

    pub fn run_one_generation(&mut self, f: impl FnMut(&mut Net) -> f32) {
        self.create_initial_population();
        self.evaluate_population(f);
        self.create_next_generation();
    }

    pub fn create_initial_population(&mut self) {
        while self.nets.len() < self.population_size {
            let mut net = Net::new(self.input_count, self.output_count);
            net.mutate_self(&self.mutation_params);
            assert!(net.is_evaluation_order_up_to_date);
            self.nets.push(net);
        }
    }

    pub fn evaluate_population(&mut self, mut f: impl FnMut(&mut Net) -> f32) {
        for net in self.nets.iter_mut() {
            net.fitness = f(net);
        }
    }

    pub fn create_next_generation(&mut self) {
        // Sort population by fitness
        self.nets.sort_by(|a,b| Ordering::reverse(a.fitness.partial_cmp(&b.fitness).unwrap()));
        assert!(self.nets[0].fitness >= self.nets[self.nets.len() - 1].fitness);
        //for i in 0..self.nets.len() {
        //    let net = &self.nets[i];
        //    println!("i={i}, id={}, fitness={}", net.id, net.fitness);
        //}

        // Allocate points based on fitness scores
        let mut fitness_prev = self.nets[0].fitness;
        let mut points_cur = 100.0_f32;
        let mut points_sum = 0.0_f32;
        let mut net_points = vec![0.0_f32; self.nets.len()];
        for (i, net) in self.nets.iter().enumerate() {
            points_cur *= 1.0 - ((fitness_prev - net.fitness) / fitness_prev);
            points_cur = points_cur.max(1.0);
            fitness_prev = net.fitness;
            net_points[i] = points_cur;
            points_sum += points_cur;
        }

        // Forward propigate most fit net
        let mut nets_new = Vec::<Net>::with_capacity(self.nets.len());
        nets_new.push(self.nets[0].clone());

        // Fill out 10% of population by randomly choosing from current population, in proportion
        // to their fitness.
        let ten_percent = (self.population_size as f32 * 0.1).round() as usize - 1;
        for _ in 0..ten_percent {
            let net_chosen = &self.nets[self.choose(points_sum, &net_points)];
            nets_new.push(net_chosen.clone());
        }

        // Randomly choose nets to cross proportionally by fitness
        while nets_new.len() < self.population_size {
            let net_chosen_a = &self.nets[self.choose(points_sum, &net_points)];
            let net_chosen_b = &self.nets[self.choose(points_sum, &net_points)];
            let net_new = net_chosen_a.cross_into_new_net(net_chosen_b, &self.mutation_params);
            nets_new.push(net_new);
        }
        self.nets = nets_new;
    }

    // TODO: Replace this with more efficient choosing mechanism
    fn choose(&self, points_sum: f32, net_points: &[f32]) -> usize {
        let choice = thread_rng().gen::<f32>() * points_sum;
        let mut sum = 0.0_f32;
        for (i, &points_cur) in net_points.iter().enumerate() {
            sum += points_cur;
            if sum > choice { return i; }
        }
        panic!();
    }
}
