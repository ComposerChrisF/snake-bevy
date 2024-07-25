use std::collections::VecDeque;
use rand::Rng;
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Grid {
    pub width: i16,
    pub height: i16,
    pub cells: Vec::<Cell>,
}

impl Grid {
    pub const WIDTH: i16 = 25;
    pub const HEIGHT: i16 = 25;
    pub const _MAX_WIDTH_HEIGHT: i16 = 25; // Maximum of width & height, i.e. WIDTH.max(HEIGHT)

    pub fn new() -> Grid {
        let mut cells = Vec::<Cell>::with_capacity((Self::WIDTH * Self::HEIGHT) as usize);
        for x in 0..Self::WIDTH {
            for y in 0..Self::HEIGHT {
                Self::set_cell(&mut cells, x, y, CellKind::Empty);
            }
        }
        for x in 0..Self::WIDTH {
            Self::set_cell(&mut cells, x, 0, CellKind::Wall);
            Self::set_cell(&mut cells, x, Self::HEIGHT - 1, CellKind::Wall);
        }
        for y in 0..Self::HEIGHT {
            Self::set_cell(&mut cells, 0, y, CellKind::Wall);
            Self::set_cell(&mut cells, Self::WIDTH - 1, y, CellKind::Wall);
        }
        Grid {
            width: Self::WIDTH,
            height: Self::HEIGHT,
            cells
        }
    }
    fn set_cell(cells: &mut Vec::<Cell>, x: i16, y: i16, kind: CellKind) {
        if x < 0 || y < 0 || x >= Self::WIDTH || y >= Self::HEIGHT { return; }
        let i = y * Self::WIDTH + x;
        cells[i as usize] = Cell { kind };
    }
    pub fn is_in_bounds(&self, pt: &Point) -> bool {
        pt.x >= 0 && pt.y >= 0 && pt.x < Self::WIDTH && pt.y < Self::HEIGHT
    }
    pub fn get_cell(&self, pt: &Point) -> &Cell {
        if !self.is_in_bounds(&pt) { return &self.cells[0]; }
        let i = pt.y * Self::WIDTH + pt.x;
        &self.cells[i as usize]
    }
    pub fn get_cell_mut(&mut self, pt: &Point) -> &mut Cell {
        if !self.is_in_bounds(&pt)  { return &mut self.cells[0]; }
        let i = pt.y * Self::WIDTH + pt.x;
        &mut self.cells[i as usize]
    }
    pub fn rand_point(&self) -> Point {
        Point {
            x: rand::thread_rng().gen_range(1..(self.width - 2)),
            y: rand::thread_rng().gen_range(1..(self.height - 2)),
        }
    }
    pub fn new_viable_apple_location(&self) -> Point {
        for _ in 0..10000 {
            let loc = self.rand_point();
            if self.get_cell(&loc).kind != CellKind::Empty { continue; }
            return loc;
        }
        panic!("No room for apple!");
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum CellKind {
    Empty,
    Apple,
    Snake,
    Wall,
}

#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub struct Cell {
    pub kind: CellKind
}

#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum Direction {
    North,
    East,
    South,
    West,
}

impl Direction {
    pub fn random() -> Direction {
        match rand::thread_rng().gen_range(0..3) {
            0 => Direction::North,
            1 => Direction::East,
            2 => Direction::South,
            3 => Direction::West,
            _ => panic!("Bad range for gen_range() in Direction::random()"),
        }
    }

    pub fn to_point(&self) -> Point {
        match self {
            Direction::North => Point { x: 0, y: -1, },
            Direction::East  => Point { x: 1, y: 0, },
            Direction::South => Point { x: 0, y: 1, },
            Direction::West => Point { x: -1, y: 0, },
        }
    }

    pub fn to_index(&self) -> usize {
        match self {
            Direction::North => 0,
            Direction::East  => 1,
            Direction::South => 2,
            Direction::West => 3,
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub struct Point {
    pub x: i16,
    pub y: i16,
}

impl Point {
    pub fn new(x: i16, y: i16) -> Self {
        Self {
            x, y
        }
    }
    pub fn add(&self, other: &Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Snake {
    pub head_location: Point,
    pub locations: VecDeque<Point>,
    pub to_grow: usize,
}

impl Snake {
    pub(self) fn new(grid: &mut Grid) -> Snake {
        let mut locations = VecDeque::<Point>::with_capacity(grid.width as usize * grid.height as usize);
        for _ in 0..1000 {
            let head = grid.rand_point();
            if grid.get_cell(&head).kind != CellKind::Empty { continue; }
            let dir = Direction::random();
            let offset = dir.to_point();
            let tail = Point::add(&head, &offset);
            if grid.get_cell(&tail).kind != CellKind::Empty { continue; }
            grid.get_cell_mut(&head).kind = CellKind::Snake;
            grid.get_cell_mut(&tail).kind = CellKind::Snake;
            locations.push_back(tail);
            locations.push_back(head);
            return Snake { head_location: head, locations, to_grow: 0 };
        }
        panic!("No room for snake!");
    }

    pub fn length(&self) -> usize {
        self.locations.len()
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Apple {
    pub location: Point,
}



#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum GameState {
    Running,
    GameOver,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SnakeGame {
    pub grid: Grid,
    pub snake: Snake,
    pub apple: Apple,
    pub apples_eaten: usize,
    pub state: GameState,
}

impl SnakeGame {
    pub const GROW_INCREMENT: usize = 5;

    pub fn new() -> Self {
        let mut grid = Grid::new();
        let snake = Snake::new(&mut grid);
        let apple = Apple { location: grid.new_viable_apple_location(), };
        let apple_cell = grid.get_cell_mut(&apple.location);
        apple_cell.kind = CellKind::Apple;
        Self {
            grid,
            snake,
            apple,
            apples_eaten: 0,
            state: GameState::Running,
        }
    }

    pub fn move_snake(&mut self, direction: Direction) -> Option<CellKind> {
        if self.state != GameState::Running { return None; }
        let offset = direction.to_point();
        let new_location = self.snake.head_location.add(&offset);
        let new_cell = self.grid.get_cell_mut(&new_location);
        let kind_hit = new_cell.kind;
        match kind_hit {
            CellKind::Apple => {
                new_cell.kind = CellKind::Snake;
                self.apples_eaten += 1;
                self.apple.location = self.grid.new_viable_apple_location();
                let new_apple = self.grid.get_cell_mut(&self.apple.location);
                new_apple.kind = CellKind::Apple;
                self.snake.to_grow += Self::GROW_INCREMENT;
            }
            CellKind::Empty => {
                new_cell.kind = CellKind::Snake;
            }
            CellKind::Snake | CellKind::Wall => {
                self.state = GameState::GameOver;
            }
        };
        if self.snake.to_grow == 0 {
            // Snake keeps same size, so we pop off the tail, then push the new head to simulate
            // movement
            let old_tail_location = self.snake.locations.pop_front().unwrap();
            let old_tail_cell = self.grid.get_cell_mut(&old_tail_location);
            old_tail_cell.kind = CellKind::Empty;
        } else {
            self.snake.to_grow -= 1;
        }
        self.snake.locations.push_front(new_location);
        Some(kind_hit)
    }

    pub fn wall_and_body_distances(&self) -> ([i16; 4], [i16; 4]) {
        let mut dist_walls: [i16; 4] = [0; 4];
        let mut dist_snake: [i16; 4] = [0; 4];
        let head = self.snake.head_location;
        for dir in [Direction::North, Direction::East, Direction::South, Direction::West] {
            let i = dir.to_index();
            dist_walls[i] = self.distance_to(&head, dir, CellKind::Wall);
            dist_snake[i] = self.distance_to(&head, dir, CellKind::Snake);
        }
        (dist_walls, dist_snake)
    }
    fn distance_to(&self, pt_start: &Point, direction: Direction, target: CellKind) -> i16 {
        let offset = direction.to_point();
        let mut distance = 0;
        let mut pt_test = pt_start.add(&offset);
        while self.grid.is_in_bounds(&pt_test) && self.grid.get_cell(&pt_test).kind != target {
            distance += 1;
            pt_test = pt_start.add(&offset);
        }
        distance
    }
}