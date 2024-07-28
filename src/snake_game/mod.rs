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
        let mut new_grid = Grid { 
            width: Self::WIDTH, 
            height: Self::HEIGHT, 
            cells: Vec::<Cell>::with_capacity((Self::WIDTH * Self::HEIGHT) as usize),
        };
        for _ in 0..(Self::WIDTH * Self::HEIGHT) {
            new_grid.cells.push(Cell { kind: CellKind::Empty });
        }
        new_grid.restart();
        new_grid
    }
    pub fn restart(&mut self) {
        let cells = &mut self.cells;
        for x in 0..Self::WIDTH {
            for y in 0..Self::HEIGHT {
                Self::set_cell(cells, x, y, CellKind::Empty);
            }
        }
        for x in 0..Self::WIDTH {
            Self::set_cell(cells, x, 0, CellKind::Wall);
            Self::set_cell(cells, x, Self::HEIGHT - 1, CellKind::Wall);
        }
        for y in 0..Self::HEIGHT {
            Self::set_cell(cells, 0, y, CellKind::Wall);
            Self::set_cell(cells, Self::WIDTH - 1, y, CellKind::Wall);
        }
        //TESTING: Self::set_cell(cells, 0, 0, CellKind::Crash);        // So we can see where origin is
    }
    fn set_cell(cells: &mut [Cell], x: i16, y: i16, kind: CellKind) {
        if x < 0 || y < 0 || x >= Self::WIDTH || y >= Self::HEIGHT { return; }
        let i = y * Self::WIDTH + x;
        cells[i as usize] = Cell { kind };
    }
    pub fn is_in_bounds(&self, pt: Point) -> bool {
        pt.x >= 0 && pt.y >= 0 && pt.x < Self::WIDTH && pt.y < Self::HEIGHT
    }
    pub fn get_cell(&self, pt: Point) -> &Cell {
        if !self.is_in_bounds(pt) { return &self.cells[0]; }
        let i = pt.y * Self::WIDTH + pt.x;
        &self.cells[i as usize]
    }
    pub fn get_cell_mut(&mut self, pt: Point) -> &mut Cell {
        if !self.is_in_bounds(pt)  { return &mut self.cells[0]; }
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
            if self.get_cell(loc).kind != CellKind::Empty { continue; }
            return loc;
        }
        panic!("No room for apple!");
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum CellKind {
    Empty,
    Apple,
    SnakeHeadN,
    SnakeHeadE,
    SnakeHeadS,
    SnakeHeadW,
    SnakeBodyNS,
    SnakeBodyEW,
    SnakeBodyNE,
    SnakeBodySE,
    SnakeBodySW,
    SnakeBodyNW,
    SnakeTailN,
    SnakeTailE,
    SnakeTailS,
    SnakeTailW,
    Wall,
    Crash,
}
// FUTURE: Head North/South/East/West; Tail N/S/E/W; body NS/EW
// FUTURE: Head eating apple N/S/E/W; body containing apple NW/EW


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
        Self::from_index(rand::thread_rng().gen_range(0..3))
    }

    pub fn from_index(i: usize) -> Direction {
        match i {
            0 => Direction::North,
            1 => Direction::East,
            2 => Direction::South,
            3 => Direction::West,
            _ => panic!("Bad i in from_index()"),
        }
    }
    pub fn to_point(self) -> Point {
        match self {
            Direction::North => Point { x: 0, y: 1, },
            Direction::East  => Point { x: 1, y: 0, },
            Direction::South => Point { x: 0, y: -1, },
            Direction::West => Point { x: -1, y: 0, },
        }
    }

    pub fn to_index(self) -> usize {
        match self {
            Direction::North => 0,
            Direction::East  => 1,
            Direction::South => 2,
            Direction::West => 3,
        }
    }
    
    fn of_offset(offset: Point) -> Direction {
        match (offset.x, offset.y) {
            (0, 1) => Direction::North,
            (1, 0) => Direction::East,
            (0, -1) => Direction::South,
            (-1, 0) => Direction::West,
            _ => panic!("Unexpected offset value: {offset:?}"),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Debug, Default)]
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
    pub fn add(self, other: Point) -> Point {
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

fn dir_to_snake_head(dir: Direction) -> CellKind {
    match dir {
        Direction::North => CellKind::SnakeHeadN,
        Direction::East  => CellKind::SnakeHeadE,
        Direction::South => CellKind::SnakeHeadS,
        Direction::West  => CellKind::SnakeHeadW,
    }
}

fn dir_to_snake_tail(dir: Direction) -> CellKind {
    match dir {
        Direction::North => CellKind::SnakeTailN,
        Direction::East  => CellKind::SnakeTailE,
        Direction::South => CellKind::SnakeTailS,
        Direction::West  => CellKind::SnakeTailW,
    }
}

fn dir_and_head_to_body(dir: Direction, kind_old_head: CellKind) -> CellKind {
    match (dir, kind_old_head) {
        (Direction::North, CellKind::SnakeHeadN) => CellKind::SnakeBodyNS,
        (Direction::North, CellKind::SnakeHeadE) => CellKind::SnakeBodyNW,
        (Direction::North, CellKind::SnakeHeadS) => CellKind::SnakeBodyNS,  // Death!
        (Direction::North, CellKind::SnakeHeadW) => CellKind::SnakeBodyNE,

        (Direction::East, CellKind::SnakeHeadN) => CellKind::SnakeBodySE,
        (Direction::East, CellKind::SnakeHeadE) => CellKind::SnakeBodyEW,
        (Direction::East, CellKind::SnakeHeadS) => CellKind::SnakeBodyNE,
        (Direction::East, CellKind::SnakeHeadW) => CellKind::SnakeBodyEW,   // Death!

        (Direction::South, CellKind::SnakeHeadN) => CellKind::SnakeBodyNS,  // Death!
        (Direction::South, CellKind::SnakeHeadE) => CellKind::SnakeBodySW,
        (Direction::South, CellKind::SnakeHeadS) => CellKind::SnakeBodyNS,
        (Direction::South, CellKind::SnakeHeadW) => CellKind::SnakeBodySE,
        
        (Direction::West, CellKind::SnakeHeadN) => CellKind::SnakeBodySW,
        (Direction::West, CellKind::SnakeHeadE) => CellKind::SnakeBodyEW,   // Death!
        (Direction::West, CellKind::SnakeHeadS) => CellKind::SnakeBodyNW,
        (Direction::West, CellKind::SnakeHeadW) => CellKind::SnakeBodyEW,

        _ => panic!("unexpected kind for dir_and_head_to_body(): {kind_old_head:?}"),
    }
}

impl Snake {
    pub(self) fn new(grid: &mut Grid) -> Snake {
        let locations = VecDeque::<Point>::with_capacity(grid.width as usize * grid.height as usize);
        let mut new_snake = Snake { head_location: Point::default(), locations, to_grow: 0 };
        new_snake.restart(grid);
        new_snake
    }
    pub(self) fn restart(&mut self, grid: &mut Grid) {
        self.locations.clear();
        self.to_grow = 0;
        for _ in 0..1000 {
            let tail = grid.rand_point();
            if grid.get_cell(tail).kind != CellKind::Empty { continue; }
            let dir = Direction::random();
            let offset = dir.to_point();
            let head = tail.add(offset);
            if grid.get_cell(head).kind != CellKind::Empty { continue; }
            grid.get_cell_mut(head).kind = dir_to_snake_head(dir);
            grid.get_cell_mut(tail).kind = dir_to_snake_tail(dir);
            self.locations.push_front(tail);
            self.locations.push_front(head);
            self.head_location = head;
            return;
        }
        panic!("No room for snake!");
    }

    pub fn length(&self) -> usize {
        self.locations.len()
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub struct Apple {
    pub location: Point,
}



#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum GameState {
    Running,
    GameOver,
}


#[derive(Copy, Clone, PartialEq, Eq, Serialize, Deserialize, Debug)]
pub enum PlaybackEvents {
    NewGame,                    // Initialize grid
    NewAppleLocation(Point),    // Place apple
    MoveSnake(Direction),       // Move snake
    GameOver,
}


#[derive(Serialize, Deserialize, Debug)]
pub struct SnakeGame {
    pub grid: Grid,
    pub snake: Snake,
    pub apple: Apple,
    pub apples_eaten: usize,
    pub state: GameState,
    pub playback_events: Vec<PlaybackEvents>,
    pub grid_changes: Vec<Point>,
}

impl SnakeGame {
    pub const GROW_INCREMENT: usize = 5;

    pub fn new(new_apple_location: Option<Point>) -> Self {
        let mut grid = Grid::new();
        let snake = Snake::new(&mut grid);
        let apple = Apple { 
            location: match new_apple_location {
                None => grid.new_viable_apple_location(),
                Some(pt) => pt,
            },
        };
        let apple_cell = grid.get_cell_mut(apple.location);
        apple_cell.kind = CellKind::Apple;
        let mut new_grid = Self {
            grid,
            snake,
            apple,
            apples_eaten: 0,
            state: GameState::Running,
            playback_events: Vec::with_capacity(256),
            grid_changes: Vec::with_capacity(4),
        };
        new_grid.playback_events.clear();
        new_grid.playback_events.push(PlaybackEvents::NewGame);
        new_grid.playback_events.push(PlaybackEvents::NewAppleLocation(apple.location));
        new_grid
    }

    pub fn restart(&mut self, new_apple_location: Option<Point>) {
        self.grid.restart();
        self.snake.restart(&mut self.grid);
        self.apple.location = match new_apple_location {
            None => self.grid.new_viable_apple_location(),
            Some(pt) => pt,
        };
        self.apples_eaten = 0;
        self.state = GameState::Running;
        self.playback_events.clear();
        self.playback_events.push(PlaybackEvents::NewGame);
        self.playback_events.push(PlaybackEvents::NewAppleLocation(self.apple.location));
    }

    /// Move the snake, typically from user input.  For playback, `new_apple_location` allows
    /// provision of where the next apple tile is place.  For live play, `new_apple_location`
    /// sould be `None`, in which case a random location is chosen.
    /// 
    /// One must observe what has changed in the `snake_game` after `move_snake()` is called,
    /// as this informs what happened.  E.g. `snake_game.state` may have changed.  Also,
    /// `snake_game.grid_changes` contains the locations in the grid that have changed as a
    /// result of calling `move_snake()`, and informs the caller what needs to change in their
    /// visualization.
    pub fn move_snake(&mut self, direction: Direction, new_apple_location: Option<Point>) {
        //info!("move_snake({direction:#?}, {new_apple_location:#?}); snake.to_grow={}; GameState={:?}", self.snake.to_grow, self.state);
        self.grid_changes.clear();
        if self.state != GameState::Running { return; }
        self.playback_events.push(PlaybackEvents::MoveSnake(direction));
        let offset = direction.to_point();
        let new_location = self.snake.head_location.add(offset);
        let old_head_cell = self.grid.get_cell_mut(self.snake.head_location);
        old_head_cell.kind = dir_and_head_to_body(direction, old_head_cell.kind);
        self.grid_changes.push(self.snake.head_location);
        let new_cell = self.grid.get_cell_mut(new_location);
        let kind_hit = new_cell.kind;
        new_cell.kind = dir_to_snake_head(direction);
        self.grid_changes.push(new_location);
        match kind_hit {
            CellKind::Apple => {
                self.apples_eaten += 1;
                self.apple.location = match new_apple_location {
                    None => self.grid.new_viable_apple_location(),
                    Some(pt) => pt,
                };
                let new_apple_cell = self.grid.get_cell_mut(self.apple.location);
                new_apple_cell.kind = CellKind::Apple;
                self.grid_changes.push(self.apple.location);
                self.playback_events.push(PlaybackEvents::NewAppleLocation(self.apple.location));
                self.snake.to_grow += Self::GROW_INCREMENT;
            }
            CellKind::Empty => {}
            _ => {
                self.playback_events.push(PlaybackEvents::GameOver);
                self.state = GameState::GameOver;
                new_cell.kind = CellKind::Crash;
                self.grid_changes.push(new_location);
            }
        };

        // Push on new Head
        self.snake.locations.push_front(new_location);
        self.snake.head_location = new_location;

        // Move tail, if needed
        if self.snake.to_grow == 0 {
            // Snake keeps same size, so we pop off the tail to keep the same length, since we already pushed on a new head.
            let old_tail_location = self.snake.locations.pop_back().unwrap();
            let old_tail_cell = self.grid.get_cell_mut(old_tail_location);
            old_tail_cell.kind = CellKind::Empty;
            self.grid_changes.push(old_tail_location);
            // Now fix up the old body segment to now be a tail
            let new_tail_location = Self::peek_back(&self.snake.locations, 0);
            let new_first_body_location = Self::peek_back(&self.snake.locations, 1);
            let tail_direction = Direction::of_offset(Point { x: new_first_body_location.x - new_tail_location.x, y: new_first_body_location.y - new_tail_location.y });
            let new_tail_cell = self.grid.get_cell_mut(new_tail_location);
            new_tail_cell.kind = dir_to_snake_tail(tail_direction);
            self.grid_changes.push(new_tail_location);
        } else {
            self.snake.to_grow -= 1;
        };
    }

    fn peek_back(locations: &VecDeque<Point>, offset_from_end: usize) -> Point {
        locations[locations.len() - 1 - offset_from_end]
    }

    // FUTURE: For snake body, provide distance from tail? I.e. how long until tile disappears?
    pub fn wall_and_body_distances(&self) -> ([i16; 4], [i16; 4]) {
        let mut dist_walls: [i16; 4] = [0; 4];
        let mut dist_snake: [i16; 4] = [0; 4];
        let head = self.snake.head_location;
        for dir in [Direction::North, Direction::East, Direction::South, Direction::West] {
            let i = dir.to_index();
            dist_walls[i] = self.distance_to(head, dir, &[CellKind::Wall]);
            dist_snake[i] = self.distance_to(head, dir, &[
                CellKind::SnakeHeadN, CellKind::SnakeHeadE, CellKind::SnakeHeadS, CellKind::SnakeHeadW,
                CellKind::SnakeBodyNS, CellKind::SnakeBodyEW, 
                CellKind::SnakeBodyNE, CellKind::SnakeBodySW, CellKind::SnakeBodySE, CellKind::SnakeBodyNW,
                CellKind::SnakeTailN, CellKind::SnakeTailE, CellKind::SnakeTailS, CellKind::SnakeTailW,
            ]);
        }
        (dist_walls, dist_snake)
    }
    fn distance_to(&self, pt_start: Point, direction: Direction, target: &[CellKind]) -> i16 {
        let offset = direction.to_point();
        let mut distance = 0;
        let mut pt_test = pt_start.add(offset);
        while self.grid.is_in_bounds(pt_test) && !target.contains(&self.grid.get_cell(pt_test).kind) {
            distance += 1;
            pt_test = pt_start.add(offset);
        }
        distance
    }
}