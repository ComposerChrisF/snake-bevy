//! Handle player input and translate it into movement.
//! Note that the approach used here is simple for demonstration purposes.
//! If you want to move the player in a smoother way,
//! consider using a [fixed timestep](https://github.com/bevyengine/bevy/blob/main/examples/movement/physics_in_fixed_timestep.rs).

use std::collections::VecDeque;

use bevy::prelude::*;
use bevy_ecs_tilemap::map::TilemapId;
use bevy_ecs_tilemap::map::TilemapSize;
use bevy_ecs_tilemap::map::TilemapTexture;
use bevy_ecs_tilemap::map::TilemapTileSize;
use bevy_ecs_tilemap::map::TilemapType;
use bevy_ecs_tilemap::prelude::get_tilemap_center_transform;
use bevy_ecs_tilemap::tiles::TileBundle;
use bevy_ecs_tilemap::tiles::TilePos;
use bevy_ecs_tilemap::tiles::TileStorage;
use bevy_ecs_tilemap::tiles::TileTextureIndex;
use bevy_ecs_tilemap::TilemapBundle;
use bevy_ecs_tilemap::TilemapPlugin;

use crate::screen::Screen;
use crate::snake_game;
use crate::snake_game::GameState;
use crate::AppSet;

use super::assets::HandleMap;
use super::assets::ImageKey;
use super::assets::SfxKey;
use super::audio::sfx::PlaySfx;

#[derive(Event, Debug)]
pub struct UpdateScore(usize);



#[derive(Event, Debug)]
pub struct SpawnLevel;


#[derive(Component)]
struct MySnakeGame {
    snake_game: snake_game::SnakeGame,
    location_apple_prev: snake_game::GridPoint,
    location_tail_prev: snake_game::GridPoint,
}

pub(super) fn plugin(app: &mut App) {
    // Register (i.e. record) what movement the player takes via keyboard/etc.
    app.register_type::<SnakeMovementController>();
    app.add_systems(Update, record_movement_controller.in_set(AppSet::RecordInput));

    // Apply movement based on controls.
    app.add_systems(Update, apply_movement.in_set(AppSet::Update));

    // We make use of these Bevy plugins:
    app.add_plugins(TilemapPlugin);

    // We watch for these events:
    app.observe(spawn_level);
    app.observe(update_score);
}




#[derive(Reflect, Copy, Clone, Default, PartialEq, Eq)]
pub enum Dir {
    #[default]
    Up,
    Down,
    Left,
    Right,
}


impl Dir {
    pub fn to_snake_direction(self) -> snake_game::Direction {
        match self {
            Dir::Up    => snake_game::Direction::North,
            Dir::Down  => snake_game::Direction::South,
            Dir::Left  => snake_game::Direction::West,
            Dir::Right => snake_game::Direction::East,
        }
    }
}

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct SnakeMovementController {
    player_movement_intent: Option<Dir>,
    is_paused: bool,
}

fn record_movement_controller(
    input: Res<ButtonInput<KeyCode>>,
    mut controller_query: Query<(&mut SnakeMovementController, &mut LastUpdate)>,
) {
    // Collect directional input.
    let mut player_movement_intent = None;
    let mut should_reset_timer = false;
    // FUTURE: Ignore reversing direction, since this always produces a crash
    if input.pressed(KeyCode::KeyW) || input.pressed(KeyCode::ArrowUp) {
        player_movement_intent = Some(Dir::Up);
        if input.just_pressed(KeyCode::KeyW) || input.just_pressed(KeyCode::ArrowUp)    { should_reset_timer = true; }
    }
    if input.pressed(KeyCode::KeyS) || input.pressed(KeyCode::ArrowDown) {
        player_movement_intent = Some(Dir::Down);
        if input.just_pressed(KeyCode::KeyS) || input.just_pressed(KeyCode::ArrowDown)  { should_reset_timer = true; }
    }
    if input.pressed(KeyCode::KeyA) || input.pressed(KeyCode::ArrowLeft) {
        player_movement_intent = Some(Dir::Left);
        if input.just_pressed(KeyCode::KeyA) || input.just_pressed(KeyCode::ArrowLeft)  { should_reset_timer = true; }
    }
    if input.pressed(KeyCode::KeyD) || input.pressed(KeyCode::ArrowRight) {
        player_movement_intent = Some(Dir::Right);
        if input.just_pressed(KeyCode::KeyD) || input.just_pressed(KeyCode::ArrowRight) { should_reset_timer = true; }
    }

    let mut should_toggle_pause = false;
    if input.just_pressed(KeyCode::KeyP) || input.just_pressed(KeyCode::Pause) { should_toggle_pause = true; }

    // Apply movement intent to controllers.
    let player_intends_to_move = player_movement_intent.is_some();
    let player_provided_input = player_intends_to_move || should_toggle_pause;
    if player_provided_input {
        for (mut controller, mut last_update) in &mut controller_query {
            if player_intends_to_move { 
                controller.player_movement_intent = player_movement_intent; 
                if should_reset_timer { *last_update = LastUpdate(0.0); }
            }
            if should_toggle_pause { controller.is_paused = !controller.is_paused; }
        }
    }
}



const TILE_CRASH:         u32 = 0;
const TILE_APPLE:         u32 = 1;
const TILE_WALL:          u32 = 2;
const _TILE_SNAKE_HEAD:   u32 = 3;
const TILE_SNAKE_BODY:    u32 = 4;
const TILE_SNAKE_HEAD_N:  u32 = 5;
const TILE_SNAKE_HEAD_E:  u32 = 6;
const TILE_SNAKE_HEAD_S:  u32 = 7;
const TILE_SNAKE_HEAD_W:  u32 = 8;
const TILE_SNAKE_BODY_NS: u32 = 9;
const TILE_SNAKE_BODY_EW: u32 = 10;
const TILE_SNAKE_TAIL_N:  u32 = 11;
const TILE_SNAKE_TAIL_E:  u32 = 12;
const TILE_SNAKE_TAIL_S:  u32 = 13;
const TILE_SNAKE_TAIL_W:  u32 = 14;
const TILE_SNAKE_BODY_NE: u32 = 15;
const TILE_SNAKE_BODY_SE: u32 = 16;
const TILE_SNAKE_BODY_SW: u32 = 17;
const TILE_SNAKE_BODY_NW: u32 = 18;
// FUTURE: Body containing apple NW/EW/etc.
// FUTURE: Head eating apple N/S/E/W


fn tile_texture_index_of_cell_kind(kind: snake_game::CellKind) -> Option<u32> {
    match kind {
        snake_game::CellKind::Empty => None,
        snake_game::CellKind::Crash => Some(TILE_CRASH),
        snake_game::CellKind::Apple => Some(TILE_APPLE),
        snake_game::CellKind::Wall  => Some(TILE_WALL),
        snake_game::CellKind::Snake => Some(TILE_SNAKE_BODY),
    }
}



fn copy_grid_into_tilemap(grid: &snake_game::Grid, tilemap_entity: Entity, tile_storage: &mut TileStorage, map_size: &TilemapSize, commands: &mut Commands) {
    for x in 0..map_size.x {
        for y in 0..map_size.y {
            let tile_pos = TilePos { x, y };
            let cell = grid.get_cell(snake_game::GridPoint { x: x as i16, y: y as i16 });
            if cell.kind == snake_game::CellKind::Snake { continue; }   // Don't copy the snake; use copy_snake_into_tilemap() for that.
            let tile_texture_index = tile_texture_index_of_cell_kind(cell.kind);
            if let Some(tile_texture_index) = tile_texture_index {
                let tile_entity = commands
                    .spawn(TileBundle {
                        position: tile_pos,
                        tilemap_id: TilemapId(tilemap_entity),
                        ..Default::default()
                    })
                    .insert(TileTextureIndex(tile_texture_index))
                    .id();
                tile_storage.set(&tile_pos, tile_entity);
            };
        }
    }
}

fn dir_of_offset(offset: snake_game::GridPoint) -> Dir {
    match (offset.x, offset.y) {
        ( 0,  1) => Dir::Up,
        ( 1,  0) => Dir::Right,
        ( 0, -1) => Dir::Down,
        (-1,  0) => Dir::Left,
        _ => panic!("Unexpected offset in dir_of_offset()"),
    }
}

fn tile_texture_index_of_tail_and_direction(dir: Dir) -> u32 {
    match dir {
        Dir::Up    => TILE_SNAKE_TAIL_N,
        Dir::Right => TILE_SNAKE_TAIL_E,
        Dir::Down  => TILE_SNAKE_TAIL_S,
        Dir::Left  => TILE_SNAKE_TAIL_W,
    }
}

fn tile_texture_index_of_head_and_direction(dir: Dir) -> u32 {
    match dir {
        Dir::Up    => TILE_SNAKE_HEAD_N,
        Dir::Right => TILE_SNAKE_HEAD_E,
        Dir::Down  => TILE_SNAKE_HEAD_S,
        Dir::Left  => TILE_SNAKE_HEAD_W,
    }
}

fn tile_texture_index_of_prev_and_next_directions(dir_prev: Dir, dir_next: Dir) -> u32 {
    match (dir_prev, dir_next) {
        (Dir::Up, Dir::Up)    => TILE_SNAKE_BODY_NS,
        (Dir::Up, Dir::Right) => TILE_SNAKE_BODY_SE,
        (Dir::Up, Dir::Down)  => TILE_CRASH,
        (Dir::Up, Dir::Left)  => TILE_SNAKE_BODY_SW,

        (Dir::Right, Dir::Up)    => TILE_SNAKE_BODY_NW,
        (Dir::Right, Dir::Right) => TILE_SNAKE_BODY_EW,
        (Dir::Right, Dir::Down)  => TILE_SNAKE_BODY_SW,
        (Dir::Right, Dir::Left)  => TILE_CRASH,

        (Dir::Down, Dir::Up)    => TILE_CRASH,
        (Dir::Down, Dir::Right) => TILE_SNAKE_BODY_NE,
        (Dir::Down, Dir::Down)  => TILE_SNAKE_BODY_NS,
        (Dir::Down, Dir::Left)  => TILE_SNAKE_BODY_NW,

        (Dir::Left, Dir::Up)    => TILE_SNAKE_BODY_NE,
        (Dir::Left, Dir::Right) => TILE_CRASH,
        (Dir::Left, Dir::Down)  => TILE_SNAKE_BODY_SE,
        (Dir::Left, Dir::Left)  => TILE_SNAKE_BODY_EW,
    }
}

fn copy_snake_into_tilemap(snake_locations: &VecDeque<snake_game::GridPoint>, tilemap_entity: Entity, tile_storage: &mut TileStorage, commands: &mut Commands) {
    assert!(snake_locations.len() >= 2);
    let snake_length = snake_locations.len();
    for (i, &pt) in snake_locations.iter().enumerate() {    // Iterates from head (at snake_locations[0]) to tail (at snake_locations[len - 1])
        let is_tail = i == snake_length - 1;
        let is_head = i == 0;
        // Compute texture, based on head/body/tail calculations:
        let tile_texture_index = if is_tail {
            let pt_next = snake_locations[snake_length - 2];
            tile_texture_index_of_tail_and_direction(dir_of_offset(pt_next - pt))
        } else if is_head {
            let pt_prev = snake_locations[1];
            tile_texture_index_of_head_and_direction(dir_of_offset(pt - pt_prev))
        } else {
            let pt_prev = snake_locations[i + 1];
            let pt_next = snake_locations[i - 1];
            tile_texture_index_of_prev_and_next_directions(dir_of_offset(pt - pt_prev), dir_of_offset(pt_next - pt))
        };
        // Now place tiles
        let tile_pos = TilePos { x: pt.x as u32, y: pt.y as u32 };
        let tile_entity = commands
            .spawn(TileBundle {
                position: tile_pos,
                tilemap_id: TilemapId(tilemap_entity),
                ..Default::default()
            })
            .insert(TileTextureIndex(tile_texture_index))
            .id();
        tile_storage.set(&tile_pos, tile_entity);
    }
}


fn spawn_level(
    _trigger: Trigger<SpawnLevel>,
    mut commands: Commands,
    image_handles: Res<HandleMap<ImageKey>>,
) {
    // Create the underlying snake_game--essentially our data model
    let snake_game = snake_game::SnakeGame::new(None);

    // Create and insert the TileMap
    let tilemap_entity = commands.spawn_empty().id();
    let map_size = TilemapSize { x: snake_game.grid.width as u32, y: snake_game.grid.height as u32 };
    let mut tile_storage = TileStorage::empty(map_size);
    let map_type = TilemapType::Square;
    copy_grid_into_tilemap(&snake_game.grid, tilemap_entity, &mut tile_storage, &map_size, &mut commands);
    copy_snake_into_tilemap(&snake_game.snake.locations, tilemap_entity, &mut tile_storage, &mut commands);
    let tile_pixel_size = TilemapTileSize { x: 16.0, y: 16.0 };
    let grid_size = tile_pixel_size.into();
    let texture_handle: Handle<Image> = image_handles[&ImageKey::SnakeTiles].clone_weak(); //asset_server.load("images/snake_tiles.png");
    commands.entity(tilemap_entity).insert(
        TilemapBundle {
            grid_size,
            size: map_size,
            storage: tile_storage,
            map_type,
            texture: TilemapTexture::Single(texture_handle),
            tile_size: tile_pixel_size,
            transform: get_tilemap_center_transform(&map_size, &grid_size, &map_type, 0.0),
            ..Default::default()
        }
    ).insert(StateScoped(Screen::Playing));

    // Init and insert the MySnakeGame
    let location_apple_prev = snake_game.apple.location;
    let location_tail_prev = snake_game.snake.locations[snake_game.snake.locations.len() - 1];
    commands.spawn((
        MySnakeGame { 
            snake_game,
            location_apple_prev,
            location_tail_prev,
        },
        LastUpdate(0.0),
        SnakeMovementController { player_movement_intent: None, is_paused: false },
        StateScoped(Screen::Playing),
    ));

    // The Score
    commands.spawn((
        TextBundle::from_section(
            "Score: 0",
            TextStyle {
                font_size: 20.0,
                color: Color::WHITE,
                ..default()
            },
        )
        .with_text_justify(JustifyText::Center)
        .with_style(bevy::ui::Style {
             position_type: PositionType::Absolute,
             //align_items: AlignItems::Center,
             //align_content: AlignContent::Center,
             left: Val::Percent(0.0),
             width: Val::Percent(100.0),
             top: Val::Px(0.0),
             ..default()
        }),
        Score,
        StateScoped(Screen::Playing),
    ));
}

#[derive(Component)]
struct Score;

fn update_score(
    trigger: Trigger<UpdateScore>,
    mut query: Query<&mut Text, With<Score>>,
) {
    info!("update_score(): {}", trigger.event().0);
    let new_score = trigger.event().0;
    for mut text in query.iter_mut() {
        text.sections[0].value = format!("Score: {new_score}");
    }
}


#[derive(Component)]
struct LastUpdate(f64);

fn apply_movement(
    mut commands: Commands,
    time: Res<Time>,
    mut snake_query: Query<(&mut MySnakeGame, &mut LastUpdate, &SnakeMovementController)>,
    mut tilemap_query: Query<(&mut TileStorage, Entity)>,
    mut tile_texture_query: Query<&mut TileTextureIndex>,
) {
    for (mut my_snake_game, mut last_update, movement) in snake_query.iter_mut() {
        if movement.is_paused { continue; } 
        if let Some(dir) = movement.player_movement_intent {
            let current_time = time.elapsed_seconds_f64();
            if current_time - last_update.0 > 0.1 {
                let prev_apples_eaten = my_snake_game.snake_game.apples_eaten;
                let prev_snake_len = my_snake_game.snake_game.snake.locations.len();
                let prev_game_state = my_snake_game.snake_game.state;
                my_snake_game.snake_game.move_snake(dir.to_snake_direction(), None);
                let (tile_storage, tilemap_entity) = tilemap_query.get_single_mut().unwrap();
                update_tilemap(&mut commands, &mut my_snake_game, tilemap_entity, tile_storage, &mut tile_texture_query);
                if prev_apples_eaten != my_snake_game.snake_game.apples_eaten {
                    commands.trigger(UpdateScore(my_snake_game.snake_game.apples_eaten));
                }

                // Generate sound
                if prev_game_state != my_snake_game.snake_game.state && my_snake_game.snake_game.state == GameState::GameOver {
                    commands.trigger(PlaySfx::Key(SfxKey::Crash(0)));
                } else if prev_apples_eaten != my_snake_game.snake_game.apples_eaten {
                    commands.trigger(PlaySfx::Key(SfxKey::Eating(0)));
                } else if prev_snake_len != my_snake_game.snake_game.snake.locations.len() {
                    commands.trigger(PlaySfx::Key(SfxKey::Growing(0)));
                } else if my_snake_game.snake_game.state != GameState::GameOver {
                    commands.trigger(PlaySfx::Key(SfxKey::Tick(0)));
                }
                last_update.0 = current_time;
            }
        }
    }
}


fn update_tilemap(
    commands: &mut Commands,
    my_snake_game: &mut Mut<MySnakeGame>,
    tilemap_entity: Entity,
    mut tile_storage: Mut<TileStorage>,
    tile_texture_query: &mut Query<&mut TileTextureIndex>,
) {
    let pt_tail_prev = my_snake_game.location_tail_prev;
    let pt_apple_prev = my_snake_game.location_apple_prev;
    let snake_game = &mut my_snake_game.snake_game;

    let is_game_over = snake_game.state == snake_game::GameState::GameOver;
    //if is_game_over { return; }   // We *could*n short-circuit updating, but only if we track whether we've already updated once after a game over
    
    // We don't need to update the *entire* map, just the locations where things might have 
    // changed (see comments on SnakeGame::move_snake() for details):
    // 1. The following snake location tiles must be recomputed: 
    //     a. Head of the snake
    //         i. Normally based on movement from previous tile...
    //         ii. ...unless GameState is GameOver, then the head of the snake should be a crash.
    //     b. the tile that previously had been the head of the snake
    //     c. the tile that previously had been the tail becomes empty
    //     d. the new tail (based on the movement to the next tile)
    // 2. If the apple moved, then 
    //     a. the old apple location must either be empty or a snake
    //     b. the new apple location is an apple.
    let locations = &snake_game.snake.locations;
    let len = locations.len();
    let pt_head = locations[0];
    let pt_head_prev = locations[1];
    let pt_tail = locations[len - 1];
    let pt_almost_tail = locations[len - 2];
    let pt_apple = snake_game.apple.location;

    // Snake head
    let tile_texture_index_head = if is_game_over { 
        TILE_CRASH 
    } else { 
        let dir_head = dir_of_offset(pt_head - pt_head_prev);
        tile_texture_index_of_head_and_direction(dir_head)
    };
    update_tilemap_at_point(pt_head, Some(tile_texture_index_head), commands, tilemap_entity, &mut tile_storage, tile_texture_query);

    // Previous head (only if snake is longer than 2)
    let has_prev_head_that_is_not_tail = len > 2;
    if has_prev_head_that_is_not_tail {
        let pt_head_prev_prev = locations[2];
        let dir_head = dir_of_offset(pt_head - pt_head_prev);
        let dir_head_prev = dir_of_offset(pt_head_prev - pt_head_prev_prev);
        let tile_texture_index_head_prev = tile_texture_index_of_prev_and_next_directions(dir_head_prev, dir_head);
        update_tilemap_at_point(pt_head_prev, Some(tile_texture_index_head_prev), commands, tilemap_entity, &mut tile_storage, tile_texture_query);
    }

    // Snake Tail
    let has_tail_moved = pt_tail_prev != pt_tail;
    if has_tail_moved {
        // Erase old tail, unless the snake head replaces it (in which it's already been placed in tilemap)
        let has_head_replaced_tail = pt_head == pt_tail_prev;
        if !pt_tail_prev.is_zero() && !has_head_replaced_tail {
            update_tilemap_at_point(pt_tail_prev, None, commands, tilemap_entity, &mut tile_storage, tile_texture_query);
        }

        // Now update the tile for the new tail
        let dir_tail = dir_of_offset(pt_almost_tail - pt_tail);
        let tile_texture_index_tail = tile_texture_index_of_tail_and_direction(dir_tail);
        update_tilemap_at_point(pt_tail, Some(tile_texture_index_tail), commands, tilemap_entity, &mut tile_storage, tile_texture_query);
        my_snake_game.location_tail_prev = pt_tail;
    }

    // Apple
    let has_apple_moved = pt_apple_prev != pt_apple;
    if has_apple_moved {
        // Erase old apple unless eaten by snake (in which case the tile has already been covered by snake head)
        let was_old_apple_eaten = !pt_apple_prev.is_zero() && pt_apple_prev == pt_head;
        if !was_old_apple_eaten {
            update_tilemap_at_point(pt_apple_prev, None, commands, tilemap_entity, &mut tile_storage, tile_texture_query);
        }
        // Draw new apple
        update_tilemap_at_point(pt_apple, Some(TILE_APPLE), commands, tilemap_entity, &mut tile_storage, tile_texture_query);
        my_snake_game.location_apple_prev = pt_apple;
    }
}

fn update_tilemap_at_point(
    pt: snake_game::GridPoint,
    tile_texture_index: Option<u32>,
    commands: &mut Commands,
    tilemap_entity: Entity,
    tile_storage: &mut Mut<TileStorage>,
    tile_texture_query: &mut Query<&mut TileTextureIndex>,
) {
    let tile_position = TilePos { x: pt.x as u32, y: pt.y as u32 };
    let tile = tile_storage.get(&tile_position);
    //info!("grid_change: location={pt:#?}, CellKind={:#?}, tile found={tile:?}", cell.kind);
    match (tile_texture_index, tile) {
        (None, None) => { /* Nothing to do. */ info!("BUG: Updating a non-existent tile to be a non-existent tile."); }
        (None, Some(tile)) => { 
            // Remove from Tilemap
            tile_storage.remove(&tile_position);
            commands.entity(tile).despawn();
        }
        (Some(tile_texture_index), None) => { 
            // Create new tile entity
            let tile_entity = commands
                .spawn(TileBundle {
                    position: tile_position,
                    tilemap_id: TilemapId(tilemap_entity),
                    ..Default::default()
                })
                .insert(TileTextureIndex(tile_texture_index))
                .id();
            // Add tile entity to Tilemap
            tile_storage.set(&tile_position, tile_entity);
        }
        (Some(tile_texture_index), Some(tile)) => { 
            // Change texture of tile already in Tilemap
            if let Ok(mut current_texture) = tile_texture_query.get_mut(tile) {
                current_texture.0 = tile_texture_index;
            } else {
                info!("BUG: No texture found for tile position={:#?}", pt);
            }
        }
    }
}
