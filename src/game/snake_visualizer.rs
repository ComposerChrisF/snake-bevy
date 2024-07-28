//! Handle player input and translate it into movement.
//! Note that the approach used here is simple for demonstration purposes.
//! If you want to move the player in a smoother way,
//! consider using a [fixed timestep](https://github.com/bevyengine/bevy/blob/main/examples/movement/physics_in_fixed_timestep.rs).

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
use crate::AppSet;

use super::spawn::level::SpawnLevel;


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
            Dir::Up => snake_game::Direction::North,
            Dir::Down => snake_game::Direction::South,
            Dir::Left => snake_game::Direction::West,
            Dir::Right => snake_game::Direction::East,
        }
    }
}

#[derive(Component, Reflect, Default)]
#[reflect(Component)]
pub struct SnakeMovementController(Option<Dir>);

fn record_movement_controller(
    input: Res<ButtonInput<KeyCode>>,
    mut controller_query: Query<(&mut SnakeMovementController, &mut LastUpdate)>,
) {
    // Collect directional input.
    let mut intent = None;
    let mut should_reset_timer = false;
    // FUTURE: Ignore reversing direction, since this always produces a crash
    if input.pressed(KeyCode::KeyW) || input.pressed(KeyCode::ArrowUp) {
        intent = Some(Dir::Up);
        if input.just_pressed(KeyCode::KeyW) || input.just_pressed(KeyCode::ArrowUp) { should_reset_timer = true; }
    }
    if input.pressed(KeyCode::KeyS) || input.pressed(KeyCode::ArrowDown) {
        intent = Some(Dir::Down);
        if input.just_pressed(KeyCode::KeyS) || input.just_pressed(KeyCode::ArrowDown) { should_reset_timer = true; }
    }
    if input.pressed(KeyCode::KeyA) || input.pressed(KeyCode::ArrowLeft) {
        intent = Some(Dir::Left);
        if input.just_pressed(KeyCode::KeyA) || input.just_pressed(KeyCode::ArrowLeft) { should_reset_timer = true; }
    }
    if input.pressed(KeyCode::KeyD) || input.pressed(KeyCode::ArrowRight) {
        intent = Some(Dir::Right);
        if input.just_pressed(KeyCode::KeyD) || input.just_pressed(KeyCode::ArrowRight) { should_reset_timer = true; }
    }

    // Apply movement intent to controllers.
    if intent.is_some() {
        for (mut controller, mut last_update) in &mut controller_query {
            controller.0 = intent;
            if should_reset_timer { *last_update = LastUpdate(0.0); }
        }
    }
}





#[derive(Component)]
struct MySnakeGame(snake_game::SnakeGame);


pub(super) fn plugin(app: &mut App) {
    // Register (i.e. record) what movement the player takes via keyboard/etc.
    app.register_type::<SnakeMovementController>();
    app.add_systems(Update, record_movement_controller.in_set(AppSet::RecordInput));


    // Apply movement based on controls.
    app.add_systems(Update, apply_movement.in_set(AppSet::Update));


    app.add_plugins(TilemapPlugin);

    app.observe(spawn_level);
}

fn tile_texture_index_of_cell_kind(kind: snake_game::CellKind) -> Option<u32> {
    match kind {
        snake_game::CellKind::Empty => None,
        snake_game::CellKind::Crash => Some(0),
        snake_game::CellKind::Apple => Some(1),
        snake_game::CellKind::Wall => Some(2),
        //snake_game::CellKind::SnakeHead => Some(3),
        //snake_game::CellKind::Snake => Some(4),
        snake_game::CellKind::SnakeHeadN => Some(5),
        snake_game::CellKind::SnakeHeadE => Some(6),
        snake_game::CellKind::SnakeHeadS => Some(7),
        snake_game::CellKind::SnakeHeadW => Some(8),
        snake_game::CellKind::SnakeBodyNS => Some(9),
        snake_game::CellKind::SnakeBodyEW => Some(10),
        snake_game::CellKind::SnakeTailN => Some(11),
        snake_game::CellKind::SnakeTailE => Some(12),
        snake_game::CellKind::SnakeTailS => Some(13),
        snake_game::CellKind::SnakeTailW => Some(14),
        snake_game::CellKind::SnakeBodyNE => Some(15),
        snake_game::CellKind::SnakeBodySE => Some(16),
        snake_game::CellKind::SnakeBodySW => Some(17),
        snake_game::CellKind::SnakeBodyNW => Some(18),
    }
}

fn copy_grid_into_tilemap(grid: &snake_game::Grid, tilemap_entity: Entity, tile_storage: &mut TileStorage, map_size: &TilemapSize, commands: &mut Commands) {
    for x in 0..map_size.x {
        for y in 0..map_size.y {
            let tile_pos = TilePos { x, y };
            let cell = grid.get_cell(snake_game::Point { x: x as i16, y: y as i16 });
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

fn spawn_level(
    _trigger: Trigger<SpawnLevel>,
    mut commands: Commands,
    asset_server: Res<AssetServer>,
) {
    let texture_handle: Handle<Image> = asset_server.load("images/tiles.png");
    let snake_game = snake_game::SnakeGame::new(None);
    // Add the snake game as a resource
    let tilemap_entity = commands.spawn_empty().id();
    let map_size = TilemapSize { x: snake_game.grid.width as u32, y: snake_game.grid.height as u32 };
    let mut tile_storage = TileStorage::empty(map_size);
    let map_type = TilemapType::Square;
    copy_grid_into_tilemap(&snake_game.grid, tilemap_entity, &mut tile_storage, &map_size, &mut commands);
    let tile_pixel_size = TilemapTileSize { x: 16.0, y: 16.0 };
    let grid_size = tile_pixel_size.into();
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
    commands.spawn((
        MySnakeGame(snake_game),
        LastUpdate(0.0),
        SnakeMovementController(None),
        StateScoped(Screen::Playing),
    ));
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
        if let Some(dir) = movement.0 {
            let current_time = time.elapsed_seconds_f64();
            if current_time - last_update.0 > 0.1 {
                my_snake_game.0.move_snake(dir.to_snake_direction(), None);
                let (tile_storage, tilemap_entity) = tilemap_query.get_single_mut().unwrap();
                update_tilemap(&mut commands, &my_snake_game, tilemap_entity, tile_storage, &mut tile_texture_query);
                last_update.0 = current_time;
            }
        }
    }
}


fn update_tilemap(
    commands: &mut Commands,
    my_snake_game: &Mut<MySnakeGame>,
    tilemap_entity: Entity,
    mut tile_storage: Mut<TileStorage>,
    tile_texture_query: &mut Query<&mut TileTextureIndex>,
) {
    let snake_game = &my_snake_game.0;
    //let map_size = TilemapSize { x: snake_game.grid.width as u32, y: snake_game.grid.height as u32 };
    //copy_grid_into_tilemap(&snake_game.grid, tilemap_entity, &mut tile_storage, &map_size, commands);
    
    for pt in &snake_game.grid_changes {
        let cell = snake_game.grid.get_cell(*pt);
        let tile_position = TilePos { x: pt.x as u32, y: pt.y as u32 };
        let tile = tile_storage.get(&tile_position);
        //info!("grid_change: location={pt:#?}, CellKind={:#?}, tile found={tile:?}", cell.kind);
        match (cell.kind, tile) {
            (snake_game::CellKind::Empty, None) => { /* Nothing to do. (Shouldn't happen.) */ info!("BUG!"); }
            (snake_game::CellKind::Empty, Some(tile)) => { 
                // Remove from Tilemap
                tile_storage.remove(&tile_position);
                commands.entity(tile).despawn();
            }
            (cell_kind, None) => { 
                // Create new tile entity
                let tile_texture_index = tile_texture_index_of_cell_kind(cell_kind).unwrap();
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
            (cell_kind, Some(tile)) => { 
                // Change texture of tile already in Tilemap
                if let Ok(mut tile_texture_index) = tile_texture_query.get_mut(tile) {
                    tile_texture_index.0 = tile_texture_index_of_cell_kind(cell_kind).unwrap();
                } else {
                    info!("No texture found for CellKind={:?} at position={:#?}", cell_kind, pt);
                }
            }
        }
    }
}
