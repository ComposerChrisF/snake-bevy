// Disable console on Windows for non-dev builds.
#![cfg_attr(not(feature = "dev"), windows_subsystem = "windows")]

use bevy::prelude::*;
use snake_bevy::nn_plays_snake::NnPlaysSnake;
//use snake_bevy::AppPlugin;


fn main() -> AppExit {
    let mut nn_player = NnPlaysSnake::new();
    nn_player.run_x_generations();
    AppExit::Success
    //App::new().add_plugins(AppPlugin).run()
}
