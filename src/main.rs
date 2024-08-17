// Disable console on Windows for non-dev builds.
#![cfg_attr(not(feature = "dev"), windows_subsystem = "windows")]

use bevy::prelude::*;
use clap::Parser;
use snake_bevy::nn_plays_snake::NnPlaysSnake;
use snake_bevy::{cmdline::Args, AppPlugin};


fn main() -> AppExit {
    let args = Args::parse();
    if args.sim {
        let mut nn_player = NnPlaysSnake::new();
        nn_player.run_x_generations();
        AppExit::Success
    } else {
        App::new().add_plugins(AppPlugin).run()
    }
}
