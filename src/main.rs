// Disable console on Windows for non-dev builds.
#![cfg_attr(not(feature = "dev"), windows_subsystem = "windows")]

use std::fs;

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
        if let Some(path) = &args.playback {
            if !fs::exists(path).unwrap() {
                panic!("--playback <file>: file not found: {path:?}")
            }
        }
        if let Some(path) = &args.net {
            if !fs::exists(path).unwrap() {
                panic!("--net <file>: file not found: {path:?}")
            }
        }
        App::new()
            .insert_resource(args)
            .add_plugins(AppPlugin)
            .run()
    }
}
