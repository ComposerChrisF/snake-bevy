use std::path::PathBuf;

use bevy::prelude::Resource;
use clap::Parser;

#[derive(Resource, Parser, Debug)]
pub struct Args {
    #[arg(long)]
    pub sim: bool,

    #[arg(long)]
    pub playback: Option<PathBuf>,

    #[arg(long)]
    pub net: Option<PathBuf>,
}

