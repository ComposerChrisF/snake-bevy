use bevy::{audio::PlaybackMode, prelude::*};
use rand::{thread_rng, Rng};

use crate::game::assets::{HandleMap, SfxKey};

pub(super) fn plugin(app: &mut App) {
    app.observe(play_sfx);
}

fn play_sfx(
    trigger: Trigger<PlaySfx>,
    mut commands: Commands,
    sfx_handles: Res<HandleMap<SfxKey>>,
) {
    let sfx_key = match trigger.event() {
        PlaySfx::Key(SfxKey::Crash(0))   => SfxKey::Crash(  my_random( 4)),
        PlaySfx::Key(SfxKey::Eating(0))  => SfxKey::Eating( my_random( 4)),
        PlaySfx::Key(SfxKey::Growing(0)) => SfxKey::Growing(my_random(11)),
        PlaySfx::Key(SfxKey::Tick(0))    => SfxKey::Tick(   my_random(12)),
        PlaySfx::Key(key) => *key,
    };
    commands.spawn(AudioSourceBundle {
        source: sfx_handles[&sfx_key].clone_weak(),
        settings: PlaybackSettings {
            mode: PlaybackMode::Despawn,
            ..default()
        },
    });
}

fn my_random(i: usize) -> usize {
    thread_rng().gen_range(1..i)
}

/// Trigger this event to play a single sound effect.
#[derive(Event)]
pub enum PlaySfx {
    Key(SfxKey),
}
