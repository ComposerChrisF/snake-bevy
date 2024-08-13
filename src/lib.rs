#[cfg(feature = "dev")]
mod dev_tools;
mod game;
mod screen;
mod ui;
mod snake_game;
mod neural_net;
pub mod nn_plays_snake;

use bevy::{
    asset::AssetMetaCheck, audio::{AudioPlugin, Volume}, prelude::*, render::camera::ScalingMode, window::WindowResolution
};

// TODO: Base these off of the snake game size?
pub const WINDOW_SIZE_X: f32 = 40.0 * 16.0 + 40.0;
pub const WINDOW_SIZE_Y: f32 = 30.0 * 16.0 + 40.0;

pub struct AppPlugin;

impl Plugin for AppPlugin {
    fn build(&self, app: &mut App) {
        // Order new `AppStep` variants by adding them here:
        app.configure_sets(
            Update,
            (AppSet::TickTimers, AppSet::RecordInput, AppSet::Update).chain(),
        );

        // Spawn the main camera.
        app.add_systems(Startup, spawn_camera);

        // Add Bevy plugins.
        app.add_plugins(
            DefaultPlugins
                .set(AssetPlugin {
                    // Wasm builds will check for meta files (that don't exist) if this isn't set.
                    // This causes errors and even panics on web build on itch.
                    // See https://github.com/bevyengine/bevy_github_ci_template/issues/48.
                    meta_check: AssetMetaCheck::Never,
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Window {
                        title: "Snake Bevy".to_string(),
                        canvas: Some("#bevy".to_string()),
                        fit_canvas_to_parent: true,
                        prevent_default_event_handling: true,
                        resolution: WindowResolution::new(WINDOW_SIZE_X, WINDOW_SIZE_Y).with_scale_factor_override(1.0),
                        ..default()
                    }
                    .into(),
                    ..default()
                })
                .set(AudioPlugin {
                    global_volume: GlobalVolume {
                        volume: Volume::new(0.3),
                    },
                    ..default()
                }),
        );

        // Add other plugins.
        app.add_plugins((game::plugin, screen::plugin, ui::plugin));

        // Enable dev tools for dev builds.
        #[cfg(feature = "dev")]
        app.add_plugins(dev_tools::plugin);
    }
}

/// High-level groupings of systems for the app in the `Update` schedule.
/// When adding a new variant, make sure to order it in the `configure_sets`
/// call above.
#[derive(SystemSet, Debug, Clone, Copy, Eq, PartialEq, Hash)]
enum AppSet {
    /// Tick timers.
    TickTimers,
    /// Record player input.
    RecordInput,
    /// Do everything else (consider splitting this into further variants).
    Update,
}

fn spawn_camera(mut commands: Commands) {
    let mut camera = Camera2dBundle::default();

    // Automatically change camera based on size of containing window:
    camera.projection.scaling_mode = ScalingMode::FixedVertical(WINDOW_SIZE_Y);
    camera.projection.area = Rect::new(0.0, 0.0, WINDOW_SIZE_X, WINDOW_SIZE_Y);
    
    commands.spawn((
        Name::new("Camera"),
        camera,
        // Render all UI to this camera.
        // Not strictly necessary since we only use one camera,
        // but if we don't use this component, our UI will disappear as soon
        // as we add another camera. This includes indirect ways of adding cameras like using
        // [ui node outlines](https://bevyengine.org/news/bevy-0-14/#ui-node-outline-gizmos)
        // for debugging. So it's good to have this here for future-proofing.
        IsDefaultUiCamera,
    ));
}
