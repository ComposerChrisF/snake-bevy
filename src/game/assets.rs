use bevy::{
    prelude::*,
    render::texture::{ImageLoaderSettings, ImageSampler},
    utils::HashMap,
};

pub(super) fn plugin(app: &mut App) {
    app.register_type::<HandleMap<ImageKey>>();
    app.init_resource::<HandleMap<ImageKey>>();

    app.register_type::<HandleMap<SfxKey>>();
    app.init_resource::<HandleMap<SfxKey>>();

    app.register_type::<HandleMap<SoundtrackKey>>();
    app.init_resource::<HandleMap<SoundtrackKey>>();
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Reflect)]
pub enum ImageKey {
    SnakeTiles,
}

impl AssetKey for ImageKey {
    type Asset = Image;
}

impl FromWorld for HandleMap<ImageKey> {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        [(
            ImageKey::SnakeTiles,
            asset_server.load_with_settings(
                "images/snake_tiles.png",
                |settings: &mut ImageLoaderSettings| {
                    settings.sampler = ImageSampler::nearest();
                },
            ),
        )]
        .into()
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Reflect)]
pub enum SfxKey {
    ButtonHover,
    ButtonPress,
    Crash(usize),
    Eating(usize),
    Growing(usize),
    Tick(usize),
}

impl AssetKey for SfxKey {
    type Asset = AudioSource;
}

impl FromWorld for HandleMap<SfxKey> {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        [
            ( SfxKey::ButtonHover, asset_server.load("audio/sfx/button_hover.ogg") ),
            ( SfxKey::ButtonPress, asset_server.load("audio/sfx/button_press.ogg") ),
            ( SfxKey::Crash(1), asset_server.load("audio/sfx/crash_1.ogg") ),
            ( SfxKey::Crash(2), asset_server.load("audio/sfx/crash_2.ogg") ),
            ( SfxKey::Crash(3), asset_server.load("audio/sfx/crash_3.ogg") ),
            ( SfxKey::Crash(4), asset_server.load("audio/sfx/crash_4.ogg") ),
            ( SfxKey::Eating(1), asset_server.load("audio/sfx/eating_1.ogg") ),
            ( SfxKey::Eating(2), asset_server.load("audio/sfx/eating_2.ogg") ),
            ( SfxKey::Eating(3), asset_server.load("audio/sfx/eating_3.ogg") ),
            ( SfxKey::Eating(4), asset_server.load("audio/sfx/eating_4.ogg") ),
            ( SfxKey::Growing( 1), asset_server.load("audio/sfx/growing_1.ogg") ),
            ( SfxKey::Growing( 2), asset_server.load("audio/sfx/growing_2.ogg") ),
            ( SfxKey::Growing( 3), asset_server.load("audio/sfx/growing_3.ogg") ),
            ( SfxKey::Growing( 4), asset_server.load("audio/sfx/growing_4.ogg") ),
            ( SfxKey::Growing( 5), asset_server.load("audio/sfx/growing_5.ogg") ),
            ( SfxKey::Growing( 6), asset_server.load("audio/sfx/growing_6.ogg") ),
            ( SfxKey::Growing( 7), asset_server.load("audio/sfx/growing_7.ogg") ),
            ( SfxKey::Growing( 8), asset_server.load("audio/sfx/growing_8.ogg") ),
            ( SfxKey::Growing( 9), asset_server.load("audio/sfx/growing_9.ogg") ),
            ( SfxKey::Growing(10), asset_server.load("audio/sfx/growing_a.ogg") ),
            ( SfxKey::Growing(11), asset_server.load("audio/sfx/growing_b.ogg") ),
            ( SfxKey::Tick( 1), asset_server.load("audio/sfx/timer_tick_1.ogg") ),
            ( SfxKey::Tick( 2), asset_server.load("audio/sfx/timer_tick_2.ogg") ),
            ( SfxKey::Tick( 3), asset_server.load("audio/sfx/timer_tick_3.ogg") ),
            ( SfxKey::Tick( 4), asset_server.load("audio/sfx/timer_tick_4.ogg") ),
            ( SfxKey::Tick( 5), asset_server.load("audio/sfx/timer_tick_5.ogg") ),
            ( SfxKey::Tick( 6), asset_server.load("audio/sfx/timer_tick_6.ogg") ),
            ( SfxKey::Tick( 7), asset_server.load("audio/sfx/timer_tick_7.ogg") ),
            ( SfxKey::Tick( 8), asset_server.load("audio/sfx/timer_tick_8.ogg") ),
            ( SfxKey::Tick( 9), asset_server.load("audio/sfx/timer_tick_9.ogg") ),
            ( SfxKey::Tick(10), asset_server.load("audio/sfx/timer_tick_a.ogg") ),
            ( SfxKey::Tick(11), asset_server.load("audio/sfx/timer_tick_b.ogg") ),
            ( SfxKey::Tick(12), asset_server.load("audio/sfx/timer_tick_c.ogg") ),
        ]
        .into()
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Reflect)]
pub enum SoundtrackKey {
    Credits,
    Gameplay,
}

impl AssetKey for SoundtrackKey {
    type Asset = AudioSource;
}

impl FromWorld for HandleMap<SoundtrackKey> {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        [
            (
                SoundtrackKey::Credits,
                asset_server.load("audio/soundtracks/Monkeys Spinning Monkeys.ogg"),
            ),
            (
                SoundtrackKey::Gameplay,
                asset_server.load("audio/soundtracks/Fluffing A Duck.ogg"),
            ),
        ]
        .into()
    }
}

pub trait AssetKey: Sized {
    type Asset: Asset;
}

#[derive(Resource, Reflect, Deref, DerefMut)]
#[reflect(Resource)]
pub struct HandleMap<K: AssetKey>(HashMap<K, Handle<K::Asset>>);

impl<K: AssetKey, T> From<T> for HandleMap<K>
where
    T: Into<HashMap<K, Handle<K::Asset>>>,
{
    fn from(value: T) -> Self {
        Self(value.into())
    }
}

impl<K: AssetKey> HandleMap<K> {
    pub fn all_loaded(&self, asset_server: &AssetServer) -> bool {
        self.values()
            .all(|x| asset_server.is_loaded_with_dependencies(x))
    }
}
