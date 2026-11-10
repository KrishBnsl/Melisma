//! `dhunn audio`: a small audio player for another program (the Elsewhere terminal player on
//! Linux and Windows, where it has no system audio framework of its own). It plays files and
//! live streams, reports where playback is, measures a spectrum of what's playing, and reads
//! song tags and embedded cover art. JSON lines: one request per line on stdin, one reply per
//! line on stdout.
//!
//! ```text
//! {"op":"start","path":"…","volume":0.7}   play a file           → {"ok":true} or an error
//! {"op":"live","url":"https://…","volume":0.7}  play a stream     → {"ok":true}
//! {"op":"live_status"}                 → {"status": 1 playing, 0 connecting, -1 failed}
//! {"op":"state"}                       → {"elapsed":s,"duration":s,"playing":bool}
//! {"op":"pause"} {"op":"resume"} {"op":"stop"} {"op":"seek","to":s} {"op":"volume","volume":0.7}
//! {"op":"spectrum","bands":24}         → {"bands":[0..1, …]} of what's playing
//! {"op":"metadata","path":"…"}         → {"title","artist","album","album_artist","artwork"}
//! {"op":"devices"}                     → available outputs
//! {"op":"output","id":"…"}              → switch output and stop the old sink
//! {"op":"prepare_next","path":"…","volume":0.7}  enqueue one gapless continuation
//! {"op":"adopt_next"} / {"op":"cancel_next"}      commit or cancel that continuation
//! {"op":"artwork","path":"…"}          → {"data": base64 of the embedded front cover}
//! ```
//!
//! The sound device is opened on the first song, so tag and cover requests work without one.
//! With `DHUNN_AUDIO_NULL=1` it plays to nowhere, in real time (for tests on machines without
//! sound).

use rodio::cpal::traits::{DeviceTrait, HostTrait};
use std::fs::File;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::atomic::{AtomicI8, AtomicU8, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use lofty::picture::PictureType;
use lofty::prelude::*;
use lofty::tag::ItemKey;
use realfft::RealFftPlanner;
use rodio::{ChannelCount, Sample, SampleRate, Source};
use serde_json::{Value, json};

pub fn run() -> Result<()> {
    let mut audio = Audio::default();
    let stdin = std::io::stdin();
    let mut out = std::io::stdout().lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Value>(&line) {
            Ok(req) => audio
                .handle(&req)
                .unwrap_or_else(|e| json!({"ok": false, "error": format!("{e:#}")})),
            Err(e) => json!({"ok": false, "error": format!("bad request: {e}")}),
        };
        writeln!(out, "{reply}")?;
        out.flush()?;
    }
    Ok(())
}

/// The latest mono samples heard, shared between the audio thread and requests.
#[derive(Default)]
struct Shared {
    ring: Mutex<Vec<f32>>,
    rate: AtomicU32,
    /// Samples heard since the current song started (a stream is "playing" once this moves).
    heard: AtomicU64,
}

const RING: usize = 4096;
const FFT: usize = 2048;
const NEXT_WAITING: u8 = 0;
const NEXT_STARTED: u8 = 1;
const NEXT_CANCELLED: u8 = 2;

/// Passes samples through unchanged and keeps a mono copy for the spectrum.
struct Tap<S> {
    inner: S,
    shared: Arc<Shared>,
    local: Vec<f32>,
    acc: f32,
    n: u16,
    transition: Option<Arc<AtomicU8>>,
    started: bool,
}

impl<S: Source> Tap<S> {
    fn new(inner: S, shared: Arc<Shared>) -> Self {
        Self {
            inner,
            shared,
            local: Vec::with_capacity(256),
            acc: 0.0,
            n: 0,
            transition: None,
            started: false,
        }
    }
}

impl<S: Source> Iterator for Tap<S> {
    type Item = Sample;

    fn next(&mut self) -> Option<Sample> {
        if self
            .transition
            .as_ref()
            .is_some_and(|t| t.load(Ordering::Acquire) == NEXT_CANCELLED)
        {
            return None;
        }
        let s = self.inner.next()?;
        if !self.started {
            if let Some(t) = &self.transition
                && t.compare_exchange(
                    NEXT_WAITING,
                    NEXT_STARTED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_err()
            {
                return None;
            }
            self.started = true;
            self.shared
                .rate
                .store(self.inner.sample_rate().get(), Ordering::Relaxed);
            self.shared.heard.store(0, Ordering::Relaxed);
            if let Ok(mut r) = self.shared.ring.lock() {
                r.clear();
            }
        }
        self.acc += s;
        self.n += 1;
        if self.n >= self.inner.channels().get() {
            self.local.push(self.acc / f32::from(self.n));
            self.acc = 0.0;
            self.n = 0;
            if self.local.len() >= 256 {
                if let Ok(mut r) = self.shared.ring.lock() {
                    r.extend_from_slice(&self.local);
                    let len = r.len();
                    if len > RING {
                        r.drain(..len - RING);
                    }
                }
                self.shared
                    .heard
                    .fetch_add(self.local.len() as u64, Ordering::Relaxed);
                self.local.clear();
            }
        }
        Some(s)
    }
}

impl<S: Source> Source for Tap<S> {
    fn current_span_len(&self) -> Option<usize> {
        self.inner.current_span_len()
    }
    fn channels(&self) -> ChannelCount {
        self.inner.channels()
    }
    fn sample_rate(&self) -> SampleRate {
        self.inner.sample_rate()
    }
    fn total_duration(&self) -> Option<Duration> {
        self.inner.total_duration()
    }
    fn try_seek(&mut self, pos: Duration) -> Result<(), rodio::source::SeekError> {
        self.inner.try_seek(pos)?;
        if let Ok(mut r) = self.shared.ring.lock() {
            r.clear();
        }
        Ok(())
    }
}

/// A network stream for the decoder: it reads forward and never seeks.
struct Stream {
    inner: Mutex<Box<dyn Read + Send>>,
    pos: u64,
}

impl Read for Stream {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self
            .inner
            .get_mut()
            .map_err(|_| std::io::Error::other("stream lock"))?
            .read(buf)?;
        self.pos += n as u64;
        Ok(n)
    }
}

impl Seek for Stream {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        match to {
            SeekFrom::Current(0) => Ok(self.pos),
            SeekFrom::Start(p) if p == self.pos => Ok(self.pos),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                "a live stream can't seek",
            )),
        }
    }
}

/// Live stream status: 1 playing, 0 connecting, -1 failed.
const CONNECTING: i8 = 0;
const PLAYING: i8 = 1;
const FAILED: i8 = -1;

struct Prepared {
    duration: Option<Duration>,
    state: Arc<AtomicU8>,
}
impl Prepared {
    fn started(&self) -> bool {
        self.state.load(Ordering::Acquire) == NEXT_STARTED
    }
}

fn local_decoder(path: &str) -> Result<(rodio::Decoder<BufReader<File>>, Option<Duration>)> {
    let file = File::open(path).with_context(|| format!("can't open {path}"))?;
    let len = file.metadata().map(|m| m.len()).ok();
    let mut builder = rodio::Decoder::builder()
        .with_data(BufReader::new(file))
        .with_seekable(true)
        .with_gapless(true);
    if let Some(len) = len {
        builder = builder.with_byte_len(len);
    }
    if let Some(ext) = Path::new(path).extension().and_then(|e| e.to_str()) {
        builder = builder.with_hint(ext);
    }
    let decoder = builder.build().context("can't decode this file")?;
    let duration = decoder.total_duration().or_else(|| {
        lofty::read_from_path(path)
            .ok()
            .map(|f| f.properties().duration())
    });
    Ok((decoder, duration))
}

#[derive(Default)]
struct Audio {
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    media: Option<crate::audio_media::DesktopMedia>,
    device: Option<rodio::MixerDeviceSink>,
    output_id: Option<String>,
    /// The silent output, when there's no device on purpose.
    null: Option<rodio::mixer::Mixer>,
    player: Option<Arc<rodio::Player>>,
    shared: Arc<Shared>,
    duration: Option<Duration>,
    prepared: Option<Prepared>,
    /// For a live stream: its status, and a counter that retires old connections.
    live: Option<Arc<AtomicI8>>,
    generation: Arc<AtomicU64>,
}

impl Audio {
    fn cancel_next(&mut self) {
        if let Some(next) = self.prepared.take() {
            next.state.store(NEXT_CANCELLED, Ordering::Release);
        }
    }

    /// A fresh player on the sound device (opened the first time).
    fn fresh_player(&mut self, volume: f32) -> Result<Arc<rodio::Player>> {
        self.cancel_next();
        if let Some(p) = self.player.take() {
            p.stop();
        }
        let silent = std::env::var_os("DHUNN_AUDIO_NULL").is_some_and(|v| v == "1");
        if silent && self.null.is_none() {
            self.null = Some(null_output());
        }
        if !silent && self.device.is_none() {
            let mut d =
                rodio::DeviceSinkBuilder::open_default_sink().context("no sound output found")?;
            d.log_on_drop(false);
            self.device = Some(d);
        }
        let mixer = match (&self.null, &self.device) {
            (Some(m), _) => m,
            (None, Some(d)) => d.mixer(),
            (None, None) => unreachable!("one was opened above"),
        };
        let p = Arc::new(rodio::Player::connect_new(mixer));
        p.set_volume(volume.clamp(0.0, 2.0));
        self.player = Some(p.clone());
        Ok(p)
    }

    fn handle(&mut self, req: &Value) -> Result<Value> {
        let ok = json!({"ok": true});
        let volume = req["volume"].as_f64().unwrap_or(0.7) as f32;
        Ok(match req["op"].as_str().unwrap_or("") {
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            "media_start" => {
                self.media = crate::audio_media::DesktopMedia::open();
                json!({"ok": self.media.is_some(), "error": "desktop media controls unavailable"})
            }
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            "media_stop" => {
                self.media = None;
                ok
            }
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            "media_poll" => self.media.as_ref().map_or(ok, |media| media.poll()),
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            "media_update" => {
                if let Some(media) = &mut self.media {
                    media.update(req);
                }
                ok
            }
            "devices" => {
                let host = rodio::cpal::default_host();
                let current = self.output_id.clone().or_else(|| {
                    host.default_output_device()?
                        .id()
                        .ok()
                        .map(|id| id.to_string())
                });
                let devices: Vec<_> = host
                    .output_devices()?
                    .filter_map(|device| {
                        let id = device.id().ok()?.to_string();
                        let name = device.description().ok()?.name().to_owned();
                        Some(json!({"current": current.as_ref() == Some(&id), "id": id, "name": name}))
                    })
                    .collect();
                json!({"ok": true, "devices": devices})
            }
            "output" => {
                let id = req["id"].as_str().context("output needs a device ID")?;
                let host = rodio::cpal::default_host();
                let device = host
                    .output_devices()?
                    .find(|device| device.id().is_ok_and(|found| found.to_string() == id))
                    .context("output device is no longer available")?;
                let mut sink = rodio::DeviceSinkBuilder::from_device(device)?.open_stream()?;
                sink.log_on_drop(false);
                self.cancel_next();
                if let Some(player) = self.player.take() {
                    player.stop();
                }
                self.generation.fetch_add(1, Ordering::SeqCst);
                self.live = None;
                self.device = Some(sink);
                self.output_id = Some(id.to_owned());
                ok
            }
            "start" => {
                let path = req["path"].as_str().context("start needs a path")?;
                self.generation.fetch_add(1, Ordering::SeqCst);
                self.live = None;
                let (decoder, duration) = local_decoder(path)?;
                self.duration = duration;
                let player = self.fresh_player(volume)?;
                player.append(Tap::new(decoder, self.shared.clone()));
                player.play();
                ok
            }
            "prepare_next" => {
                anyhow::ensure!(
                    self.prepared.is_none() && self.live.is_none(),
                    "transition unavailable"
                );
                let player = self.player.as_ref().context("nothing playing")?;
                // Both sources share the sink volume until the next source is adopted.
                anyhow::ensure!(
                    (volume - player.volume()).abs() < 0.001,
                    "track volume changes"
                );
                let path = req["path"].as_str().context("prepare_next needs a path")?;
                let (decoder, duration) = local_decoder(path)?;
                let state = Arc::new(AtomicU8::new(NEXT_WAITING));
                let mut tap = Tap::new(decoder, self.shared.clone());
                tap.transition = Some(state.clone());
                player.append(tap);
                self.prepared = Some(Prepared { duration, state });
                ok
            }
            "adopt_next" => {
                let ready = self.prepared.as_ref().is_some_and(Prepared::started);
                if ready && let Some(next) = self.prepared.take() {
                    self.duration = next.duration;
                }
                json!({"ok": true, "adopted": ready})
            }
            "cancel_next" => {
                self.cancel_next();
                ok
            }
            "live" => {
                let url = req["url"].as_str().context("live needs a url")?.to_owned();
                anyhow::ensure!(url.starts_with("https://"), "only https streams");
                let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
                let player = self.fresh_player(volume)?;
                self.duration = None;
                // Nothing heard from this stream yet (the last song's count would look like
                // a stream that already ended).
                self.shared.heard.store(0, Ordering::SeqCst);
                let status = Arc::new(AtomicI8::new(CONNECTING));
                self.live = Some(status.clone());
                let (shared, current) = (self.shared.clone(), self.generation.clone());
                std::thread::spawn(move || {
                    let result = connect(&url).map(|d| Tap::new(d, shared));
                    if current.load(Ordering::SeqCst) != generation {
                        return; // Something else started meanwhile.
                    }
                    match result {
                        Ok(source) => {
                            player.append(source);
                            player.play();
                        }
                        Err(e) => {
                            eprintln!("audio: {url}: {e:#}");
                            status.store(FAILED, Ordering::SeqCst);
                        }
                    }
                });
                ok
            }
            "live_status" => {
                let status = match (&self.live, &self.player) {
                    (Some(s), Some(p)) => {
                        let s = s.load(Ordering::SeqCst);
                        if s == FAILED {
                            FAILED
                        } else if p.is_paused() {
                            CONNECTING
                        } else if self.shared.heard.load(Ordering::Relaxed) > 0 {
                            // Playing until the stream runs out.
                            if p.empty() { FAILED } else { PLAYING }
                        } else {
                            CONNECTING
                        }
                    }
                    _ => FAILED,
                };
                json!({"ok": true, "status": status})
            }
            "state" => {
                if self.prepared.as_ref().is_some_and(Prepared::started) {
                    let duration = self.duration.map_or(0.0, |d| d.as_secs_f64());
                    return Ok(
                        json!({"ok": true, "elapsed": duration, "duration": duration, "playing": false}),
                    );
                }
                let (elapsed, playing) = match &self.player {
                    Some(p) => (p.get_pos().as_secs_f64(), !p.is_paused() && !p.empty()),
                    None => (0.0, false),
                };
                let (elapsed, duration) = if self.live.is_some() {
                    (0.0, 0.0)
                } else {
                    (elapsed, self.duration.map_or(0.0, |d| d.as_secs_f64()))
                };
                json!({"ok": true, "elapsed": elapsed, "duration": duration, "playing": playing})
            }
            "pause" => {
                if let Some(p) = &self.player {
                    p.pause();
                }
                ok
            }
            "resume" => {
                let resumed = match &self.player {
                    Some(p) if !p.empty() => {
                        p.play();
                        true
                    }
                    _ => false,
                };
                json!({"ok": resumed, "error": if resumed { "" } else { "nothing to resume" }})
            }
            "stop" => {
                self.cancel_next();
                self.generation.fetch_add(1, Ordering::SeqCst);
                if let Some(p) = self.player.take() {
                    p.stop();
                }
                self.live = None;
                ok
            }
            "seek" => {
                let to = req["to"].as_f64().unwrap_or(0.0).max(0.0);
                if let Some(p) = &self.player
                    && self.live.is_none()
                {
                    p.try_seek(Duration::from_secs_f64(to))
                        .map_err(|e| anyhow::anyhow!("can't seek: {e}"))?;
                }
                ok
            }
            "volume" => {
                if let Some(p) = &self.player {
                    p.set_volume(volume.clamp(0.0, 2.0));
                }
                ok
            }
            "spectrum" => {
                let n = req["bands"].as_u64().unwrap_or(24).clamp(1, 128) as usize;
                let playing = self
                    .player
                    .as_ref()
                    .is_some_and(|p| !p.is_paused() && !p.empty());
                match spectrum(&self.shared, n).filter(|_| playing) {
                    Some(bands) => json!({"ok": true, "bands": bands}),
                    None => json!({"ok": true, "bands": vec![0.0; n]}),
                }
            }
            "metadata" => {
                let path = req["path"].as_str().context("metadata needs a path")?;
                metadata(path)
            }
            "artwork" => {
                let path = req["path"].as_str().context("artwork needs a path")?;
                match artwork(path) {
                    Some(data) => json!({"ok": true, "data": base64(&data)}),
                    None => json!({"ok": false, "error": "no embedded cover"}),
                }
            }
            other => anyhow::bail!("unknown op {other:?}"),
        })
    }
}

/// A mixer that plays to nowhere: a thread takes its samples at the speed a device would.
fn null_output() -> rodio::mixer::Mixer {
    const RATE: u32 = 44_100;
    let (mixer, mut source) = rodio::mixer::mixer(
        ChannelCount::new(2).expect("nonzero"),
        SampleRate::new(RATE).expect("nonzero"),
    );
    std::thread::spawn(move || {
        let tick = Duration::from_millis(20);
        let per_tick = (RATE as usize * 2) / 50;
        let mut next = std::time::Instant::now();
        loop {
            for _ in 0..per_tick {
                source.next();
            }
            next += tick;
            std::thread::sleep(next.saturating_duration_since(std::time::Instant::now()));
        }
    });
    mixer
}

/// Opens a live stream and its decoder.
fn connect(url: &str) -> Result<rodio::Decoder<Stream>> {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_connect(Some(Duration::from_secs(8)))
        .timeout_recv_response(Some(Duration::from_secs(10)))
        .user_agent("elsewhere (https://github.com/KrishBnsl/elsewhere)")
        .build()
        .into();
    let resp = agent.get(url).call()?;
    let kind = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_lowercase();
    let hint = if kind.contains("ogg") || kind.contains("opus") {
        "ogg"
    } else if kind.contains("aac") || kind.contains("mp4") {
        "aac"
    } else if kind.contains("flac") {
        "flac"
    } else {
        "mp3"
    };
    let reader: Box<dyn Read + Send> = Box::new(resp.into_body().into_reader());
    let decoder = rodio::Decoder::builder()
        .with_data(Stream {
            inner: Mutex::new(reader),
            pos: 0,
        })
        .with_seekable(false)
        .with_hint(hint)
        .build()
        .context("can't decode this stream")?;
    Ok(decoder)
}

/// Log-spaced bands (0..1) of the latest samples: 40 Hz to 16 kHz.
fn spectrum(shared: &Shared, n: usize) -> Option<Vec<f32>> {
    let mut input: Vec<f32> = {
        let r = shared.ring.lock().ok()?;
        if r.len() < FFT {
            return None;
        }
        r[r.len() - FFT..].to_vec()
    };
    for (i, x) in input.iter_mut().enumerate() {
        *x *= 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / FFT as f32).cos();
    }
    let fft = RealFftPlanner::<f32>::new().plan_fft_forward(FFT);
    let mut out = fft.make_output_vec();
    fft.process(&mut input, &mut out).ok()?;
    let rate = shared.rate.load(Ordering::Relaxed).max(8000) as f32;
    let bin_hz = rate / FFT as f32;
    let (lo, hi) = (40f32, 16_000f32.min(rate / 2.0));
    Some(
        (0..n)
            .map(|b| {
                let f0 = lo * (hi / lo).powf(b as f32 / n as f32);
                let f1 = lo * (hi / lo).powf((b + 1) as f32 / n as f32);
                let i0 = ((f0 / bin_hz) as usize).max(1);
                let i1 = ((f1 / bin_hz) as usize).max(i0 + 1).min(out.len());
                let p = out[i0..i1]
                    .iter()
                    .map(|c| c.norm_sqr())
                    .fold(0.0f32, f32::max);
                // dB relative to a full-scale sine, with a treble lift (music has less up there).
                let mag = p.sqrt() / (FFT as f32 / 4.0);
                let db = 20.0 * mag.max(1e-9).log10() + 12.0 * (b as f32 / n as f32);
                ((db + 62.0) / 52.0).clamp(0.0, 1.0)
            })
            .collect(),
    )
}

fn metadata(path: &str) -> Value {
    let Ok(file) = lofty::read_from_path(path) else {
        return json!({"ok": true, "title": "", "artist": "", "album": "", "album_artist": "",
                      "artwork": false});
    };
    let tag = file.primary_tag().or(file.first_tag());
    let get = |key: ItemKey| {
        tag.and_then(|t| t.get_string(key))
            .unwrap_or("")
            .trim()
            .to_owned()
    };
    let artwork = file.tags().iter().any(|t| !t.pictures().is_empty());
    json!({
        "ok": true,
        "title": get(ItemKey::TrackTitle),
        "artist": get(ItemKey::TrackArtist),
        "album": get(ItemKey::AlbumTitle),
        "album_artist": get(ItemKey::AlbumArtist),
        "artwork": artwork,
        "disc": get(ItemKey::DiscNumber),
        "number": get(ItemKey::TrackNumber),
        "track_gain": get(ItemKey::ReplayGainTrackGain),
        "album_gain": get(ItemKey::ReplayGainAlbumGain),
    })
}

/// The embedded front cover (or the first picture), as stored.
fn artwork(path: &str) -> Option<Vec<u8>> {
    let file = lofty::read_from_path(path).ok()?;
    let pics: Vec<_> = file.tags().iter().flat_map(|t| t.pictures()).collect();
    let pic = pics
        .iter()
        .find(|p| p.pic_type() == PictureType::CoverFront)
        .or(pics.first())?;
    Some(pic.data().to_vec())
}

fn base64(data: &[u8]) -> String {
    const ABC: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(ABC[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rodio::source::SineWave;

    #[test]
    fn base64_matches_the_standard() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn a_tone_lights_its_band() {
        let shared = Arc::new(Shared::default());
        let tap = Tap::new(SineWave::new(440.0), shared.clone());
        assert_eq!(tap.take(RING * 2).count(), RING * 2);
        let bands = spectrum(&shared, 24).unwrap();
        let peak = (0..24)
            .max_by(|&a, &b| bands[a].total_cmp(&bands[b]))
            .unwrap();
        // 440 Hz on a 40 Hz–16 kHz log scale of 24 bands is band ~9.
        assert!((8..=10).contains(&peak), "peak band {peak}: {bands:?}");
        assert!(shared.heard.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn requests_without_a_sound_device() {
        let mut a = Audio::default();
        let state = a.handle(&json!({"op": "state"})).unwrap();
        assert_eq!(state["playing"], false);
        let status = a.handle(&json!({"op": "live_status"})).unwrap();
        assert_eq!(status["status"], FAILED);
        let meta = a
            .handle(&json!({"op": "metadata", "path": "/no/such/file.mp3"}))
            .unwrap();
        assert_eq!(meta["title"], "");
        assert!(
            a.handle(&json!({"op": "start", "path": "/no/such/file.mp3"}))
                .is_err()
        );
        assert!(a.handle(&json!({"op": "nope"})).is_err());
    }

    #[test]
    fn plays_to_nowhere_in_real_time() {
        // SAFETY: tests in this module don't read the variable concurrently with other writes.
        unsafe { std::env::set_var("DHUNN_AUDIO_NULL", "1") };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tone.wav");
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 8000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..4000 {
            let s = (i as f32 * 440.0 * std::f32::consts::TAU / 8000.0).sin();
            w.write_sample((s * 12000.0) as i16).unwrap();
        }
        w.finalize().unwrap();
        let mut a = Audio::default();
        let p = path.to_str().unwrap();
        assert_eq!(
            a.handle(&json!({"op": "start", "path": p, "volume": 0.5}))
                .unwrap()["ok"],
            true
        );
        std::thread::sleep(Duration::from_millis(250));
        let s = a.handle(&json!({"op": "state"})).unwrap();
        assert_eq!(s["playing"], true);
        assert!((s["duration"].as_f64().unwrap() - 0.5).abs() < 0.05, "{s}");
        assert!(s["elapsed"].as_f64().unwrap() > 0.1, "{s}");
        a.handle(&json!({"op": "prepare_next", "path": p, "volume": 0.5}))
            .unwrap();
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        while !a.handle(&json!({"op": "adopt_next"})).unwrap()["adopted"]
            .as_bool()
            .unwrap()
        {
            assert!(
                std::time::Instant::now() < deadline,
                "next source never started"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(a.handle(&json!({"op": "state"})).unwrap()["playing"], true);
        a.handle(&json!({"op": "prepare_next", "path": p, "volume": 0.5}))
            .unwrap();
        a.handle(&json!({"op": "cancel_next"})).unwrap();
        std::thread::sleep(Duration::from_millis(600));
        assert_eq!(a.handle(&json!({"op": "state"})).unwrap()["playing"], false);
    }

    #[test]
    fn streams_never_seek_backward() {
        let mut s = Stream {
            inner: Mutex::new(Box::new(std::io::Cursor::new(vec![1u8, 2, 3]))),
            pos: 0,
        };
        let mut buf = [0u8; 2];
        assert_eq!(s.read(&mut buf).unwrap(), 2);
        assert_eq!(s.stream_position().unwrap(), 2);
        assert!(s.seek(SeekFrom::Start(0)).is_err());
    }
}
