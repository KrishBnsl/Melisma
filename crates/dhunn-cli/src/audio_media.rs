//! Desktop media controls for Elsewhere's audio subprocess.
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

use serde_json::{Value, json};
use souvlaki::{
    MediaControlEvent, MediaControls, MediaMetadata, MediaPlayback, MediaPosition, PlatformConfig,
};

pub(crate) struct DesktopMedia {
    controls: MediaControls,
    actions: Receiver<u8>,
    path: String,
    #[cfg(target_os = "windows")]
    window: MediaWindow,
}

#[cfg(target_os = "windows")]
struct MediaWindow(windows_sys::Win32::Foundation::HWND);

#[cfg(target_os = "windows")]
impl MediaWindow {
    fn open() -> Option<Self> {
        use std::ptr::null_mut;
        use windows_sys::Win32::UI::WindowsAndMessaging::CreateWindowExW;
        let class = windows_sys::w!("STATIC");
        let title = windows_sys::w!("Elsewhere");
        // SMTC needs an HWND owned by this process. The window has no visible style.
        let handle = unsafe {
            CreateWindowExW(
                0,
                class,
                title,
                0,
                0,
                0,
                0,
                0,
                null_mut(),
                null_mut(),
                null_mut(),
                std::ptr::null(),
            )
        };
        (!handle.is_null()).then_some(Self(handle))
    }

    fn dispatch(&self) {
        use windows_sys::Win32::UI::WindowsAndMessaging::{
            DispatchMessageW, MSG, PM_REMOVE, PeekMessageW, TranslateMessage,
        };
        let mut message = MSG::default();
        // This is called on the same thread that owns the hidden window.
        unsafe {
            while PeekMessageW(&mut message, self.0, 0, 0, PM_REMOVE) != 0 {
                TranslateMessage(&message);
                DispatchMessageW(&message);
            }
        }
    }
}

#[cfg(target_os = "windows")]
impl Drop for MediaWindow {
    fn drop(&mut self) {
        unsafe { windows_sys::Win32::UI::WindowsAndMessaging::DestroyWindow(self.0) };
    }
}

impl DesktopMedia {
    pub(crate) fn open() -> Option<Self> {
        #[cfg(target_os = "windows")]
        let window = MediaWindow::open()?;
        #[cfg(target_os = "windows")]
        let hwnd = Some(window.0);
        #[cfg(not(target_os = "windows"))]
        let hwnd = None;
        let mut controls = MediaControls::new(PlatformConfig {
            dbus_name: "Elsewhere",
            display_name: "Elsewhere",
            hwnd,
        })
        .ok()?;
        let (send, actions) = mpsc::sync_channel(16);
        controls
            .attach(move |event| {
                let action = match event {
                    MediaControlEvent::Toggle => 1,
                    MediaControlEvent::Next => 2,
                    MediaControlEvent::Previous => 3,
                    MediaControlEvent::Play => 4,
                    MediaControlEvent::Pause | MediaControlEvent::Stop => 5,
                    _ => return,
                };
                let _ = send.try_send(action);
            })
            .ok()?;
        Some(Self {
            controls,
            actions,
            path: String::new(),
            #[cfg(target_os = "windows")]
            window,
        })
    }

    pub(crate) fn poll(&self) -> Value {
        #[cfg(target_os = "windows")]
        self.window.dispatch();
        json!({"ok": true, "action": self.actions.try_recv().unwrap_or(0)})
    }

    pub(crate) fn update(&mut self, req: &Value) {
        let seconds = |key: &str| {
            req[key]
                .as_f64()
                .and_then(|v| Duration::try_from_secs_f64(v).ok())
        };
        let path = req["path"].as_str().unwrap_or_default();
        if self.path != path
            && self
                .controls
                .set_metadata(MediaMetadata {
                    title: req["title"].as_str(),
                    artist: req["artist"].as_str(),
                    album: req["album"].as_str(),
                    duration: seconds("duration"),
                    ..Default::default()
                })
                .is_ok()
        {
            self.path = path.to_owned();
        }
        let progress = seconds("elapsed").map(MediaPosition);
        let state = if req["playing"].as_bool().unwrap_or(false) {
            MediaPlayback::Playing { progress }
        } else {
            MediaPlayback::Paused { progress }
        };
        let _ = self.controls.set_playback(state);
    }
}
