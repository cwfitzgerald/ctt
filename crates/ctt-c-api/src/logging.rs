//! Routing of the ctt library's [`log`] output to a C callback.
//!
//! ctt's core emits records through the [`log`] facade. This module installs a
//! global logger the first time [`ctt_set_log_callback`] or
//! [`ctt_set_log_level`] is called and forwards each record to a caller-supplied
//! callback. Until a callback is set, records are discarded.

use std::ffi::{CString, c_char, c_void};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Mutex, Once};

/// Severity of a log record.
///
/// Ordered from least to most verbose. As an argument to
/// [`ctt_set_log_level`], a level enables that severity and everything above
/// it (e.g. `CTT_LOG_LEVEL_INFO` passes error, warning, and info records but
/// drops debug and trace). `CTT_LOG_LEVEL_OFF` disables logging entirely and
/// never appears as the `level` of a callback invocation.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Off = 0,
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

/// Callback invoked for each log record ctt emits.
///
/// `level` is the record's severity (never `CTT_LOG_LEVEL_OFF`). `message` is
/// a NUL-terminated UTF-8 string valid only for the duration of the call —
/// copy it if you need to retain it. `user_data` is the opaque pointer passed
/// to [`ctt_set_log_callback`], forwarded unchanged.
///
/// The callback may be invoked from any thread and from multiple threads
/// concurrently; it must be thread-safe. It must not call back into ctt.
///
/// A `NULL` function pointer clears the callback (see [`ctt_set_log_callback`]).
pub type LogCallback =
    Option<unsafe extern "C" fn(level: LogLevel, message: *const c_char, user_data: *mut c_void)>;

/// Current maximum level. Defaults to [`LogLevel::Trace`] so that, once a
/// callback is installed, every record is delivered ("all messages").
static LOG_LEVEL: AtomicU8 = AtomicU8::new(LogLevel::Trace as u8);

struct CallbackSlot {
    callback: LogCallback,
    user_data: *mut c_void,
}

// The raw `user_data` pointer is only ever handed back to the callback that
// supplied it; ctt never dereferences it. Guarding the slot with a `Mutex`
// makes concurrent updates and reads safe.
unsafe impl Send for CallbackSlot {}

static CALLBACK: Mutex<CallbackSlot> = Mutex::new(CallbackSlot {
    callback: None,
    user_data: std::ptr::null_mut(),
});

// `LogCallback` is itself an `Option`, so `None` is the "unset" state.

struct CttLogger;

impl log::Log for CttLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        level_to_filter(current_level()) >= metadata.level()
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        // Copy the callback out from under the lock so the user code runs
        // without holding it (a callback that re-enters `ctt_set_log_callback`
        // would otherwise deadlock).
        let (callback, user_data) = {
            let slot = CALLBACK.lock().unwrap_or_else(|e| e.into_inner());
            (slot.callback, slot.user_data)
        };
        let Some(callback) = callback else {
            return;
        };

        let message = format!("{}", record.args());
        let message = CString::new(message)
            .unwrap_or_else(|_| CString::new("log message contained interior NUL").unwrap());

        // Safety: `callback` is a valid function pointer supplied by the caller
        // and `user_data` is theirs to interpret; the message pointer is valid
        // for the duration of the call.
        unsafe {
            callback(LogLevel::from(record.level()), message.as_ptr(), user_data);
        }
    }

    fn flush(&self) {}
}

static LOGGER: CttLogger = CttLogger;
static INIT: Once = Once::new();

/// Install the global [`log`] logger once. Idempotent; safe to call from any
/// entry point. If the host application has already installed its own logger,
/// `set_logger` fails and ctt's records will not reach the callback — this is
/// expected and not an error.
fn ensure_logger_installed() {
    INIT.call_once(|| {
        let _ = log::set_logger(&LOGGER);
        log::set_max_level(level_to_filter(current_level()));
    });
}

fn current_level() -> LogLevel {
    match LOG_LEVEL.load(Ordering::Relaxed) {
        0 => LogLevel::Off,
        1 => LogLevel::Error,
        2 => LogLevel::Warn,
        3 => LogLevel::Info,
        4 => LogLevel::Debug,
        _ => LogLevel::Trace,
    }
}

fn level_to_filter(level: LogLevel) -> log::LevelFilter {
    match level {
        LogLevel::Off => log::LevelFilter::Off,
        LogLevel::Error => log::LevelFilter::Error,
        LogLevel::Warn => log::LevelFilter::Warn,
        LogLevel::Info => log::LevelFilter::Info,
        LogLevel::Debug => log::LevelFilter::Debug,
        LogLevel::Trace => log::LevelFilter::Trace,
    }
}

impl From<log::Level> for LogLevel {
    fn from(level: log::Level) -> Self {
        match level {
            log::Level::Error => LogLevel::Error,
            log::Level::Warn => LogLevel::Warn,
            log::Level::Info => LogLevel::Info,
            log::Level::Debug => LogLevel::Debug,
            log::Level::Trace => LogLevel::Trace,
        }
    }
}

/// Set the callback that receives ctt's log records, replacing any previous
/// one. Pass `NULL` to stop delivery.
///
/// `user_data` is stored and forwarded unchanged to every invocation of
/// `callback`; ctt never dereferences it. It is the caller's responsibility to
/// keep whatever it points at alive until the callback is cleared or replaced.
///
/// The first call to this function (or to [`ctt_set_log_level`]) installs
/// ctt's global logger. If the host process has already installed a different
/// `log` logger, ctt's records cannot be routed here.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_set_log_callback(callback: LogCallback, user_data: *mut c_void) {
    {
        let mut slot = CALLBACK.lock().unwrap_or_else(|e| e.into_inner());
        slot.callback = callback;
        slot.user_data = user_data;
    }
    ensure_logger_installed();
}

/// Set the maximum log level ctt will emit. Records more verbose than `level`
/// are dropped before reaching the callback.
///
/// The default is `CTT_LOG_LEVEL_TRACE` (all messages). Pass
/// `CTT_LOG_LEVEL_OFF` to disable logging.
#[unsafe(no_mangle)]
pub extern "C" fn ctt_set_log_level(level: LogLevel) {
    LOG_LEVEL.store(level as u8, Ordering::Relaxed);
    ensure_logger_installed();
    log::set_max_level(level_to_filter(level));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;
    use std::sync::Mutex as StdMutex;

    static CAPTURED: StdMutex<Vec<(LogLevel, String)>> = StdMutex::new(Vec::new());

    unsafe extern "C" fn capture(level: LogLevel, message: *const c_char, _user: *mut c_void) {
        let msg = unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned();
        CAPTURED
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((level, msg));
    }

    #[test]
    fn callback_receives_records_at_configured_level() {
        unsafe { ctt_set_log_callback(Some(capture), std::ptr::null_mut()) };
        ctt_set_log_level(LogLevel::Warn);

        log::info!("info message");
        log::warn!("warn message");
        log::error!("error message");

        let captured = CAPTURED.lock().unwrap_or_else(|e| e.into_inner());
        // Info is below the Warn threshold and must be dropped.
        assert!(
            !captured.iter().any(|(_, m)| m == "info message"),
            "info should be filtered out, got: {captured:?}"
        );
        assert!(
            captured
                .iter()
                .any(|(l, m)| *l == LogLevel::Warn && m == "warn message")
        );
        assert!(
            captured
                .iter()
                .any(|(l, m)| *l == LogLevel::Error && m == "error message")
        );
    }
}
