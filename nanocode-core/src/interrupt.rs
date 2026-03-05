use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Canonical interruption error used across core + CLI.
pub const USER_INTERRUPTED_ERROR: &str = "Generation interrupted by user";

/// Build canonical interruption error.
pub fn user_interrupted_error() -> String {
    USER_INTERRUPTED_ERROR.to_string()
}

/// True when an error string is the canonical interruption error.
pub fn is_user_interrupted_error(err: &str) -> bool {
    err == USER_INTERRUPTED_ERROR
}

/// Set interruption signal with cross-thread visibility.
pub fn set_interrupt_signal(interrupt_signal: &Arc<AtomicBool>) {
    interrupt_signal.store(true, Ordering::Release);
}

/// Clear interruption signal with cross-thread visibility.
pub fn clear_interrupt_signal(interrupt_signal: &Arc<AtomicBool>) {
    interrupt_signal.store(false, Ordering::Release);
}

/// Return canonical interruption error when signal is set.
pub fn check_interrupt_signal(interrupt_signal: Option<&Arc<AtomicBool>>) -> Result<(), String> {
    if interrupt_signal
        .map(|signal| signal.load(Ordering::Acquire))
        .unwrap_or(false)
    {
        Err(user_interrupted_error())
    } else {
        Ok(())
    }
}
