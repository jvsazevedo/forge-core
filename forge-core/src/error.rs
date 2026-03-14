//! Thread-local error handling for FFI.
//!
//! Pattern: FFI functions return null / -1 on error and store the message here.
//! The caller retrieves it with `forge_last_error()`.

use std::cell::RefCell;

use furst_macro::furst_export;

thread_local! {
    static LAST_ERROR: RefCell<String> = RefCell::new(String::new());
}

/// Store an error message for the current thread.
pub fn set_last_error(msg: String) {
    LAST_ERROR.with(|e| *e.borrow_mut() = msg);
}

/// Copy the last error message into the caller's buffer.
///
/// Returns the number of bytes written, or -1 if no error is stored.
/// If the buffer is too small, the message is truncated.
#[furst_export]
pub fn forge_last_error(buf: *mut u8, buf_len: usize) -> i32 {
    LAST_ERROR.with(|e| {
        let msg = e.borrow();
        if msg.is_empty() {
            return -1;
        }
        let bytes = msg.as_bytes();
        let copy_len = bytes.len().min(buf_len);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, copy_len);
        }
        copy_len as i32
    })
}
