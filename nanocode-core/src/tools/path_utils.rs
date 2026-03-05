//! Shared path validation for file tools.
//!
//! Policy:
//! - Accept both relative and absolute paths.
//! - Resolve relative paths against `std::env::current_dir()`.
//! - Reject traversal that escapes above the working directory.
//! - Normalize the final path to remove `.` / `..` components.

use crate::types::ToolError;
use std::path::{Component, Path, PathBuf};

/// Validate and resolve a tool path argument.
///
/// Returns a canonical-ish path that is guaranteed to be within (or equal to)
/// the current working directory.
pub fn validate_and_resolve(raw: &str) -> Result<PathBuf, ToolError> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Err(ToolError::InvalidArguments(
            "Path cannot be empty".to_string(),
        ));
    }

    let cwd = std::env::current_dir().map_err(|e| {
        ToolError::ExecutionFailed(format!("Cannot determine working directory: {e}"))
    })?;

    // Build the full path: if absolute, use as-is; if relative, join with cwd.
    let full = if Path::new(raw).is_absolute() {
        PathBuf::from(raw)
    } else {
        cwd.join(raw)
    };

    // Normalize: resolve `.` and `..` without hitting the filesystem (which may
    // not exist yet for write targets).
    let normalized = normalize_path(&full);

    // Security: the resolved path must start with (or be equal to) the cwd.
    if !normalized.starts_with(&cwd) {
        return Err(ToolError::InvalidArguments(format!(
            "Path escapes working directory: {}",
            raw,
        )));
    }

    Ok(normalized)
}

/// Normalize a path by resolving `.` and `..` logically (no filesystem access).
fn normalize_path(path: &Path) -> PathBuf {
    let mut result = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                // Only pop if we have something beyond the root.
                if result.parent().is_some()
                    && !matches!(result.components().next_back(), Some(Component::RootDir))
                {
                    result.pop();
                }
            }
            Component::CurDir => { /* skip */ }
            other => result.push(other),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_path_resolves_to_cwd() {
        let result = validate_and_resolve("foo.txt");
        assert!(result.is_ok());
        let p = result.unwrap();
        assert!(p.is_absolute());
        assert!(p.ends_with("foo.txt"));
    }

    #[test]
    fn traversal_above_cwd_is_rejected() {
        // Going up enough levels should escape cwd.
        let result = validate_and_resolve("../../../../etc/passwd");
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("escapes working directory"));
    }

    #[test]
    fn absolute_path_within_cwd_is_accepted() {
        let cwd = std::env::current_dir().unwrap();
        let target = cwd.join("some_file.rs");
        let result = validate_and_resolve(target.to_str().unwrap());
        assert!(result.is_ok());
    }

    #[test]
    fn absolute_path_outside_cwd_is_rejected() {
        let result = validate_and_resolve("/etc/passwd");
        // This should fail unless cwd happens to be /etc, which is unlikely.
        if !std::env::current_dir()
            .unwrap()
            .starts_with("/etc")
        {
            assert!(result.is_err());
        }
    }

    #[test]
    fn empty_path_is_rejected() {
        let result = validate_and_resolve("");
        assert!(result.is_err());
    }
}
