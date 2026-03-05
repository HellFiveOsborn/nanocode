//! Session management

use crate::types::LlmMessage;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use uuid::Uuid;

/// Session info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub model: String,
    pub agent: String,
    pub message_count: usize,
}

/// Session logger
pub struct SessionLogger {
    session_id: String,
    path: PathBuf,
    file: BufWriter<File>,
    message_count: usize,
}

impl SessionLogger {
    /// Create a new session logger
    pub fn new(dir: &Path, _model: &str, _agent: &str) -> Result<Self> {
        std::fs::create_dir_all(dir)?;

        let session_id = Uuid::new_v4().to_string();
        let filename = format!(
            "session_{}_{}.jsonl",
            session_id,
            Utc::now().format("%Y%m%d_%H%M%S")
        );
        let path = dir.join(filename);

        let file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)?;
        let writer = BufWriter::new(file);

        Ok(Self {
            session_id,
            path,
            file: writer,
            message_count: 0,
        })
    }

    /// Resume logging to an existing session file.
    pub fn resume(dir: &Path, session_id_query: &str) -> Result<Self> {
        std::fs::create_dir_all(dir)?;

        let Some((resolved_id, path)) = find_session_path_by_id_sync(dir, session_id_query)? else {
            return Err(anyhow::anyhow!("Session '{}' not found", session_id_query));
        };

        let existing_count = load_session_sync(&path)?.len();
        let file = OpenOptions::new().append(true).open(&path)?;
        let writer = BufWriter::new(file);

        Ok(Self {
            session_id: resolved_id,
            path,
            file: writer,
            message_count: existing_count,
        })
    }

    /// Append a message to the session
    pub async fn append(&mut self, msg: &LlmMessage) -> Result<()> {
        let json = serde_json::to_string(msg)?;
        writeln!(self.file, "{}", json)?;
        self.file.flush()?;
        self.message_count += 1;
        Ok(())
    }

    /// Get session ID
    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Get session file path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get message count
    pub fn message_count(&self) -> usize {
        self.message_count
    }

    /// Finish the session
    pub fn finish(&mut self) -> Result<()> {
        self.file.flush()?;
        Ok(())
    }
}

fn session_id_from_path(path: &Path) -> Option<String> {
    let stem = path.file_stem()?.to_str()?;
    let rest = stem.strip_prefix("session_")?;
    let id = rest.split('_').next().unwrap_or(rest);
    if id.is_empty() {
        None
    } else {
        Some(id.to_string())
    }
}

fn session_paths_sorted(dir: &Path) -> Result<Vec<PathBuf>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut entries = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().map(|ext| ext == "jsonl").unwrap_or(false))
        .collect::<Vec<_>>();

    entries.sort_by(|a, b| {
        let a_modified = std::fs::metadata(a)
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let b_modified = std::fs::metadata(b)
            .ok()
            .and_then(|m| m.modified().ok())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        b_modified.cmp(&a_modified)
    });
    Ok(entries)
}

/// Load messages from a session file
pub fn load_session_sync(path: &Path) -> Result<Vec<LlmMessage>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let mut messages = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if let Ok(msg) = serde_json::from_str::<LlmMessage>(&line) {
            messages.push(msg);
        }
    }

    Ok(messages)
}

/// Async wrapper to preserve existing call sites.
pub async fn load_session(path: &Path) -> Result<Vec<LlmMessage>> {
    load_session_sync(path)
}

/// Find a session path by full or partial ID.
pub fn find_session_path_by_id_sync(
    dir: &Path,
    session_id_query: &str,
) -> Result<Option<(String, PathBuf)>> {
    let query = session_id_query.trim().to_ascii_lowercase();
    if query.is_empty() {
        return Ok(None);
    }

    for path in session_paths_sorted(dir)? {
        let Some(id) = session_id_from_path(&path) else {
            continue;
        };
        if id.to_ascii_lowercase().starts_with(&query) {
            return Ok(Some((id, path)));
        }
    }

    Ok(None)
}

/// Load a specific session by full or partial ID.
pub fn load_session_by_id_sync(
    dir: &Path,
    session_id_query: &str,
) -> Result<Option<(String, Vec<LlmMessage>)>> {
    let Some((id, path)) = find_session_path_by_id_sync(dir, session_id_query)? else {
        return Ok(None);
    };
    let messages = load_session_sync(&path)?;
    Ok(Some((id, messages)))
}

/// Return the latest session ID.
pub fn latest_session_id_sync(dir: &Path) -> Result<Option<String>> {
    for path in session_paths_sorted(dir)? {
        if let Some(id) = session_id_from_path(&path) {
            return Ok(Some(id));
        }
    }
    Ok(None)
}

/// Load latest session and return (session_id, messages)
pub fn load_latest_session_sync(dir: &Path) -> Result<Option<(String, Vec<LlmMessage>)>> {
    for path in session_paths_sorted(dir)? {
        if let Some(id) = session_id_from_path(&path) {
            let messages = load_session_sync(&path)?;
            return Ok(Some((id, messages)));
        }
    }
    Ok(None)
}

/// Load the latest session from a directory
pub async fn load_latest_session(dir: &Path) -> Result<Option<Vec<LlmMessage>>> {
    Ok(load_latest_session_sync(dir)?.map(|(_, messages)| messages))
}

/// List all sessions
pub fn list_sessions_sync(dir: &Path) -> Result<Vec<SessionInfo>> {
    let mut sessions = Vec::new();

    for path in session_paths_sorted(dir)? {
        let Some(id) = session_id_from_path(&path) else {
            continue;
        };
        let messages = load_session_sync(&path)?;
        let start_time = std::fs::metadata(&path)
            .ok()
            .and_then(|m| m.modified().ok())
            .map(DateTime::<Utc>::from)
            .unwrap_or_else(Utc::now);

        sessions.push(SessionInfo {
            id,
            start_time,
            end_time: None,
            model: "unknown".to_string(),
            agent: "unknown".to_string(),
            message_count: messages.len(),
        });
    }

    Ok(sessions)
}

/// Async wrapper to preserve existing call sites.
pub async fn list_sessions(dir: &Path) -> Result<Vec<SessionInfo>> {
    list_sessions_sync(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_session_id_from_legacy_and_new_names() {
        let legacy = PathBuf::from("session_abcd1234_20260304_120000.jsonl");
        let modern =
            PathBuf::from("session_123e4567-e89b-12d3-a456-426614174000_20260304_120000.jsonl");

        assert_eq!(session_id_from_path(&legacy).as_deref(), Some("abcd1234"));
        assert_eq!(
            session_id_from_path(&modern).as_deref(),
            Some("123e4567-e89b-12d3-a456-426614174000")
        );
    }
}
