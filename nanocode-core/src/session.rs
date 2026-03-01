//! Session management

use crate::types::LlmMessage;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
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
    session_id: Uuid,
    file: BufWriter<File>,
    message_count: usize,
}

impl SessionLogger {
    /// Create a new session logger
    pub fn new(dir: &Path, _model: &str, _agent: &str) -> Result<Self> {
        std::fs::create_dir_all(dir)?;

        let session_id = Uuid::new_v4();
        let filename = format!(
            "session_{}_{}.jsonl",
            session_id.to_string()[..8].to_string(),
            Utc::now().format("%Y%m%d_%H%M%S")
        );
        let path = dir.join(filename);

        let file = File::create(&path)?;
        let writer = BufWriter::new(file);

        Ok(Self {
            session_id,
            file: writer,
            message_count: 0,
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
    pub fn session_id(&self) -> Uuid {
        self.session_id
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

/// Load messages from a session file
pub async fn load_session(path: &Path) -> Result<Vec<LlmMessage>> {
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

/// Load the latest session from a directory
pub async fn load_latest_session(dir: &Path) -> Result<Option<Vec<LlmMessage>>> {
    if !dir.exists() {
        return Ok(None);
    }

    let mut entries = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "jsonl")
                .unwrap_or(false)
        })
        .collect::<Vec<_>>();

    // Sort by modification time (newest first)
    entries.sort_by(|a, b| {
        let a_meta = a.metadata().ok();
        let b_meta = b.metadata().ok();
        match (a_meta, b_meta) {
            (Some(a), Some(b)) => b
                .modified()
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
                .cmp(&a.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH)),
            _ => std::cmp::Ordering::Equal,
        }
    });

    if let Some(entry) = entries.first() {
        let messages = load_session(&entry.path()).await?;
        Ok(Some(messages))
    } else {
        Ok(None)
    }
}

/// List all sessions
pub async fn list_sessions(dir: &Path) -> Result<Vec<SessionInfo>> {
    if !dir.exists() {
        return Ok(Vec::new());
    }

    let mut sessions = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|ext| ext == "jsonl").unwrap_or(false) {
            if let Ok(messages) = load_session(&path).await {
                let start_time = messages
                    .first()
                    .and_then(|_| {
                        // Try to extract timestamp from filename or first message
                        None
                    })
                    .unwrap_or_else(Utc::now);

                sessions.push(SessionInfo {
                    id: path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string(),
                    start_time,
                    end_time: None,
                    model: "qwen3".to_string(),
                    agent: "default".to_string(),
                    message_count: messages.len(),
                });
            }
        }
    }

    Ok(sessions)
}
