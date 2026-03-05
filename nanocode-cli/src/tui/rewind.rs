use anyhow::{anyhow, Context, Result};
use nanocode_core::NcConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Clone)]
pub struct RewindChange {
    pub path: String,
    pub existed_before: bool,
    pub before_content: Option<Vec<u8>>,
}

#[derive(Clone, Serialize, Deserialize)]
struct RewindFileSnapshot {
    path: String,
    existed_before: bool,
    blob_hash: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
struct RewindEntry {
    id: u64,
    changes: Vec<RewindFileSnapshot>,
}

#[derive(Serialize, Deserialize, Default)]
struct RewindStack {
    entries: Vec<RewindEntry>,
}

pub enum RewindOutcome {
    Empty,
    Applied {
        files: Vec<String>,
        remaining: usize,
    },
}

#[derive(Default)]
pub struct RewindManager {
    session_id: Option<String>,
    session_dir: Option<PathBuf>,
    git_dir: Option<PathBuf>,
    stack: RewindStack,
    next_id: u64,
}

impl RewindManager {
    pub fn set_session(&mut self, session_id: &str) -> Result<()> {
        if self.session_id.as_deref() == Some(session_id) {
            return Ok(());
        }

        let session_dir = NcConfig::data_dir().join("rewind").join(session_id);
        let git_dir = session_dir.join("git");
        std::fs::create_dir_all(&session_dir).with_context(|| {
            format!(
                "Failed to create rewind session directory at {}",
                session_dir.display()
            )
        })?;
        init_bare_repo(&git_dir)?;

        let stack_path = session_dir.join("stack.json");
        let stack = if stack_path.exists() {
            let raw = std::fs::read_to_string(&stack_path).with_context(|| {
                format!("Failed to read rewind stack at {}", stack_path.display())
            })?;
            serde_json::from_str::<RewindStack>(&raw).with_context(|| {
                format!("Invalid rewind stack format at {}", stack_path.display())
            })?
        } else {
            RewindStack::default()
        };

        let next_id = stack
            .entries
            .iter()
            .map(|entry| entry.id)
            .max()
            .unwrap_or(0)
            .saturating_add(1);

        self.session_id = Some(session_id.to_string());
        self.session_dir = Some(session_dir);
        self.git_dir = Some(git_dir);
        self.stack = stack;
        self.next_id = next_id;
        Ok(())
    }

    pub fn record_change_set(&mut self, changes: Vec<RewindChange>) -> Result<()> {
        if changes.is_empty() {
            return Ok(());
        }
        let git_dir = self
            .git_dir
            .as_ref()
            .ok_or_else(|| anyhow!("Rewind session is not initialized yet"))?;

        let mut snapshots: Vec<RewindFileSnapshot> = Vec::new();
        let mut seen = HashSet::<String>::new();
        for change in changes {
            if !seen.insert(change.path.clone()) {
                continue;
            }
            let blob_hash = if change.existed_before {
                let bytes = change.before_content.ok_or_else(|| {
                    anyhow!(
                        "Cannot store rewind snapshot for {} (missing previous content)",
                        change.path
                    )
                })?;
                Some(hash_object_bytes(git_dir, &bytes)?)
            } else {
                None
            };

            snapshots.push(RewindFileSnapshot {
                path: change.path,
                existed_before: change.existed_before,
                blob_hash,
            });
        }

        if snapshots.is_empty() {
            return Ok(());
        }

        let entry = RewindEntry {
            id: self.next_id,
            changes: snapshots,
        };
        self.next_id = self.next_id.saturating_add(1);
        self.stack.entries.push(entry);
        self.persist_stack()
    }

    pub fn rewind_once(&mut self) -> Result<RewindOutcome> {
        let git_dir = self
            .git_dir
            .as_ref()
            .ok_or_else(|| anyhow!("Rewind session is not initialized yet"))?;
        let Some(entry) = self.stack.entries.last().cloned() else {
            return Ok(RewindOutcome::Empty);
        };

        let mut files = Vec::new();
        for snapshot in &entry.changes {
            let path = PathBuf::from(&snapshot.path);
            if snapshot.existed_before {
                let blob = snapshot.blob_hash.as_deref().ok_or_else(|| {
                    anyhow!("Corrupted rewind entry {} for {}", entry.id, snapshot.path)
                })?;
                let bytes = read_blob_bytes(git_dir, blob)?;
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent).with_context(|| {
                        format!(
                            "Failed to create parent directory while rewinding {}",
                            snapshot.path
                        )
                    })?;
                }
                std::fs::write(&path, bytes).with_context(|| {
                    format!("Failed to restore file while rewinding: {}", snapshot.path)
                })?;
            } else if path.exists() {
                std::fs::remove_file(&path).with_context(|| {
                    format!("Failed to remove file while rewinding: {}", snapshot.path)
                })?;
            }
            files.push(snapshot.path.clone());
        }

        let _ = self.stack.entries.pop();
        self.persist_stack()?;
        Ok(RewindOutcome::Applied {
            files,
            remaining: self.stack.entries.len(),
        })
    }

    fn persist_stack(&self) -> Result<()> {
        let session_dir = self
            .session_dir
            .as_ref()
            .ok_or_else(|| anyhow!("Rewind session is not initialized yet"))?;
        let stack_path = session_dir.join("stack.json");
        let payload = serde_json::to_string_pretty(&self.stack)?;
        std::fs::write(&stack_path, payload)
            .with_context(|| format!("Failed to write rewind stack at {}", stack_path.display()))
    }
}

fn init_bare_repo(git_dir: &Path) -> Result<()> {
    if git_dir.exists() {
        return Ok(());
    }

    if let Some(parent) = git_dir.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }

    let output = Command::new("git")
        .arg("init")
        .arg("--bare")
        .arg(git_dir)
        .output()
        .with_context(|| "Failed to execute `git init --bare` for rewind storage".to_string())?;
    if !output.status.success() {
        return Err(anyhow!(
            "Failed to initialize rewind git store: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(())
}

fn hash_object_bytes(git_dir: &Path, bytes: &[u8]) -> Result<String> {
    let mut child = Command::new("git")
        .arg("--git-dir")
        .arg(git_dir)
        .args(["hash-object", "-w", "--stdin"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| "Failed to execute `git hash-object` for rewind snapshot".to_string())?;

    if let Some(stdin) = child.stdin.as_mut() {
        stdin
            .write_all(bytes)
            .with_context(|| "Failed to write rewind snapshot bytes".to_string())?;
    }

    let output = child
        .wait_with_output()
        .with_context(|| "Failed to wait for `git hash-object`".to_string())?;
    if !output.status.success() {
        return Err(anyhow!(
            "Failed to persist rewind snapshot: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }

    let hash = String::from_utf8(output.stdout)?.trim().to_string();
    if hash.is_empty() {
        return Err(anyhow!("Invalid rewind snapshot hash returned by git"));
    }
    Ok(hash)
}

fn read_blob_bytes(git_dir: &Path, hash: &str) -> Result<Vec<u8>> {
    let output = Command::new("git")
        .arg("--git-dir")
        .arg(git_dir)
        .args(["cat-file", "-p", hash])
        .output()
        .with_context(|| "Failed to execute `git cat-file` for rewind restore".to_string())?;
    if !output.status.success() {
        return Err(anyhow!(
            "Failed to read rewind snapshot {}: {}",
            hash,
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    Ok(output.stdout)
}
