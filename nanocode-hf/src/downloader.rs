//! Model downloader

use crate::catalog::ModelSpec;
use crate::quantization::{is_compatible_quant_size, QuantizationVariant};
use crate::registry::get_download_url;
use anyhow::Result;
use futures_util::StreamExt;
use reqwest::Client;
use reqwest::StatusCode;
use std::path::{Path, PathBuf};
use tokio::sync::mpsc;

/// Download progress
#[derive(Debug, Clone)]
pub struct DownloadProgress {
    pub downloaded: u64,
    pub total: u64,
    pub speed_bps: u64,
    pub eta_seconds: u64,
}

/// Downloader for models
pub struct Downloader {
    client: Client,
    hf_token: Option<String>,
}

impl Downloader {
    pub fn new() -> Self {
        // Try to load HF token from environment
        let hf_token = std::env::var("HF_TOKEN").ok();

        Self {
            client: Client::new(),
            hf_token,
        }
    }

    /// Download a quantization
    pub async fn download(
        &self,
        model: &ModelSpec,
        quant: &QuantizationVariant,
        dest_dir: &Path,
        progress_tx: mpsc::Sender<DownloadProgress>,
    ) -> Result<PathBuf> {
        // Create destination directory
        std::fs::create_dir_all(dest_dir)?;

        let url = get_download_url(model, quant);
        let dest_path = dest_dir.join(quant.filename);
        let part_path = dest_dir.join(format!("{}.part", quant.filename));

        // Check if already downloaded
        if dest_path.exists() {
            let existing_size = std::fs::metadata(&dest_path)?.len();
            if is_compatible_quant_size(existing_size, quant.size_bytes) {
                return Ok(dest_path);
            }
            std::fs::remove_file(&dest_path)?;
        }

        // Resume from partial download
        let mut start_offset = if part_path.exists() {
            std::fs::metadata(&part_path)?.len()
        } else {
            0
        };

        // Build request
        let mut request = self.client.get(&url);

        if let Some(ref token) = self.hf_token {
            request = request.header("Authorization", format!("Bearer {}", token));
        }

        if start_offset > 0 {
            request = request.header("Range", format!("bytes={}-", start_offset));
        }

        // Download with streaming
        let response = request.send().await?;
        let status = response.status();

        if !status.is_success() {
            let body_preview = response
                .text()
                .await
                .unwrap_or_else(|_| "<unreadable body>".to_string());
            let body_preview = body_preview.trim().replace('\n', " ");
            return Err(anyhow::anyhow!(
                "Download request failed for {} (status {}): {}",
                quant.filename,
                status,
                truncate_preview(&body_preview, 200)
            ));
        }

        let mut file = if start_offset > 0 && status != StatusCode::PARTIAL_CONTENT {
            start_offset = 0;
            std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&part_path)?
        } else {
            std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&part_path)?
        };

        let server_total_size = response.content_length().map(|c| c + start_offset);
        let total_size = server_total_size.unwrap_or(quant.size_bytes);

        let mut downloaded = start_offset;
        let mut last_update = std::time::Instant::now();
        let mut bytes_since_last: u64 = 0;

        use std::io::Write;

        let mut stream = response.bytes_stream();
        while let Some(chunk_result) = stream.next().await {
            let chunk = chunk_result?;
            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;
            bytes_since_last += chunk.len() as u64;

            // Update progress every 500ms
            let elapsed = last_update.elapsed();
            if elapsed.as_millis() >= 500 {
                let speed_bps = bytes_since_last as u64 * 1000 / elapsed.as_millis().max(1) as u64;
                let remaining = total_size.saturating_sub(downloaded);
                let eta = if speed_bps > 0 {
                    remaining / speed_bps
                } else {
                    0
                };

                progress_tx
                    .send(DownloadProgress {
                        downloaded,
                        total: total_size,
                        speed_bps,
                        eta_seconds: eta,
                    })
                    .await
                    .ok();

                last_update = std::time::Instant::now();
                bytes_since_last = 0;
            }
        }

        // Final flush
        file.flush()?;

        // Validate final size. Prefer exact server-reported size when available.
        let final_size = std::fs::metadata(&part_path)?.len();
        if let Some(expected_server_size) = server_total_size {
            if final_size != expected_server_size {
                let _ = std::fs::remove_file(&part_path);
                return Err(anyhow::anyhow!(
                    "Downloaded size mismatch: expected {}, got {}",
                    expected_server_size,
                    final_size
                ));
            }
        } else if !is_compatible_quant_size(final_size, quant.size_bytes) {
            let _ = std::fs::remove_file(&part_path);
            return Err(anyhow::anyhow!(
                "Downloaded size mismatch: expected ~{}, got {}",
                quant.size_bytes,
                final_size
            ));
        }

        // Rename to final location only after validation
        std::fs::rename(&part_path, &dest_path)?;

        Ok(dest_path)
    }
}

impl Default for Downloader {
    fn default() -> Self {
        Self::new()
    }
}

fn truncate_preview(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    input.chars().take(max_chars).collect::<String>() + "..."
}
