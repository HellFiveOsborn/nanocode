//! Model downloader

use crate::quantization::QuantizationVariant;
use crate::registry::get_download_url;
use anyhow::Result;
use futures_util::StreamExt;
use reqwest::Client;
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
        quant: &QuantizationVariant,
        dest_dir: &Path,
        progress_tx: mpsc::Sender<DownloadProgress>,
    ) -> Result<PathBuf> {
        // Create destination directory
        std::fs::create_dir_all(dest_dir)?;

        let url = get_download_url(quant);
        let dest_path = dest_dir.join(quant.filename);
        let part_path = dest_dir.join(format!("{}.part", quant.filename));

        // Check if already downloaded
        if dest_path.exists() {
            tracing::info!("Model already downloaded: {:?}", dest_path);
            return Ok(dest_path);
        }

        // Resume from partial download
        let start_offset = if part_path.exists() {
            std::fs::metadata(&part_path)?.len()
        } else {
            0
        };

        tracing::info!("Starting download from offset {}", start_offset);

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

        let total_size = response
            .content_length()
            .map(|c| c + start_offset)
            .unwrap_or(quant.size_bytes);

        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&part_path)?;

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
                let remaining = total_size - downloaded;
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

        // Rename to final location
        std::fs::rename(&part_path, &dest_path)?;

        // Verify size (allow 5% tolerance)
        let final_size = std::fs::metadata(&dest_path)?.len();
        let min_size = quant.size_bytes * 95 / 100;
        let max_size = quant.size_bytes * 105 / 100;
        if final_size < min_size || final_size > max_size {
            return Err(anyhow::anyhow!(
                "Downloaded size mismatch: expected ~{}, got {}",
                quant.size_bytes,
                final_size
            ));
        }

        Ok(dest_path)
    }
}

impl Default for Downloader {
    fn default() -> Self {
        Self::new()
    }
}
