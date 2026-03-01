//! First-time setup flow

use anyhow::Result;
use nanocode_core::NcConfig;
use nanocode_hf::{
    find_installed_quant, recommend, Downloader, HardwareInfo, QWEN3_4B_QUANTIZATIONS,
};
use std::io::{self, Write};
use tokio::sync::mpsc;

/// Run first-time setup
pub async fn run_first_time_setup(config: &mut NcConfig) -> Result<()> {
    println!("╭─────────────────────────────────────────────────────────────╮");
    println!("│  Nano Code — Primeiro Setup                                │");
    println!("╰─────────────────────────────────────────────────────────────╯");
    println!();

    // Detect hardware
    println!("Detecting hardware...");
    let hw = HardwareInfo::detect();
    println!("Hardware: {}", hw.display());
    println!();

    // Show quantizations
    println!("Available quantizations:");
    println!();
    println!(
        "  {:<4} {:<12} {:>10} {:>12} {}",
        "#", "Quantização", "Tamanho", "Qualidade", "Notas"
    );
    println!("  {}", "-".repeat(72));

    // Find recommended
    let recommended = recommend(&hw);

    // Find installed
    let installed = find_installed_quant(&NcConfig::models_dir());

    // Get names for comparison
    let installed_name = installed.map(|q| q.name);
    let recommended_name = recommended.map(|q| q.name);

    for (i, quant) in QWEN3_4B_QUANTIZATIONS.iter().enumerate() {
        let mut notes = String::new();

        if installed_name == Some(quant.name) {
            notes.push_str("✓ Instalada");
        }

        if recommended_name == Some(quant.name) {
            if !notes.is_empty() {
                notes.push_str(" · ");
            }
            notes.push_str("★ Recomendada");
        }

        println!(
            "  {:<4} {:<12} {:>10} {:>12} {}",
            i + 1,
            quant.name,
            quant.size_human(),
            quant.quality.label(),
            notes
        );

        if let Some(note) = quant.notes {
            println!("                         └── {}", note);
        }
    }

    let default_idx = if let Some(installed_q) = installed {
        QWEN3_4B_QUANTIZATIONS
            .iter()
            .position(|q| q.name == installed_q.name)
            .unwrap_or(0)
    } else if let Some(recommended_q) = recommended {
        QWEN3_4B_QUANTIZATIONS
            .iter()
            .position(|q| q.name == recommended_q.name)
            .unwrap_or(0)
    } else {
        0
    };

    println!();
    println!(
        "Selecione a quantização (1-{}), Enter para padrão [{}]:",
        QWEN3_4B_QUANTIZATIONS.len(),
        default_idx + 1
    );

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    let selected_idx = if input.is_empty() {
        default_idx
    } else {
        match input.parse::<usize>() {
            Ok(n) if n >= 1 && n <= QWEN3_4B_QUANTIZATIONS.len() => n - 1,
            _ => {
                println!("Entrada inválida, usando padrão.");
                default_idx
            }
        }
    };

    let selected = &QWEN3_4B_QUANTIZATIONS[selected_idx];

    println!("Selected: {} ({})", selected.name, selected.size_human());

    if matches!(
        selected.quality,
        nanocode_hf::QualityTier::Low | nanocode_hf::QualityTier::VeryLow
    ) {
        println!(
            "Aviso: quantizações Q2/IQ2/IQ1 podem degradar bastante a qualidade de raciocínio."
        );
    }

    config.active_quant = Some(selected.name.to_string());
    config.save()?;

    let selected_was_installed = installed_name == Some(selected.name);

    // Download if selected quantization is not installed
    if !selected_was_installed {
        println!();
        println!("Downloading model...");

        let downloader = Downloader::new();
        let (tx, mut rx) = mpsc::channel(100);

        // Spawn download task
        let dest_dir = NcConfig::models_dir();
        let quant = selected;

        let handle = tokio::spawn(async move { downloader.download(quant, &dest_dir, tx).await });

        // Show progress
        while let Some(progress) = rx.recv().await {
            let percent = (progress.downloaded as f64 / progress.total as f64 * 100.0) as u32;
            let bar_len: usize = 30;
            let filled: usize = bar_len * progress.downloaded as usize / progress.total as usize;
            let bar: String = format!(
                "{}{}",
                "█".repeat(filled),
                "░".repeat(bar_len.saturating_sub(filled))
            );

            let speed_mb = progress.speed_bps / (1024 * 1024);

            print!(
                "\r  [{}] {:>3}% · {} / {} · {} MB/s · ~{}s     ",
                bar,
                percent,
                format_size(progress.downloaded),
                format_size(progress.total),
                speed_mb,
                progress.eta_seconds
            );
            io::stdout().flush().ok();
        }

        println!();

        // Wait for completion
        match handle.await? {
            Ok(path) => {
                println!("Download complete: {:?}", path);
                config.active_quant = Some(selected.name.to_string());
                config.save()?;
            }
            Err(e) => {
                println!("Download failed: {}", e);
                return Err(e);
            }
        }
    }

    if selected_was_installed {
        println!("Modelo já está instalado localmente para a quantização selecionada.");
    }

    println!();
    println!("Setup complete! Run 'nanocode' to start.");

    Ok(())
}

fn format_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else {
        format!("{} KB", bytes / KB)
    }
}
