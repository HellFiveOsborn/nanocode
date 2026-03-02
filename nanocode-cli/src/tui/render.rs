use anyhow::Result;
use nanocode_core::NcConfig;
use nanocode_hf::{list_cached_quants, models};
use nanocode_hf::{ComputeMode, GpuVendor, RuntimeTelemetry};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Clear, Paragraph, Wrap};
use ratatui::Terminal;
use std::io::Stdout;

use super::commands::MAX_SLASH_SUGGESTIONS;
use super::state::{
    AppState, ApprovalOption, ChatItem, InputMode, ModelSetupView, ToolState, UiScreen,
};

const NANO_CODE_VERSION: &str = env!("CARGO_PKG_VERSION");
const SPINNER_FRAMES: [&str; 3] = ["∗", "⁕", "⋇"];
const LOADING_GRADIENT: [Color; 5] = [
    Color::Rgb(255, 216, 0),
    Color::Rgb(255, 175, 0),
    Color::Rgb(255, 130, 5),
    Color::Rgb(250, 80, 15),
    Color::Rgb(225, 5, 0),
];
const ANSI_BRIGHT_BLACK: Color = Color::Indexed(8);
const UI_BG: Color = Color::Rgb(35, 41, 57);
const WELCOME_HIGHLIGHT: &str = "Nano Code";
const CAT_LOGO_LINES: [&str; 3] = ["⠀⠀⡠   ⠴⠒⠒⠢⡠⢤⢤", "⠀⢸   ⡎⠀⠀⢣⠀⢄⠈⢩⡄", "  ⠉⠒⢜⣀⣴⠥⠼⣐⣲⠚⠊"];

#[derive(Clone, Copy)]
struct UiTheme {
    fg: Color,
    bg: Color,
    border: Color,
    accent: Color,
    muted: Color,
    success: Color,
    warning: Color,
    danger: Color,
    info_blue: Color,
    info_cyan: Color,
}

const UI_THEME: UiTheme = UiTheme {
    fg: Color::Rgb(215, 219, 224),
    bg: UI_BG,
    border: Color::Rgb(88, 94, 111),
    accent: Color::Rgb(255, 130, 5),
    muted: ANSI_BRIGHT_BLACK,
    success: Color::Green,
    warning: Color::Yellow,
    danger: Color::Red,
    info_blue: Color::Blue,
    info_cyan: Color::Cyan,
};

fn short_preview(text: &str, n: usize) -> String {
    let mut out = String::new();
    for ch in text.chars().take(n) {
        out.push(ch);
    }
    if text.chars().count() > n {
        out.push('…');
    }
    out
}

fn spinner_char(frame: usize) -> &'static str {
    SPINNER_FRAMES[frame % SPINNER_FRAMES.len()]
}

fn input_prompt_prefix(mode: InputMode) -> &'static str {
    match mode {
        InputMode::Default => "› ",
        InputMode::Slash => "/ ",
    }
}

fn text_lines_or_empty(text: &str) -> Vec<&str> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        vec![""]
    } else {
        lines
    }
}

fn wrap_soft(text: &str, width: usize) -> Vec<String> {
    let mut wrapped = Vec::new();
    let w = width.max(20);

    for raw in text.lines() {
        let mut remaining = raw.trim_end().to_string();
        if remaining.is_empty() {
            wrapped.push(String::new());
            continue;
        }

        while remaining.chars().count() > w {
            let candidate: String = remaining.chars().take(w).collect();
            let split_at = candidate
                .rfind(char::is_whitespace)
                .filter(|idx| *idx > 12)
                .unwrap_or(candidate.len());
            let (head, tail) = remaining.split_at(split_at);
            wrapped.push(head.trim_end().to_string());
            remaining = tail.trim_start_matches(char::is_whitespace).to_string();
        }

        wrapped.push(remaining);
    }

    if wrapped.is_empty() {
        wrapped.push(String::new());
    }

    wrapped
}

fn thinking_teleprompter_lines(text: &str, collapsed: bool) -> (Vec<String>, bool) {
    const TELEPROMPTER_WRAP: usize = 112;
    const TELEPROMPTER_MAX_LINES: usize = 6;

    let lines = wrap_soft(text, TELEPROMPTER_WRAP);
    if !collapsed {
        return (lines, false);
    }

    let truncated = lines.len() > TELEPROMPTER_MAX_LINES;
    let displayed = if truncated {
        lines[lines.len() - TELEPROMPTER_MAX_LINES..].to_vec()
    } else {
        lines
    };

    (displayed, truncated)
}

fn rail_line(rail_color: Color, content: Vec<Span<'static>>) -> Line<'static> {
    let mut spans = Vec::with_capacity(content.len() + 1);
    spans.push(Span::styled(
        "┃ ",
        Style::default().fg(rail_color).add_modifier(Modifier::BOLD),
    ));
    spans.extend(content);
    Line::from(spans)
}

fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let w = width.min(area.width.saturating_sub(2).max(1));
    let h = height.min(area.height.saturating_sub(2).max(1));
    let x = area.x + area.width.saturating_sub(w) / 2;
    let y = area.y + area.height.saturating_sub(h) / 2;
    Rect {
        x,
        y,
        width: w,
        height: h,
    }
}

fn gradient_spans(text: &str, offset: usize) -> Vec<Span<'static>> {
    text.chars()
        .enumerate()
        .map(|(idx, ch)| {
            let color = LOADING_GRADIENT[(idx + offset) % LOADING_GRADIENT.len()];
            Span::styled(
                ch.to_string(),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            )
        })
        .collect()
}

fn count_downloaded_models() -> usize {
    let models_dir = NcConfig::models_dir();
    models()
        .iter()
        .filter(|model| !list_cached_quants(&models_dir, model).is_empty())
        .count()
}

fn render_brand_banner(
    model_label: &str,
    active_agent: &str,
    theme: UiTheme,
) -> Vec<Line<'static>> {
    let model = short_preview(&model_label.to_ascii_lowercase(), 24);
    let downloaded_models = count_downloaded_models();
    let model_label = if downloaded_models == 1 {
        "1 model baixado".to_string()
    } else {
        format!("{downloaded_models} modelos baixados")
    };

    let mut lines = Vec::new();
    lines.push(Line::from(vec![
        Span::styled(CAT_LOGO_LINES[0], Style::default().fg(theme.muted)),
        Span::raw("  "),
        Span::styled(
            "Nano Code",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw(" "),
        Span::styled(
            format!("v{NANO_CODE_VERSION} · "),
            Style::default().fg(theme.muted),
        ),
        Span::styled(model, Style::default().fg(theme.info_cyan)),
    ]));
    lines.push(Line::from(vec![
        Span::styled(CAT_LOGO_LINES[1], Style::default().fg(theme.muted)),
        Span::raw("  "),
        Span::styled(
            format!("{model_label} · agent {active_agent} · 0 MCP servers · 0 skills"),
            Style::default().fg(theme.muted),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled(CAT_LOGO_LINES[2], Style::default().fg(theme.muted)),
        Span::raw("  "),
        Span::styled("Type ", Style::default().fg(theme.fg)),
        Span::styled("/help", Style::default().fg(theme.info_blue)),
        Span::styled(" for more information", Style::default().fg(theme.fg)),
    ]));
    lines.push(Line::raw(""));
    lines
}

fn render_chat_lines(
    chat: &[ChatItem],
    tools_collapsed: bool,
    thinking_collapsed: bool,
    spin: usize,
    model_label: &str,
    active_agent: &str,
    theme: UiTheme,
) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    for item in chat {
        match item {
            ChatItem::Banner => lines.extend(render_brand_banner(model_label, active_agent, theme)),
            ChatItem::User(text) => {
                for line in text_lines_or_empty(text) {
                    let mut row = Vec::new();
                    row.push(Span::styled(
                        line.to_string(),
                        Style::default()
                            .fg(theme.accent)
                            .add_modifier(Modifier::BOLD),
                    ));
                    lines.push(rail_line(theme.accent, row));
                }
                lines.push(Line::raw(""));
            }
            ChatItem::Thinking { text, active } => {
                let icon = spinner_char(spin).to_string();
                let title = if *active {
                    format!("{} Thinking...", icon)
                } else {
                    "⁕ Thinking".to_string()
                };
                lines.push(Line::from(vec![Span::styled(
                    title,
                    Style::default()
                        .fg(theme.warning)
                        .add_modifier(Modifier::BOLD),
                )]));

                let (visible_lines, truncated) =
                    thinking_teleprompter_lines(text, thinking_collapsed);
                for line in visible_lines {
                    lines.push(rail_line(
                        theme.muted,
                        vec![Span::styled(line, Style::default().fg(theme.muted))],
                    ));
                }

                if thinking_collapsed {
                    let hint = if truncated {
                        "▸ teleprompter mode (Ctrl+O to expand full thinking)"
                    } else {
                        "▸ Ctrl+O to expand full thinking"
                    };
                    lines.push(rail_line(
                        theme.muted,
                        vec![Span::styled(hint, Style::default().fg(theme.muted))],
                    ));
                } else {
                    lines.push(rail_line(
                        theme.muted,
                        vec![Span::styled(
                            "▾ full thinking expanded (Ctrl+O to collapse)",
                            Style::default().fg(theme.muted),
                        )],
                    ));
                }

                lines.push(Line::raw(""));
            }
            ChatItem::Assistant(text) => {
                for line in text_lines_or_empty(text) {
                    lines.push(Line::from(vec![Span::raw(line.to_string())]));
                }
                lines.push(Line::raw(""));
            }
            ChatItem::Tool {
                summary,
                stream,
                detail,
                state,
                ..
            } => {
                let color = match state {
                    ToolState::Running => theme.warning,
                    ToolState::Ok => theme.success,
                    ToolState::Error => theme.danger,
                };
                lines.push(Line::from(vec![
                    Span::styled(
                        "● ",
                        Style::default().fg(color).add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(
                        summary.clone(),
                        Style::default()
                            .fg(theme.info_cyan)
                            .add_modifier(Modifier::BOLD),
                    ),
                ]));

                if let Some(s) = stream {
                    lines.push(rail_line(
                        theme.muted,
                        vec![Span::styled(
                            format!("⌊ {}", s),
                            Style::default().fg(theme.muted),
                        )],
                    ));
                }

                if let Some(d) = detail {
                    if tools_collapsed {
                        lines.push(rail_line(
                            theme.muted,
                            vec![Span::styled(
                                "▸ tool output hidden (Ctrl+O)",
                                Style::default().fg(theme.muted),
                            )],
                        ));
                    } else {
                        for detail_line in text_lines_or_empty(d) {
                            lines.push(rail_line(
                                theme.muted,
                                vec![Span::styled(
                                    detail_line.to_string(),
                                    Style::default().fg(theme.muted),
                                )],
                            ));
                        }
                    }
                }

                lines.push(Line::raw(""));
            }
            ChatItem::Error(text) => {
                lines.push(rail_line(
                    theme.danger,
                    vec![
                        Span::styled(
                            "error ",
                            Style::default()
                                .fg(theme.danger)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(text.clone(), Style::default().fg(theme.danger)),
                    ],
                ));
                lines.push(Line::raw(""));
            }
        }
    }

    lines
}

fn render_loading_line(app: &AppState, theme: UiTheme) -> Line<'static> {
    if !app.busy {
        return Line::raw("");
    }

    let elapsed_secs = app
        .busy_started_at
        .map(|start| start.elapsed().as_secs())
        .unwrap_or(0);

    let mut spans = vec![Span::styled(
        format!("{} ", spinner_char(app.spinner_idx)),
        Style::default().fg(LOADING_GRADIENT[app.spinner_idx % LOADING_GRADIENT.len()]),
    )];
    spans.extend(gradient_spans("Generating", app.spinner_idx));
    spans.push(Span::styled(
        "… ",
        Style::default().fg(LOADING_GRADIENT[(app.spinner_idx + 1) % LOADING_GRADIENT.len()]),
    ));
    spans.push(Span::styled(
        format!("({}s esc to interrupt)", elapsed_secs),
        Style::default().fg(theme.fg),
    ));
    Line::from(spans)
}

fn render_input_lines(app: &AppState, theme: UiTheme) -> Vec<Line<'static>> {
    let prefix = input_prompt_prefix(app.input_mode);

    if app.input.is_empty() {
        return vec![Line::from(vec![Span::styled(
            prefix,
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        )])];
    }

    let mut lines = Vec::new();
    for (idx, line) in text_lines_or_empty(&app.input).iter().enumerate() {
        let prompt = if idx == 0 { prefix } else { "  " };
        let prompt_style = if idx == 0 {
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(theme.muted)
        };
        lines.push(Line::from(vec![
            Span::styled(prompt, prompt_style),
            Span::raw(line.to_string()),
        ]));
    }
    lines
}

fn mb_to_gib_text(mb: u64) -> String {
    format!("{:.1} GB", mb as f64 / 1024.0)
}

fn telemetry_vendor_label(vendor: GpuVendor) -> &'static str {
    match vendor {
        GpuVendor::Nvidia => "NVIDIA",
        GpuVendor::Apple => "Apple",
        GpuVendor::Amd => "AMD",
        GpuVendor::Unknown => "GPU",
    }
}

fn telemetry_text(telemetry: &RuntimeTelemetry) -> String {
    let mut parts = Vec::new();

    match telemetry.mode {
        ComputeMode::Gpu => {
            let label = telemetry
                .gpu_vendor
                .map(telemetry_vendor_label)
                .unwrap_or("GPU");
            parts.push(label.to_string());

            if let Some(gpu_usage) = telemetry.gpu_usage_pct {
                parts.push(format!("GPU {:.0}%", gpu_usage));
            }
            if let Some(gpu_temp) = telemetry.gpu_temp_c {
                parts.push(format!("{}°C", gpu_temp.round() as i32));
            }
            if let (Some(used), Some(total)) = (telemetry.vram_used_mb, telemetry.vram_total_mb) {
                parts.push(format!(
                    "VRAM {}/{}",
                    mb_to_gib_text(used),
                    mb_to_gib_text(total)
                ));
            } else if let Some(total) = telemetry.vram_total_mb {
                parts.push(format!("VRAM {}", mb_to_gib_text(total)));
            }
            parts.push(format!("CPU {:.0}%", telemetry.cpu_usage_pct));
            if let Some(cpu_temp) = telemetry.cpu_temp_c {
                parts.push(format!("CPU {}°C", cpu_temp.round() as i32));
            }
            parts.push(format!(
                "RAM {}/{}",
                mb_to_gib_text(telemetry.ram_used_mb),
                mb_to_gib_text(telemetry.ram_total_mb)
            ));
        }
        ComputeMode::Cpu => {
            parts.push("CPU".to_string());
            parts.push(format!("{:.0}%", telemetry.cpu_usage_pct));
            if let Some(cpu_temp) = telemetry.cpu_temp_c {
                parts.push(format!("{}°C", cpu_temp.round() as i32));
            }
            parts.push(format!(
                "RAM {}/{}",
                mb_to_gib_text(telemetry.ram_used_mb),
                mb_to_gib_text(telemetry.ram_total_mb)
            ));
        }
    }

    parts.join(" · ")
}

fn format_download_size(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

fn build_progress_bar(downloaded: u64, total: u64, width: usize) -> String {
    let percent = if total > 0 {
        (downloaded as f64 / total as f64).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let filled = (percent * width as f64).round() as usize;
    let mut bar = String::with_capacity(width);
    for idx in 0..width {
        if idx < filled {
            bar.push('█');
        } else {
            bar.push('░');
        }
    }
    bar
}

fn render_download_progress_overlay(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    setup: &super::state::ModelSetupState,
    theme: UiTheme,
) {
    let Some(progress) = setup.download_progress.as_ref() else {
        return;
    };

    let popup = centered_rect(72, 7, area);
    frame.render_widget(Clear, popup);

    let percent = if progress.total > 0 {
        (progress.downloaded as f64 / progress.total as f64 * 100.0).round() as u32
    } else {
        0
    };
    let progress_bar = build_progress_bar(progress.downloaded, progress.total, 28);
    let speed_mb_s = progress.speed_bps as f64 / (1024.0 * 1024.0);

    let lines = vec![
        Line::from(vec![
            Span::styled("Baixando ", Style::default().fg(theme.muted)),
            Span::styled(
                progress.filename.clone(),
                Style::default()
                    .fg(theme.info_cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::raw(""),
        Line::from(vec![
            Span::styled("[", Style::default().fg(theme.border)),
            Span::styled(progress_bar, Style::default().fg(theme.accent)),
            Span::styled("] ", Style::default().fg(theme.border)),
            Span::styled(
                format!("{:>3}%", percent),
                Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![Span::styled(
            format!(
                "{} / {} · {:.1} MB/s · ~{}s restantes",
                format_download_size(progress.downloaded),
                format_download_size(progress.total),
                speed_mb_s,
                progress.eta_seconds
            ),
            Style::default().fg(theme.fg),
        )]),
        Line::raw(""),
        Line::from(vec![Span::styled(
            "Ctrl+C para cancelar (retoma depois do ponto parcial)",
            Style::default().fg(theme.muted),
        )]),
    ];

    let widget = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_type(BorderType::Rounded)
                .border_style(Style::default().fg(theme.border).bg(theme.bg)),
        )
        .style(Style::default().bg(theme.bg).fg(theme.fg))
        .alignment(Alignment::Left);
    frame.render_widget(widget, popup);
}

fn render_telemetry_line(app: &AppState, theme: UiTheme) -> Line<'static> {
    Line::from(vec![Span::styled(
        telemetry_text(&app.telemetry),
        Style::default().fg(theme.muted),
    )])
}

fn render_slash_suggestion_lines(app: &AppState, theme: UiTheme) -> Vec<Line<'static>> {
    if app.slash_suggestions.is_empty() {
        return vec![Line::raw("")];
    }

    let total = app.slash_suggestions.len();
    let visible = total.min(MAX_SLASH_SUGGESTIONS);
    let start = app
        .slash_selected
        .saturating_add(1)
        .saturating_sub(visible)
        .min(total.saturating_sub(visible));
    let end = (start + visible).min(total);

    let mut lines = Vec::with_capacity(visible);
    for (idx, cmd) in app.slash_suggestions[start..end].iter().enumerate() {
        let absolute_idx = start + idx;
        let selected = absolute_idx == app.slash_selected;
        let alias_style = if selected {
            Style::default()
                .fg(theme.fg)
                .add_modifier(Modifier::BOLD | Modifier::REVERSED)
        } else {
            Style::default().fg(theme.fg).add_modifier(Modifier::BOLD)
        };
        if app.slash_details_expanded {
            let desc_style = if selected {
                Style::default()
                    .fg(theme.fg)
                    .add_modifier(Modifier::ITALIC | Modifier::REVERSED)
            } else {
                Style::default().fg(theme.muted)
            };
            lines.push(Line::from(vec![
                Span::styled(cmd.alias.to_string(), alias_style),
                Span::raw("  "),
                Span::styled(cmd.description.to_string(), desc_style),
            ]));
        } else {
            lines.push(Line::from(vec![Span::styled(
                cmd.alias.to_string(),
                alias_style,
            )]));
        }
    }

    if !app.slash_details_expanded {
        lines.push(Line::from(vec![Span::styled(
            "Ctrl+O for command details",
            Style::default().fg(theme.muted),
        )]));
    }

    lines
}

fn render_welcome_screen(
    frame: &mut ratatui::Frame<'_>,
    app: &AppState,
    size: Rect,
    theme: UiTheme,
) {
    let base = Block::default().style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(base, size);

    let welcome = centered_rect(58, 3, size);
    let mut welcome_spans = vec![Span::styled("Welcome to ", Style::default().fg(theme.fg))];
    welcome_spans.extend(gradient_spans(WELCOME_HIGHLIGHT, app.spinner_idx));
    welcome_spans.push(Span::styled(
        " - Let's get you started!",
        Style::default().fg(theme.fg),
    ));

    let welcome_block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(theme.border).bg(theme.bg));
    let welcome_text = Paragraph::new(vec![Line::from(welcome_spans)])
        .alignment(Alignment::Center)
        .style(Style::default().bg(theme.bg).fg(theme.fg))
        .block(welcome_block);
    frame.render_widget(welcome_text, welcome);

    let hint = Rect {
        x: welcome.x,
        y: welcome.y.saturating_add(welcome.height).saturating_add(1),
        width: welcome.width,
        height: 1,
    };
    let hint_line = Paragraph::new(vec![Line::from(vec![Span::styled(
        "Press Enter ↵",
        Style::default()
            .fg(theme.muted)
            .add_modifier(Modifier::BOLD),
    )])])
    .alignment(Alignment::Center)
    .style(Style::default().bg(theme.bg));
    frame.render_widget(hint_line, hint);
}

fn render_model_setup_screen(
    frame: &mut ratatui::Frame<'_>,
    app: &AppState,
    size: Rect,
    theme: UiTheme,
) {
    let base = Block::default().style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(base, size);

    let Some(setup) = app.model_setup.as_ref() else {
        let empty = Paragraph::new("Setup state unavailable")
            .alignment(Alignment::Center)
            .style(Style::default().bg(theme.bg).fg(theme.danger));
        frame.render_widget(empty, size);
        return;
    };

    let header = centered_rect(58, 3, size);
    let title = Paragraph::new(vec![Line::from(vec![
        Span::styled(
            "One last thing",
            Style::default()
                .fg(theme.accent)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled("...", Style::default().fg(theme.fg)),
    ])])
    .alignment(Alignment::Center)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme.border).bg(theme.bg)),
    )
    .style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(title, header);

    // Keep content area strictly inside the terminal buffer. The header is
    // vertically centered, so we must compute remaining height from its Y offset.
    let content_y = header.y.saturating_add(header.height).saturating_add(1);
    let content_bottom = size.y.saturating_add(size.height);
    let content = Rect {
        x: size.x.saturating_add(2),
        y: content_y,
        width: size.width.saturating_sub(4),
        height: content_bottom.saturating_sub(content_y).saturating_sub(1),
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(2),
            Constraint::Length(1),
        ])
        .split(content);

    let selected_model = setup.models.get(setup.selected_model_idx);
    let mut info_lines = vec![Line::from(vec![
        Span::styled("Hardware: ", Style::default().fg(theme.muted)),
        Span::styled(
            setup.hardware_display.clone(),
            Style::default().fg(theme.fg),
        ),
    ])];
    if let Some(model) = selected_model {
        info_lines.push(Line::from(vec![
            Span::styled("Model: ", Style::default().fg(theme.muted)),
            Span::styled(
                model.model_display_name.clone(),
                Style::default().fg(theme.info_cyan),
            ),
            Span::raw(" · "),
            Span::styled(
                format!(
                    "{} · thinking:{} · vision:{}",
                    model.category_label,
                    if model.supports_thinking { "yes" } else { "no" },
                    if model.supports_vision { "yes" } else { "no" }
                ),
                Style::default().fg(theme.fg),
            ),
        ]));
        info_lines.push(Line::from(vec![
            Span::styled("Context: ", Style::default().fg(theme.muted)),
            Span::styled(
                format!(
                    "max {} · recommended {} (general) / {} (coding)",
                    model.max_context_tokens,
                    model.recommended_context_general,
                    model.recommended_context_coding
                ),
                Style::default().fg(theme.fg),
            ),
        ]));
    }
    let info = Paragraph::new(info_lines).style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(info, chunks[0]);

    let table_title = if setup.view == ModelSetupView::Models {
        "Select model".to_string()
    } else {
        format!("Select quantization · {}", setup.current_model_display_name)
    };
    let table_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.border).bg(theme.bg))
        .title(table_title);
    let table_inner = table_block.inner(chunks[1]);
    frame.render_widget(table_block, chunks[1]);

    let visible_rows = table_inner.height.saturating_sub(1) as usize;
    let total = setup.choices.len();
    let mut start = 0usize;
    if total > visible_rows && visible_rows > 0 {
        start = setup
            .selected_idx
            .saturating_sub(visible_rows.saturating_div(2))
            .min(total.saturating_sub(visible_rows));
    }
    let end = (start + visible_rows).min(total);

    let mut rows: Vec<Line<'static>> = Vec::new();
    match setup.view {
        ModelSetupView::Models => {
            let total_models = setup.models.len();
            let mut model_start = 0usize;
            if total_models > visible_rows && visible_rows > 0 {
                model_start = setup
                    .selected_model_idx
                    .saturating_sub(visible_rows.saturating_div(2))
                    .min(total_models.saturating_sub(visible_rows));
            }
            let model_end = (model_start + visible_rows).min(total_models);

            for (idx, model) in setup
                .models
                .iter()
                .enumerate()
                .take(model_end)
                .skip(model_start)
            {
                let selected = idx == setup.selected_model_idx;
                let cursor = if selected { "›" } else { " " };
                let name_style = if selected {
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD | Modifier::REVERSED)
                } else {
                    Style::default().fg(theme.fg)
                };
                let cache_desc = if model.cached_quants.is_empty() {
                    "cache: none".to_string()
                } else {
                    format!(
                        "cache: {} ({})",
                        model.cached_quants.len(),
                        short_preview(&model.cached_quants.join(", "), 30)
                    )
                };
                let active_desc = model
                    .active_quant
                    .as_ref()
                    .map(|q| format!("● {}", q))
                    .unwrap_or_else(|| " ".to_string());
                let recommended_desc = model
                    .recommended_quant
                    .as_ref()
                    .map(|q| format!("★ {}", q))
                    .unwrap_or_default();

                rows.push(Line::from(vec![
                    Span::styled(format!("{} ", cursor), Style::default().fg(theme.accent)),
                    Span::styled(format!("{:<24}", model.model_display_name), name_style),
                    Span::raw(" "),
                    Span::styled(
                        format!("{:<9}", model.category_label),
                        Style::default().fg(theme.info_blue),
                    ),
                    Span::raw(" "),
                    Span::styled(cache_desc, Style::default().fg(theme.fg)),
                    Span::raw(" "),
                    Span::styled(active_desc, Style::default().fg(theme.success)),
                    Span::raw(" "),
                    Span::styled(recommended_desc, Style::default().fg(theme.warning)),
                ]));
            }

            if rows.is_empty() {
                rows.push(Line::from("No models available"));
            }
        }
        ModelSetupView::Variants => {
            for (idx, choice) in setup.choices.iter().enumerate().take(end).skip(start) {
                let selected = idx == setup.selected_idx;
                let cursor = if selected { "›" } else { " " };
                let mut flags = Vec::new();
                if choice.active {
                    flags.push("●");
                }
                if choice.cached {
                    flags.push("✓");
                }
                if choice.recommended {
                    flags.push("★");
                }

                let name_style = if selected {
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD | Modifier::REVERSED)
                } else {
                    Style::default().fg(theme.fg)
                };

                rows.push(Line::from(vec![
                    Span::styled(format!("{} ", cursor), Style::default().fg(theme.accent)),
                    Span::styled(format!("{:<12}", choice.quant_name), name_style),
                    Span::raw(" "),
                    Span::styled(
                        format!("{:>8}", choice.size_human),
                        Style::default().fg(theme.info_blue),
                    ),
                    Span::raw(" "),
                    Span::styled(
                        format!("{:<10}", choice.quality_label),
                        Style::default().fg(theme.fg),
                    ),
                    Span::raw(" "),
                    Span::styled(flags.join(" "), Style::default().fg(theme.success)),
                ]));
            }
            if rows.is_empty() {
                rows.push(Line::from("No quantizations available"));
            }
        }
    }
    let rows_paragraph = Paragraph::new(rows).style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(rows_paragraph, table_inner);

    let status = setup
        .error_line
        .as_ref()
        .map(|e| {
            Line::from(vec![
                Span::styled(
                    "Error: ",
                    Style::default()
                        .fg(theme.danger)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(e.clone(), Style::default().fg(theme.danger)),
            ])
        })
        .unwrap_or_else(|| {
            Line::from(vec![Span::styled(
                setup.status_line.clone(),
                Style::default().fg(theme.muted),
            )])
        });

    let mut footer_lines = vec![status];
    if !setup.downloading {
        if let Some(progress) = &setup.progress_line {
            footer_lines.push(Line::from(vec![Span::styled(
                progress.clone(),
                Style::default().fg(theme.info_cyan),
            )]));
        } else {
            match setup.view {
                ModelSetupView::Models => {
                    if let Some(model) = selected_model {
                        let cached = if model.cached_quants.is_empty() {
                            "none".to_string()
                        } else {
                            model.cached_quants.join(", ")
                        };
                        footer_lines.push(Line::from(vec![Span::styled(
                            format!(
                                "Capabilities: thinking={} · vision={} · cached variants: {}",
                                if model.supports_thinking { "yes" } else { "no" },
                                if model.supports_vision { "yes" } else { "no" },
                                cached
                            ),
                            Style::default().fg(theme.muted),
                        )]));
                    }
                }
                ModelSetupView::Variants => {
                    if let Some(choice) = setup.choices.get(setup.selected_idx) {
                        if let Some(note) = &choice.notes {
                            footer_lines.push(Line::from(vec![Span::styled(
                                note.clone(),
                                Style::default().fg(theme.muted),
                            )]));
                        }
                    }
                }
            }
        }
    } else {
        footer_lines.push(Line::from(vec![Span::styled(
            "Download ativo. Aguarde ou pressione Ctrl+C para cancelar.",
            Style::default().fg(theme.warning),
        )]));
    }
    let footer = Paragraph::new(footer_lines).style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(footer, chunks[2]);

    let hint = if setup.downloading {
        "Download em andamento · seleção bloqueada · Ctrl+C cancela"
    } else {
        match setup.view {
            ModelSetupView::Models => {
                if app.model_setup_can_cancel {
                    "↑/↓ select model · Enter show variants · Esc back to chat"
                } else {
                    "↑/↓ select model · Enter show variants · Esc quit"
                }
            }
            ModelSetupView::Variants => {
                "↑/↓ select variant · Enter activate/download · Esc back to models"
            }
        }
    };
    let hint_line = Paragraph::new(hint)
        .alignment(Alignment::Center)
        .style(Style::default().bg(theme.bg).fg(theme.muted));
    frame.render_widget(hint_line, chunks[3]);

    if setup.downloading {
        render_download_progress_overlay(frame, size, setup, theme);
    }
}

fn render_config_screen(
    frame: &mut ratatui::Frame<'_>,
    app: &AppState,
    size: Rect,
    theme: UiTheme,
) {
    let base = Block::default().style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(base, size);

    let Some(config_screen) = app.config_screen.as_ref() else {
        let empty = Paragraph::new("Configuration state unavailable")
            .alignment(Alignment::Center)
            .style(Style::default().bg(theme.bg).fg(theme.danger));
        frame.render_widget(empty, size);
        return;
    };

    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(4),
            Constraint::Min(6),
            Constraint::Length(2),
            Constraint::Length(1),
        ])
        .split(size);

    let title_suffix = if config_screen.is_dirty() { " *" } else { "" };
    let title = Paragraph::new(vec![Line::from(vec![Span::styled(
        format!("Configuration{}", title_suffix),
        Style::default()
            .fg(theme.accent)
            .add_modifier(Modifier::BOLD),
    )])])
    .alignment(Alignment::Center)
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme.border).bg(theme.bg)),
    )
    .style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(title, root[0]);

    let runtime = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Model: ", Style::default().fg(theme.muted)),
            Span::styled(
                config_screen.model_label.clone(),
                Style::default().fg(theme.info_cyan),
            ),
        ]),
        Line::from(vec![
            Span::styled("Runtime: ", Style::default().fg(theme.muted)),
            Span::styled(
                format!(
                    "context {} · max output {}",
                    config_screen.runtime_context_tokens, config_screen.runtime_max_tokens
                ),
                Style::default().fg(theme.fg),
            ),
            Span::raw(" · "),
            Span::styled(
                config_screen.hardware_display.clone(),
                Style::default().fg(theme.fg),
            ),
        ]),
    ])
    .style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(runtime, root[1]);

    let paths = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Config: ", Style::default().fg(theme.muted)),
            Span::styled(
                config_screen.config_path.clone(),
                Style::default().fg(theme.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("Models: ", Style::default().fg(theme.muted)),
            Span::styled(
                config_screen.models_path.clone(),
                Style::default().fg(theme.fg),
            ),
        ]),
        Line::from(vec![
            Span::styled("Sessions: ", Style::default().fg(theme.muted)),
            Span::styled(
                config_screen.sessions_path.clone(),
                Style::default().fg(theme.fg),
            ),
        ]),
    ])
    .wrap(Wrap { trim: false })
    .style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(paths, root[2]);

    let settings_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(theme.border).bg(theme.bg))
        .title("Settings");
    let settings_inner = settings_block.inner(root[3]);
    frame.render_widget(settings_block, root[3]);

    if settings_inner.height > 0 {
        let total = config_screen.fields.len();
        let visible_rows = settings_inner.height as usize;
        let mut start = 0usize;
        if total > visible_rows && visible_rows > 0 {
            start = config_screen
                .selected_idx
                .saturating_sub(visible_rows.saturating_div(2))
                .min(total.saturating_sub(visible_rows));
        }
        let end = (start + visible_rows).min(total);

        let mut rows = Vec::new();
        for (idx, field) in config_screen
            .fields
            .iter()
            .enumerate()
            .take(end)
            .skip(start)
        {
            let selected = idx == config_screen.selected_idx;
            let cursor = if selected { ">" } else { " " };
            let dirty = if field.is_dirty() { "*" } else { " " };
            let value_style = if selected {
                Style::default()
                    .fg(theme.fg)
                    .add_modifier(Modifier::REVERSED | Modifier::BOLD)
            } else {
                Style::default().fg(theme.info_blue)
            };
            rows.push(Line::from(vec![
                Span::styled(cursor, Style::default().fg(theme.accent)),
                Span::raw(" "),
                Span::styled(dirty, Style::default().fg(theme.warning)),
                Span::raw(" "),
                Span::styled(
                    format!("{:<24}", field.label),
                    Style::default().fg(theme.fg),
                ),
                Span::raw(" "),
                Span::styled(field.value.clone(), value_style),
            ]));
        }

        if rows.is_empty() {
            rows.push(Line::from("No settings available"));
        }
        let settings = Paragraph::new(rows).style(Style::default().bg(theme.bg).fg(theme.fg));
        frame.render_widget(settings, settings_inner);
    }

    let status = if let Some(err) = &config_screen.error_line {
        Line::from(vec![
            Span::styled(
                "Error: ",
                Style::default()
                    .fg(theme.danger)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(err.clone(), Style::default().fg(theme.danger)),
        ])
    } else {
        Line::from(vec![Span::styled(
            config_screen.status_line.clone(),
            Style::default().fg(theme.muted),
        )])
    };
    let status_widget =
        Paragraph::new(vec![status]).style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(status_widget, root[4]);

    let hint = if config_screen.is_dirty() {
        "Up/Down select · Enter/Space cycle · Esc save and close"
    } else {
        "Up/Down select · Enter/Space cycle · Esc close"
    };
    let hint_line = Paragraph::new(hint)
        .alignment(Alignment::Center)
        .style(Style::default().bg(theme.bg).fg(theme.muted));
    frame.render_widget(hint_line, root[5]);
}

fn render_chat_screen(frame: &mut ratatui::Frame<'_>, app: &AppState, size: Rect, theme: UiTheme) {
    let loading_h = if app.busy { 1 } else { 0 };
    let input_lines = text_lines_or_empty(&app.input).len() as u16;
    let input_h = input_lines.clamp(1, 3) + 2;
    let show_slash_suggestions =
        app.input_mode == InputMode::Slash && !app.slash_suggestions.is_empty();
    let slash_lines = if show_slash_suggestions {
        app.slash_suggestions.len().min(MAX_SLASH_SUGGESTIONS) as u16
            + if app.slash_details_expanded { 0 } else { 1 }
    } else {
        0
    };
    let slash_h = if show_slash_suggestions {
        slash_lines + 2
    } else {
        0
    };

    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),
            Constraint::Length(loading_h),
            Constraint::Length(slash_h),
            Constraint::Length(input_h),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(size);

    let mut chat_lines = render_chat_lines(
        &app.chat,
        app.tools_collapsed,
        app.thinking_collapsed,
        app.spinner_idx,
        &app.model_label,
        app.active_agent.as_str(),
        theme,
    );
    let chat_h = root[0].height as usize;
    if chat_h > chat_lines.len() {
        let pad = chat_h - chat_lines.len();
        let mut padded = Vec::with_capacity(chat_h);
        padded.extend((0..pad).map(|_| Line::raw("")));
        padded.append(&mut chat_lines);
        chat_lines = padded;
    }

    let base = Block::default().style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(base, size);

    let max_scroll = chat_lines.len().saturating_sub(chat_h) as u16;
    // chat_scroll is stored as "distance from bottom", so 0 means follow latest messages.
    let chat_scroll = max_scroll.saturating_sub(app.chat_scroll);

    let chat = Paragraph::new(chat_lines)
        .wrap(Wrap { trim: false })
        .scroll((chat_scroll, 0))
        .style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(chat, root[0]);

    if loading_h > 0 {
        let loading = Paragraph::new(vec![render_loading_line(app, theme)])
            .style(Style::default().bg(theme.bg).fg(theme.fg));
        frame.render_widget(loading, root[1]);
    }

    if slash_h > 0 {
        let suggestions_box = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.border).bg(theme.bg));
        let suggestions_area = suggestions_box.inner(root[2]);
        frame.render_widget(suggestions_box, root[2]);
        let suggestions = Paragraph::new(render_slash_suggestion_lines(app, theme))
            .wrap(Wrap { trim: false })
            .style(Style::default().bg(theme.bg).fg(theme.fg));
        frame.render_widget(suggestions, suggestions_area);
    }

    let input_box = Block::default()
        .borders(Borders::ALL)
        .title("default")
        .title_alignment(Alignment::Right)
        .border_style(Style::default().fg(theme.border).bg(theme.bg));
    let input_area = input_box.inner(root[3]);
    frame.render_widget(input_box, root[3]);
    let input = Paragraph::new(render_input_lines(app, theme))
        .wrap(Wrap { trim: false })
        .style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(input, input_area);

    let telemetry = Paragraph::new(vec![render_telemetry_line(app, theme)])
        .style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(telemetry, root[4]);

    let cwd = std::env::current_dir()
        .ok()
        .and_then(|p| p.to_str().map(|s| s.to_string()))
        .unwrap_or_else(|| ".".to_string());
    let max_ctx = app.max_context_tokens.max(1);
    let pct = ((app.stats.context_tokens as f64 / max_ctx as f64) * 100.0)
        .clamp(0.0, 999.0)
        .round() as u32;
    let token_text = format!("{}% of {}k tokens", pct, max_ctx / 1000);
    let footer_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Min(10),
            Constraint::Length((token_text.chars().count() + 1) as u16),
        ])
        .split(root[5]);
    let left = Paragraph::new(short_preview(&cwd, 90))
        .style(Style::default().bg(theme.bg).fg(theme.muted));
    frame.render_widget(left, footer_cols[0]);
    let right = Paragraph::new(token_text)
        .alignment(Alignment::Right)
        .style(Style::default().bg(theme.bg).fg(theme.muted));
    frame.render_widget(right, footer_cols[1]);

    if let Some(approval) = app.pending_approval.as_ref() {
        let popup = centered_rect(92, 13, size);
        frame.render_widget(Clear, popup);

        let option_line = |label: &str, selected: bool| -> Line<'static> {
            let marker = if selected { "●" } else { "○" };
            let style = if selected {
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme.fg)
            };
            Line::from(vec![Span::styled(format!("{} {}", marker, label), style)])
        };

        let mut lines = Vec::new();
        lines.push(Line::from(vec![Span::styled(
            "Permission Required",
            Style::default()
                .fg(theme.warning)
                .add_modifier(Modifier::BOLD),
        )]));
        lines.push(Line::from(vec![
            Span::styled("Tool: ", Style::default().fg(theme.muted)),
            Span::styled(
                approval.summary.clone(),
                Style::default()
                    .fg(theme.info_cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::from(vec![
            Span::styled("Call ID: ", Style::default().fg(theme.muted)),
            Span::styled(approval.call_id.clone(), Style::default().fg(theme.fg)),
        ]));
        lines.push(Line::raw(""));

        for line in wrap_soft(&approval.arguments, 84).into_iter().take(3) {
            lines.push(Line::from(vec![Span::styled(
                line,
                Style::default().fg(theme.muted),
            )]));
        }

        lines.push(Line::raw(""));
        lines.push(option_line(
            "Allow once",
            approval.selected_option == ApprovalOption::ApproveOnce,
        ));
        lines.push(option_line(
            "Allow always for this tool (session)",
            approval.selected_option == ApprovalOption::ApproveAlwaysToolSession,
        ));
        lines.push(option_line(
            "Deny",
            approval.selected_option == ApprovalOption::Deny,
        ));
        lines.push(Line::from(vec![Span::styled(
            "1/Y=once · 2/A=always tool-session · 3/N/Esc=deny · ←/→ or ↑/↓ select · Enter confirm",
            Style::default().fg(theme.muted),
        )]));

        let widget = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_type(BorderType::Rounded)
                    .border_style(Style::default().fg(theme.border).bg(theme.bg)),
            )
            .style(Style::default().bg(theme.bg).fg(theme.fg))
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: false });
        frame.render_widget(widget, popup);
    }
}

pub fn draw_ui(terminal: &mut Terminal<CrosstermBackend<Stdout>>, app: &AppState) -> Result<()> {
    terminal.draw(|f| {
        let theme = UI_THEME;
        let size = f.area();
        match app.screen {
            UiScreen::Welcome => render_welcome_screen(f, app, size, theme),
            UiScreen::ModelSetup => render_model_setup_screen(f, app, size, theme),
            UiScreen::Config => render_config_screen(f, app, size, theme),
            UiScreen::Chat => render_chat_screen(f, app, size, theme),
        }
    })?;
    Ok(())
}
