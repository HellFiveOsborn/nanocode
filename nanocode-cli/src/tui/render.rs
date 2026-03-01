use anyhow::Result;
use nanocode_hf::{ComputeMode, GpuVendor, RuntimeTelemetry};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders, Paragraph, Wrap};
use ratatui::Terminal;
use std::io::Stdout;

use super::commands::MAX_SLASH_SUGGESTIONS;
use super::state::{AppState, ChatItem, InputMode, ToolState, UiScreen};

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
const CAT_LOGO_LINES: [&str; 3] = ["  ⡠⣒⠄  ⡔⢄⠔⡄", " ⢸⠸⣀⡔⢉⠱⣃⡢⣂⡣", "  ⠉⠒⠣⠤⠵⠤⠬⠮⠆"];

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

fn render_brand_banner(model_label: &str, theme: UiTheme) -> Vec<Line<'static>> {
    let model = short_preview(&model_label.to_ascii_lowercase(), 24);

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
        Span::styled("v2.3.0 · ", Style::default().fg(theme.muted)),
        Span::styled(model, Style::default().fg(theme.info_cyan)),
    ]));
    lines.push(Line::from(vec![
        Span::styled(CAT_LOGO_LINES[1], Style::default().fg(theme.muted)),
        Span::raw("  "),
        Span::styled(
            "3 models · 0 MCP servers · 0 skills",
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
    theme: UiTheme,
) -> Vec<Line<'static>> {
    let mut lines: Vec<Line<'static>> = Vec::new();

    for item in chat {
        match item {
            ChatItem::Banner => lines.extend(render_brand_banner(model_label, theme)),
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

fn render_chat_screen(
    frame: &mut ratatui::Frame<'_>,
    app: &AppState,
    size: Rect,
    theme: UiTheme,
) {
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
    let pct = ((app.stats.tokens_used as f64 / max_ctx as f64) * 100.0)
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
}

pub fn draw_ui(terminal: &mut Terminal<CrosstermBackend<Stdout>>, app: &AppState) -> Result<()> {
    terminal.draw(|f| {
        let theme = UI_THEME;
        let size = f.area();
        match app.screen {
            UiScreen::Welcome => render_welcome_screen(f, app, size, theme),
            UiScreen::Chat => render_chat_screen(f, app, size, theme),
        }
    })?;
    Ok(())
}
