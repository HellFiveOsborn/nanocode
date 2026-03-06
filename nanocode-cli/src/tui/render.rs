use anyhow::Result;
use nanocode_core::agents::BuiltinAgent;
use nanocode_core::NcConfig;
use nanocode_hf::{list_cached_quants, models};
use nanocode_hf::{ComputeMode, GpuVendor, RuntimeTelemetry};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, BorderType, Borders, Clear, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState,
    Wrap,
};
use ratatui::Terminal;
use std::io::Stdout;
use std::path::Path;
use std::sync::OnceLock;
use syntect::easy::HighlightLines;
use syntect::highlighting::{Theme, ThemeSet};
use syntect::parsing::{SyntaxReference, SyntaxSet};

use super::commands::MAX_SLASH_SUGGESTIONS;
use super::state::{
    AppState, ApprovalOption, ChatItem, InputMode, ModelSetupView, PlanReviewOption, ToolState,
    UiScreen,
};

const NANO_CODE_VERSION: &str = env!("CARGO_PKG_VERSION");
const SPINNER_FRAMES: [&str; 3] = ["∗", "⁕", "⋇"];
const OPENCODE_SPINNER_WIDTH: usize = 8;
const OPENCODE_SPINNER_TRAIL: usize = 6;
const OPENCODE_SPINNER_HOLD_START: usize = 30;
const OPENCODE_SPINNER_HOLD_END: usize = 9;
const OPENCODE_SPINNER_TOTAL_FRAMES: usize = OPENCODE_SPINNER_WIDTH
    + OPENCODE_SPINNER_HOLD_END
    + (OPENCODE_SPINNER_WIDTH - 1)
    + OPENCODE_SPINNER_HOLD_START;
const OPENCODE_SPINNER_COLORS: [Color; OPENCODE_SPINNER_TRAIL] = [
    Color::Rgb(95, 201, 255),
    Color::Rgb(74, 176, 255),
    Color::Rgb(58, 150, 247),
    Color::Rgb(46, 122, 226),
    Color::Rgb(33, 93, 199),
    Color::Rgb(22, 66, 168),
];
const LOADING_GRADIENT: [Color; 5] = [
    Color::Rgb(126, 229, 255),
    Color::Rgb(93, 205, 255),
    Color::Rgb(67, 178, 255),
    Color::Rgb(44, 148, 245),
    Color::Rgb(25, 116, 221),
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
    accent: Color::Rgb(67, 178, 255),
    muted: ANSI_BRIGHT_BLACK,
    success: Color::Green,
    warning: Color::Yellow,
    danger: Color::Red,
    info_blue: Color::Blue,
    info_cyan: Color::Cyan,
};

#[derive(Clone, Copy)]
struct OpencodeSpinnerState {
    active_position: usize,
    is_holding: bool,
    hold_progress: usize,
    is_moving_forward: bool,
}

struct HighlightAssets {
    ps: SyntaxSet,
    ts: ThemeSet,
}

fn highlight_assets() -> &'static HighlightAssets {
    static ASSETS: OnceLock<HighlightAssets> = OnceLock::new();
    ASSETS.get_or_init(|| HighlightAssets {
        ps: SyntaxSet::load_defaults_newlines(),
        ts: ThemeSet::load_defaults(),
    })
}

fn select_syntect_theme(theme_set: &ThemeSet) -> Option<&Theme> {
    theme_set
        .themes
        .get("base16-ocean.dark")
        .or_else(|| theme_set.themes.get("Monokai Extended"))
        .or_else(|| theme_set.themes.get("Monokai Extended Bright"))
        .or_else(|| theme_set.themes.get("Solarized (dark)"))
        .or_else(|| theme_set.themes.get("Monokai"))
        .or_else(|| theme_set.themes.values().next())
}

fn syntax_for_path<'a>(ps: &'a SyntaxSet, path: Option<&str>) -> &'a SyntaxReference {
    let extension = path
        .and_then(|value| Path::new(value).extension())
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());
    extension
        .as_deref()
        .and_then(|ext| ps.find_syntax_by_extension(ext))
        .unwrap_or_else(|| ps.find_syntax_plain_text())
}

fn syntect_color_to_ratatui(color: syntect::highlighting::Color) -> Color {
    Color::Rgb(color.r, color.g, color.b)
}

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

fn opencode_spinner_state(frame: usize) -> OpencodeSpinnerState {
    let frame_idx = frame % OPENCODE_SPINNER_TOTAL_FRAMES;
    let forward_frames = OPENCODE_SPINNER_WIDTH;
    let hold_end_frames = OPENCODE_SPINNER_HOLD_END;
    let backward_frames = OPENCODE_SPINNER_WIDTH - 1;

    if frame_idx < forward_frames {
        OpencodeSpinnerState {
            active_position: frame_idx,
            is_holding: false,
            hold_progress: 0,
            is_moving_forward: true,
        }
    } else if frame_idx < forward_frames + hold_end_frames {
        OpencodeSpinnerState {
            active_position: OPENCODE_SPINNER_WIDTH - 1,
            is_holding: true,
            hold_progress: frame_idx - forward_frames,
            is_moving_forward: true,
        }
    } else if frame_idx < forward_frames + hold_end_frames + backward_frames {
        let backward_index = frame_idx - forward_frames - hold_end_frames;
        OpencodeSpinnerState {
            active_position: OPENCODE_SPINNER_WIDTH - 2 - backward_index,
            is_holding: false,
            hold_progress: 0,
            is_moving_forward: false,
        }
    } else {
        OpencodeSpinnerState {
            active_position: 0,
            is_holding: true,
            hold_progress: frame_idx - forward_frames - hold_end_frames - backward_frames,
            is_moving_forward: false,
        }
    }
}

fn opencode_spinner_color_index(state: OpencodeSpinnerState, char_index: usize) -> Option<usize> {
    let directional_distance = if state.is_moving_forward {
        state.active_position as isize - char_index as isize
    } else {
        char_index as isize - state.active_position as isize
    };

    let color_index = if state.is_holding {
        directional_distance + state.hold_progress as isize
    } else if directional_distance > 0 && directional_distance < OPENCODE_SPINNER_TRAIL as isize {
        directional_distance
    } else if directional_distance == 0 {
        0
    } else {
        -1
    };

    if color_index >= 0 && (color_index as usize) < OPENCODE_SPINNER_TRAIL {
        Some(color_index as usize)
    } else {
        None
    }
}

fn render_opencode_spinner(frame: usize, theme: UiTheme) -> Vec<Span<'static>> {
    let state = opencode_spinner_state(frame);
    (0..OPENCODE_SPINNER_WIDTH)
        .map(
            |char_index| match opencode_spinner_color_index(state, char_index) {
                Some(color_index) => Span::styled(
                    "■",
                    Style::default()
                        .fg(OPENCODE_SPINNER_COLORS[color_index])
                        .add_modifier(Modifier::BOLD),
                ),
                None => Span::styled("⬝", Style::default().fg(theme.muted)),
            },
        )
        .collect()
}

fn format_elapsed_clock(elapsed_secs: u64) -> String {
    if elapsed_secs < 60 {
        format!("{}s", elapsed_secs)
    } else {
        let minutes = elapsed_secs / 60;
        let seconds = elapsed_secs % 60;
        format!("{}m {}s", minutes, seconds)
    }
}

fn input_prompt_prefix(mode: InputMode) -> &'static str {
    match mode {
        InputMode::Default => "❯ ",
        InputMode::Slash => "/ ",
    }
}

fn agent_mode_label(agent: BuiltinAgent, yolo: bool) -> String {
    let base = match agent {
        BuiltinAgent::Default => "Padrão",
        BuiltinAgent::Plan => "Plano",
        BuiltinAgent::Build => "Implementação",
        BuiltinAgent::Explore => "Explorar",
    };
    if yolo {
        format!("{} (YOLO)", base)
    } else {
        base.to_string()
    }
}

fn agent_mode_color(agent: BuiltinAgent, yolo: bool, theme: UiTheme) -> Color {
    if yolo {
        return theme.danger;
    }
    match agent {
        BuiltinAgent::Default => theme.warning,
        BuiltinAgent::Plan => theme.info_blue,
        BuiltinAgent::Build => theme.success,
        BuiltinAgent::Explore => theme.info_cyan,
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

fn tool_detail_line(_first: bool, theme: UiTheme, content: Vec<Span<'static>>) -> Line<'static> {
    let mut spans = Vec::with_capacity(content.len() + 1);
    spans.push(Span::styled("  ⎿  ", Style::default().fg(theme.muted)));
    spans.extend(content);
    Line::from(spans)
}

fn explore_detail_line(theme: UiTheme, content: Vec<Span<'static>>) -> Line<'static> {
    let mut spans = Vec::with_capacity(content.len() + 1);
    spans.push(Span::styled("  ⎿ ", Style::default().fg(theme.muted)));
    spans.extend(content);
    Line::from(spans)
}

const TOOL_OUTPUT_PREVIEW_LINES: usize = 5;

fn render_tool_output_lines(
    output: &str,
    theme: UiTheme,
    collapsed: bool,
) -> Vec<Line<'static>> {
    let raw_lines = text_lines_or_empty(output);
    let total = raw_lines.len();
    let limit = if collapsed {
        TOOL_OUTPUT_PREVIEW_LINES
    } else {
        total
    };
    let mut lines = Vec::new();
    for (idx, raw) in raw_lines.iter().take(limit).enumerate() {
        let style = if raw.contains("[stderr]") {
            Style::default().fg(theme.warning)
        } else {
            Style::default().fg(theme.fg)
        };
        lines.push(tool_detail_line(
            idx == 0,
            theme,
            vec![Span::styled(raw.to_string(), style)],
        ));
    }
    if collapsed && total > TOOL_OUTPUT_PREVIEW_LINES {
        let remaining = total - TOOL_OUTPUT_PREVIEW_LINES;
        lines.push(tool_detail_line(
            false,
            theme,
            vec![Span::styled(
                format!("… +{} lines (ctrl+o to expand)", remaining),
                Style::default().fg(theme.muted),
            )],
        ));
    }
    lines
}

fn render_highlighted_code_lines(
    section_title: &str,
    path: &str,
    code: &str,
    theme: UiTheme,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    lines.push(tool_detail_line(
        true,
        theme,
        vec![
            Span::styled("▸ ", Style::default().fg(theme.info_blue)),
            Span::styled(
                section_title.to_string(),
                Style::default()
                    .fg(theme.info_blue)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(": ", Style::default().fg(theme.muted)),
            Span::styled(path.to_string(), Style::default().fg(theme.muted)),
        ],
    ));

    let assets = highlight_assets();
    let syntax = syntax_for_path(&assets.ps, Some(path));
    let Some(theme_ref) = select_syntect_theme(&assets.ts) else {
        for (idx, raw) in text_lines_or_empty(code).iter().enumerate() {
            lines.push(tool_detail_line(
                false,
                theme,
                vec![
                    Span::styled(format!("{:>4} ", idx + 1), Style::default().fg(theme.muted)),
                    Span::styled(raw.to_string(), Style::default().fg(theme.fg)),
                ],
            ));
        }
        return lines;
    };
    let mut highlighter = HighlightLines::new(syntax, theme_ref);

    for (idx, raw) in text_lines_or_empty(code).iter().enumerate() {
        let mut spans = Vec::new();
        spans.push(Span::styled(
            format!("{:>4} ", idx + 1),
            Style::default().fg(theme.muted),
        ));
        match highlighter.highlight_line(raw, &assets.ps) {
            Ok(ranges) => {
                for (style, token) in ranges {
                    spans.push(Span::styled(
                        token.to_string(),
                        Style::default().fg(syntect_color_to_ratatui(style.foreground)),
                    ));
                }
            }
            Err(_) => {
                spans.push(Span::styled(raw.to_string(), Style::default().fg(theme.fg)));
            }
        }
        lines.push(tool_detail_line(false, theme, spans));
    }

    lines
}

#[derive(Clone, Copy)]
enum DiffLineKind {
    Added,
    Removed,
    Context,
    Meta,
}

/// Parse a unified diff line.
///
/// Format: `{num:>6} {sign}{content}` where sign is `-`, `+`, or ` ` (space).
/// Returns (kind, line_number_str, sign_str, content_str).
fn parse_diff_line(raw: &str) -> (DiffLineKind, &str, &str, &str) {
    if raw.trim() == "..." {
        return (DiffLineKind::Meta, "", "", raw);
    }
    // Unified format: byte 7 is the sign
    if raw.len() >= 8 {
        let num = &raw[..6];
        match raw.as_bytes()[7] {
            b'-' => {
                let content = if raw.len() > 8 { &raw[8..] } else { "" };
                return (DiffLineKind::Removed, num, "-", content);
            }
            b'+' => {
                let content = if raw.len() > 8 { &raw[8..] } else { "" };
                return (DiffLineKind::Added, num, "+", content);
            }
            b' ' => {
                let content = if raw.len() > 8 { &raw[8..] } else { "" };
                return (DiffLineKind::Context, num, " ", content);
            }
            _ => {}
        }
    }
    // Fallback for old format or short lines
    if raw.starts_with("@@") {
        (DiffLineKind::Meta, "", "", raw)
    } else if let Some(content) = raw.strip_prefix('+') {
        (DiffLineKind::Added, "", "+", content)
    } else if let Some(content) = raw.strip_prefix('-') {
        (DiffLineKind::Removed, "", "-", content)
    } else {
        (DiffLineKind::Context, "", " ", raw)
    }
}

fn render_diff_lines(diff: &str, code_path: Option<&str>, theme: UiTheme) -> Vec<Line<'static>> {
    let mut lines = Vec::new();

    let diff_added_bg = Color::Rgb(24, 56, 45);
    let diff_removed_bg = Color::Rgb(66, 37, 47);
    let diff_context_bg = Color::Rgb(36, 42, 52);
    let diff_meta_bg = Color::Rgb(31, 36, 46);
    let diff_added_sign = Color::Rgb(127, 231, 179);
    let diff_removed_sign = Color::Rgb(238, 156, 180);
    let diff_context_fg = Color::Rgb(194, 201, 214);
    let diff_meta_fg = Color::Rgb(150, 158, 174);

    let assets = highlight_assets();
    let syntax = syntax_for_path(&assets.ps, code_path);
    let Some(theme_ref) = select_syntect_theme(&assets.ts) else {
        for raw in text_lines_or_empty(diff) {
            let (kind, num, sign, content) = parse_diff_line(raw);
            let (fg, bg) = match kind {
                DiffLineKind::Added => (diff_added_sign, Some(diff_added_bg)),
                DiffLineKind::Removed => (diff_removed_sign, Some(diff_removed_bg)),
                DiffLineKind::Context => (diff_context_fg, Some(diff_context_bg)),
                DiffLineKind::Meta => (diff_meta_fg, Some(diff_meta_bg)),
            };
            let mut spans = Vec::new();
            if !num.is_empty() {
                spans.push(Span::styled(
                    format!("{} ", num),
                    Style::default().fg(theme.muted).bg(bg.unwrap_or(theme.bg)),
                ));
            }
            if !sign.is_empty() && sign != " " {
                spans.push(Span::styled(
                    sign.to_string(),
                    Style::default()
                        .fg(fg)
                        .bg(bg.unwrap_or(theme.bg))
                        .add_modifier(Modifier::BOLD),
                ));
            } else if !num.is_empty() {
                spans.push(Span::styled(
                    " ".to_string(),
                    Style::default().bg(bg.unwrap_or(theme.bg)),
                ));
            }
            spans.push(Span::styled(
                content.to_string(),
                Style::default().fg(fg).bg(bg.unwrap_or(theme.bg)),
            ));
            lines.push(tool_detail_line(false, theme, spans));
        }
        return lines;
    };
    let mut highlighter = HighlightLines::new(syntax, theme_ref);

    for raw in text_lines_or_empty(diff) {
        let (kind, num, sign, content) = parse_diff_line(raw);

        if matches!(kind, DiffLineKind::Meta) {
            lines.push(tool_detail_line(
                false,
                theme,
                vec![Span::styled(
                    raw.to_string(),
                    Style::default().fg(diff_meta_fg).bg(diff_meta_bg),
                )],
            ));
            continue;
        }

        let line_bg = match kind {
            DiffLineKind::Added => diff_added_bg,
            DiffLineKind::Removed => diff_removed_bg,
            DiffLineKind::Context | DiffLineKind::Meta => diff_context_bg,
        };
        let sign_fg = match kind {
            DiffLineKind::Added => diff_added_sign,
            DiffLineKind::Removed => diff_removed_sign,
            DiffLineKind::Context | DiffLineKind::Meta => theme.muted,
        };

        let mut spans = Vec::new();
        if !num.is_empty() {
            spans.push(Span::styled(
                format!("{} ", num),
                Style::default().fg(theme.muted).bg(line_bg),
            ));
        }
        if sign != " " {
            spans.push(Span::styled(
                sign.to_string(),
                Style::default()
                    .fg(sign_fg)
                    .bg(line_bg)
                    .add_modifier(Modifier::BOLD),
            ));
        } else {
            spans.push(Span::styled(" ".to_string(), Style::default().bg(line_bg)));
        }

        match highlighter.highlight_line(content, &assets.ps) {
            Ok(ranges) => {
                if ranges.is_empty() {
                    spans.push(Span::styled(" ", Style::default().bg(line_bg)));
                } else {
                    for (style, token) in ranges {
                        spans.push(Span::styled(
                            token.to_string(),
                            Style::default()
                                .fg(syntect_color_to_ratatui(style.foreground))
                                .bg(line_bg),
                        ));
                    }
                }
            }
            Err(_) => {
                spans.push(Span::styled(
                    content.to_string(),
                    Style::default().fg(theme.fg).bg(line_bg),
                ));
            }
        }
        lines.push(tool_detail_line(false, theme, spans));
    }
    lines
}

fn split_tool_summary(summary: &str) -> (&str, &str) {
    if let Some(open_idx) = summary.find('(') {
        (&summary[..open_idx], &summary[open_idx..])
    } else {
        (summary, "")
    }
}

fn parse_markdown_heading(line: &str) -> Option<(usize, &str)> {
    let trimmed = line.trim_start();
    let level = trimmed.chars().take_while(|ch| *ch == '#').count();
    if (1..=6).contains(&level) {
        let rest = &trimmed[level..];
        if let Some(title) = rest.strip_prefix(' ') {
            return Some((level, title.trim()));
        }
    }
    None
}

fn parse_markdown_list_item(line: &str) -> Option<(usize, String, String)> {
    let indent = line.chars().take_while(|ch| ch.is_whitespace()).count() / 2;
    let trimmed = line.trim_start();

    if let Some(content) = trimmed.strip_prefix("- [ ] ") {
        return Some((indent, "☐".to_string(), content.to_string()));
    }
    if let Some(content) = trimmed.strip_prefix("- [x] ") {
        return Some((indent, "☑".to_string(), content.to_string()));
    }
    if let Some(content) = trimmed.strip_prefix("- [X] ") {
        return Some((indent, "☑".to_string(), content.to_string()));
    }
    if let Some(content) = trimmed
        .strip_prefix("- ")
        .or_else(|| trimmed.strip_prefix("* "))
        .or_else(|| trimmed.strip_prefix("+ "))
    {
        return Some((indent, "•".to_string(), content.to_string()));
    }

    let digit_bytes = trimmed
        .char_indices()
        .take_while(|(_, ch)| ch.is_ascii_digit())
        .map(|(idx, ch)| idx + ch.len_utf8())
        .last()
        .unwrap_or(0);
    if digit_bytes > 0 {
        let (digits, rest) = trimmed.split_at(digit_bytes);
        if let Some(content) = rest.strip_prefix(". ") {
            return Some((indent, format!("{digits}."), content.to_string()));
        }
    }

    None
}

fn parse_markdown_table_row(line: &str) -> Option<Vec<String>> {
    let trimmed = line.trim();
    if !trimmed.contains('|') {
        return None;
    }

    let core = trimmed.trim_matches('|');
    let cells = core
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect::<Vec<_>>();

    if cells.len() < 2 {
        return None;
    }
    Some(cells)
}

fn is_markdown_table_separator(cells: &[String]) -> bool {
    !cells.is_empty()
        && cells.iter().all(|cell| {
            let trimmed = cell.trim();
            !trimmed.is_empty()
                && trimmed.contains('-')
                && trimmed
                    .chars()
                    .all(|ch| matches!(ch, '-' | ':' | ' ' | '\t'))
        })
}

fn table_border(widths: &[usize], left: char, sep: char, right: char) -> String {
    let mut out = String::new();
    out.push(left);
    for (idx, width) in widths.iter().enumerate() {
        out.push_str(&"─".repeat(*width + 2));
        if idx + 1 < widths.len() {
            out.push(sep);
        }
    }
    out.push(right);
    out
}

fn table_row(cells: &[String], widths: &[usize]) -> String {
    let mut out = String::new();
    out.push('│');
    for (idx, width) in widths.iter().enumerate() {
        let cell = cells.get(idx).map(String::as_str).unwrap_or("");
        let pad = width.saturating_sub(cell.chars().count());
        out.push(' ');
        out.push_str(cell);
        out.push_str(&" ".repeat(pad));
        out.push(' ');
        out.push('│');
    }
    out
}

fn render_markdown_table_lines(rows: &[Vec<String>], theme: UiTheme) -> Vec<Line<'static>> {
    if rows.is_empty() {
        return vec![];
    }

    let col_count = rows.iter().map(|row| row.len()).max().unwrap_or(0);
    if col_count == 0 {
        return vec![];
    }

    let mut widths = vec![3usize; col_count];
    for row in rows {
        for (idx, cell) in row.iter().enumerate() {
            widths[idx] = widths[idx].max(cell.chars().count());
        }
    }

    let mut lines = Vec::new();
    lines.push(Line::from(vec![Span::styled(
        table_border(&widths, '┌', '┬', '┐'),
        Style::default().fg(theme.muted),
    )]));

    lines.push(Line::from(vec![Span::styled(
        table_row(&rows[0], &widths),
        Style::default()
            .fg(theme.info_cyan)
            .add_modifier(Modifier::BOLD),
    )]));

    if rows.len() > 1 {
        lines.push(Line::from(vec![Span::styled(
            table_border(&widths, '├', '┼', '┤'),
            Style::default().fg(theme.muted),
        )]));
        for row in rows.iter().skip(1) {
            lines.push(Line::from(vec![Span::styled(
                table_row(row, &widths),
                Style::default().fg(theme.fg),
            )]));
        }
    }

    lines.push(Line::from(vec![Span::styled(
        table_border(&widths, '└', '┴', '┘'),
        Style::default().fg(theme.muted),
    )]));
    lines
}

fn render_inline_markdown_spans(
    text: &str,
    base_style: Style,
    theme: UiTheme,
) -> Vec<Span<'static>> {
    let mut spans = Vec::new();
    let mut buffer = String::new();
    let mut idx = 0usize;
    let mut bold = false;
    let mut crossed_out = false;

    let inline_code_style = Style::default()
        .fg(theme.info_cyan)
        .bg(Color::Rgb(47, 55, 73))
        .add_modifier(Modifier::BOLD);

    while idx < text.len() {
        let remaining = &text[idx..];

        if let Some(escaped) = remaining.strip_prefix('\\') {
            if let Some(ch) = escaped.chars().next() {
                if matches!(ch, '\\' | '`' | '*' | '_' | '~') {
                    buffer.push(ch);
                    idx += 1 + ch.len_utf8();
                    continue;
                }
            }
        }

        if remaining.starts_with('`') {
            if let Some(end_rel) = text[idx + 1..].find('`') {
                if !buffer.is_empty() {
                    let mut style = base_style;
                    if bold {
                        style = style.add_modifier(Modifier::BOLD);
                    }
                    if crossed_out {
                        style = style.add_modifier(Modifier::CROSSED_OUT);
                    }
                    spans.push(Span::styled(std::mem::take(&mut buffer), style));
                }

                let end_idx = idx + 1 + end_rel;
                let code = &text[idx + 1..end_idx];
                spans.push(Span::styled(code.to_string(), inline_code_style));
                idx = end_idx + 1;
                continue;
            }

            buffer.push('`');
            idx += 1;
            continue;
        }

        if remaining.starts_with("**") || remaining.starts_with("__") {
            let delimiter = if remaining.starts_with("**") {
                "**"
            } else {
                "__"
            };
            let has_closer = text[idx + 2..].contains(delimiter);
            if bold || has_closer {
                if !buffer.is_empty() {
                    let mut style = base_style;
                    if bold {
                        style = style.add_modifier(Modifier::BOLD);
                    }
                    if crossed_out {
                        style = style.add_modifier(Modifier::CROSSED_OUT);
                    }
                    spans.push(Span::styled(std::mem::take(&mut buffer), style));
                }
                bold = !bold;
                idx += 2;
                continue;
            }
        }

        if remaining.starts_with("~~") {
            let has_closer = text[idx + 2..].contains("~~");
            if crossed_out || has_closer {
                if !buffer.is_empty() {
                    let mut style = base_style;
                    if bold {
                        style = style.add_modifier(Modifier::BOLD);
                    }
                    if crossed_out {
                        style = style.add_modifier(Modifier::CROSSED_OUT);
                    }
                    spans.push(Span::styled(std::mem::take(&mut buffer), style));
                }
                crossed_out = !crossed_out;
                idx += 2;
                continue;
            }
        }

        if let Some(ch) = remaining.chars().next() {
            buffer.push(ch);
            idx += ch.len_utf8();
        } else {
            break;
        }
    }

    if !buffer.is_empty() {
        let mut style = base_style;
        if bold {
            style = style.add_modifier(Modifier::BOLD);
        }
        if crossed_out {
            style = style.add_modifier(Modifier::CROSSED_OUT);
        }
        spans.push(Span::styled(buffer, style));
    }

    if spans.is_empty() {
        spans.push(Span::styled(String::new(), base_style));
    }
    spans
}

fn syntax_for_markdown_hint<'a>(ps: &'a SyntaxSet, lang_hint: Option<&str>) -> &'a SyntaxReference {
    if let Some(hint) = lang_hint.map(str::trim).filter(|hint| !hint.is_empty()) {
        if let Some(syntax) = ps.find_syntax_by_token(hint) {
            return syntax;
        }
        if let Some(syntax) = ps.find_syntax_by_extension(hint) {
            return syntax;
        }
    }
    ps.find_syntax_plain_text()
}

fn render_markdown_code_block_lines(
    lang_hint: Option<&str>,
    code: &str,
    theme: UiTheme,
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let code_bg = Color::Rgb(30, 37, 52);
    let label = lang_hint
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("text");

    lines.push(Line::from(vec![Span::styled(
        format!("▸ código ({label})"),
        Style::default()
            .fg(theme.info_blue)
            .add_modifier(Modifier::BOLD),
    )]));

    let assets = highlight_assets();
    let syntax = syntax_for_markdown_hint(&assets.ps, lang_hint);
    let Some(theme_ref) = select_syntect_theme(&assets.ts) else {
        for (idx, raw) in text_lines_or_empty(code).iter().enumerate() {
            lines.push(Line::from(vec![
                Span::styled(
                    format!("{:>4} ", idx + 1),
                    Style::default().fg(theme.muted).bg(code_bg),
                ),
                Span::styled(raw.to_string(), Style::default().fg(theme.fg).bg(code_bg)),
            ]));
        }
        return lines;
    };

    let mut highlighter = HighlightLines::new(syntax, theme_ref);
    for (idx, raw) in text_lines_or_empty(code).iter().enumerate() {
        let mut spans = Vec::new();
        spans.push(Span::styled(
            format!("{:>4} ", idx + 1),
            Style::default().fg(theme.muted).bg(code_bg),
        ));
        match highlighter.highlight_line(raw, &assets.ps) {
            Ok(ranges) => {
                for (style, token) in ranges {
                    spans.push(Span::styled(
                        token.to_string(),
                        Style::default()
                            .fg(syntect_color_to_ratatui(style.foreground))
                            .bg(code_bg),
                    ));
                }
            }
            Err(_) => spans.push(Span::styled(
                raw.to_string(),
                Style::default().fg(theme.fg).bg(code_bg),
            )),
        }
        lines.push(Line::from(spans));
    }

    lines
}

fn render_markdown_assistant_lines(text: &str, theme: UiTheme) -> Vec<Line<'static>> {
    let raw_lines = text_lines_or_empty(text);
    let mut lines = Vec::new();
    let mut idx = 0usize;
    let mut in_code_block = false;
    let mut code_lang: Option<String> = None;
    let mut code_lines: Vec<String> = Vec::new();

    while idx < raw_lines.len() {
        let raw = raw_lines[idx];
        let trimmed = raw.trim();

        if let Some(fence) = trimmed.strip_prefix("```") {
            if in_code_block {
                lines.extend(render_markdown_code_block_lines(
                    code_lang.as_deref(),
                    &code_lines.join("\n"),
                    theme,
                ));
                code_lines.clear();
                code_lang = None;
                in_code_block = false;
            } else {
                in_code_block = true;
                let hint = fence.trim();
                if !hint.is_empty() {
                    code_lang = Some(hint.to_string());
                }
            }
            idx += 1;
            continue;
        }

        if in_code_block {
            code_lines.push(raw.to_string());
            idx += 1;
            continue;
        }

        if parse_markdown_table_row(raw).is_some() {
            let mut table_rows: Vec<Vec<String>> = Vec::new();
            let mut cursor = idx;
            while cursor < raw_lines.len() {
                if let Some(row) = parse_markdown_table_row(raw_lines[cursor]) {
                    table_rows.push(row);
                    cursor += 1;
                } else {
                    break;
                }
            }
            if table_rows.len() >= 2 && is_markdown_table_separator(&table_rows[1]) {
                let mut normalized_rows = Vec::new();
                normalized_rows.push(table_rows[0].clone());
                for row in table_rows.into_iter().skip(2) {
                    normalized_rows.push(row);
                }
                lines.extend(render_markdown_table_lines(&normalized_rows, theme));
                idx = cursor;
                continue;
            }
        }

        if trimmed.is_empty() {
            lines.push(Line::raw(""));
            idx += 1;
            continue;
        }

        if let Some((level, title)) = parse_markdown_heading(raw) {
            let heading_style = match level {
                1 => Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
                2 => Style::default()
                    .fg(theme.info_cyan)
                    .add_modifier(Modifier::BOLD),
                _ => Style::default()
                    .fg(theme.info_blue)
                    .add_modifier(Modifier::BOLD),
            };
            lines.push(Line::from(vec![Span::styled(
                title.to_string(),
                heading_style,
            )]));
            if level <= 2 {
                lines.push(Line::from(vec![Span::styled(
                    "─".repeat(title.chars().count().max(3)),
                    Style::default().fg(theme.muted),
                )]));
            }
            idx += 1;
            continue;
        }

        if let Some(quote) = trimmed.strip_prefix("> ") {
            lines.push(rail_line(
                theme.info_cyan,
                render_inline_markdown_spans(quote, Style::default().fg(theme.fg), theme),
            ));
            idx += 1;
            continue;
        }

        if let Some((indent, marker, content)) = parse_markdown_list_item(raw) {
            let mut spans = Vec::new();
            spans.push(Span::styled(
                format!("{}{} ", "  ".repeat(indent), marker),
                Style::default()
                    .fg(theme.info_blue)
                    .add_modifier(Modifier::BOLD),
            ));
            spans.extend(render_inline_markdown_spans(
                &content,
                Style::default().fg(theme.fg),
                theme,
            ));
            lines.push(Line::from(spans));
            idx += 1;
            continue;
        }

        lines.push(Line::from(render_inline_markdown_spans(
            raw.trim_end(),
            Style::default().fg(theme.fg),
            theme,
        )));
        idx += 1;
    }

    if in_code_block {
        lines.extend(render_markdown_code_block_lines(
            code_lang.as_deref(),
            &code_lines.join("\n"),
            theme,
        ));
    }

    if lines.is_empty() {
        lines.push(Line::raw(""));
    }
    lines
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

/// Text shimmer effect: a bright "wave" travels across the text on a base color.
fn shimmer_spans(text: &str, frame: usize, base_color: Color) -> Vec<Span<'static>> {
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len().max(1);
    // Bright spot position cycles through the text
    let bright_pos = frame % (len + 4); // +4 gives a trailing gap
    chars
        .iter()
        .enumerate()
        .map(|(idx, ch)| {
            let dist = (idx as isize - bright_pos as isize).unsigned_abs();
            let color = if dist == 0 {
                Color::White
            } else if dist == 1 {
                lerp_color(Color::White, base_color, 0.4)
            } else if dist == 2 {
                lerp_color(Color::White, base_color, 0.7)
            } else {
                base_color
            };
            Span::styled(
                ch.to_string(),
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            )
        })
        .collect()
}

fn lerp_color(a: Color, b: Color, t: f32) -> Color {
    let (ar, ag, ab) = color_to_rgb(a);
    let (br, bg, bb) = color_to_rgb(b);
    Color::Rgb(
        (ar as f32 + (br as f32 - ar as f32) * t) as u8,
        (ag as f32 + (bg as f32 - ag as f32) * t) as u8,
        (ab as f32 + (bb as f32 - ab as f32) * t) as u8,
    )
}

fn color_to_rgb(c: Color) -> (u8, u8, u8) {
    match c {
        Color::Rgb(r, g, b) => (r, g, b),
        Color::White => (255, 255, 255),
        _ => (180, 180, 180),
    }
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
    _active_agent: BuiltinAgent,
    skills_count: usize,
    mcp_servers_count: usize,
    theme: UiTheme,
) -> Vec<Line<'static>> {
    let model = short_preview(&model_label.to_ascii_lowercase(), 24);
    let downloaded_models = count_downloaded_models();
    let model_label = if downloaded_models == 1 {
        "1 modelo baixado".to_string()
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
        Span::styled(format!("{model_label}"), Style::default().fg(theme.muted)),
        Span::styled(
            format!(
                " · {} servidor(es) MCP · {} skill(s)",
                mcp_servers_count, skills_count
            ),
            Style::default().fg(theme.muted),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled(CAT_LOGO_LINES[2], Style::default().fg(theme.muted)),
        Span::raw("  "),
        Span::styled("Digite ", Style::default().fg(theme.fg)),
        Span::styled("/help", Style::default().fg(theme.info_blue)),
        Span::styled(" para mais informações", Style::default().fg(theme.fg)),
    ]));
    lines.push(Line::raw(""));
    lines
}

fn render_chat_lines(
    chat: &[ChatItem],
    output_expanded: bool,
    spin: usize,
    model_label: &str,
    active_agent: BuiltinAgent,
    skills_count: usize,
    mcp_servers_count: usize,
    theme: UiTheme,
) -> Vec<Line<'static>> {
    let tools_collapsed = !output_expanded;
    let thinking_collapsed = !output_expanded;
    let code_blocks_collapsed = !output_expanded;
    let mut lines: Vec<Line<'static>> = Vec::new();

    for (item_idx, item) in chat.iter().enumerate() {
        match item {
            ChatItem::Banner => lines.extend(render_brand_banner(
                model_label,
                active_agent,
                skills_count,
                mcp_servers_count,
                theme,
            )),
            ChatItem::User(text) => {
                let text_lines = text_lines_or_empty(text);
                for (idx, line) in text_lines.iter().enumerate() {
                    if idx == 0 {
                        lines.push(Line::from(vec![
                            Span::styled(
                                "❯ ",
                                Style::default()
                                    .fg(theme.accent)
                                    .add_modifier(Modifier::BOLD),
                            ),
                            Span::styled(
                                line.to_string(),
                                Style::default()
                                    .fg(theme.fg)
                                    .add_modifier(Modifier::BOLD),
                            ),
                        ]));
                    } else {
                        lines.push(Line::from(vec![
                            Span::raw("  "),
                            Span::styled(
                                line.to_string(),
                                Style::default()
                                    .fg(theme.fg)
                                    .add_modifier(Modifier::BOLD),
                            ),
                        ]));
                    }
                }
                // Skip blank line if next item is AttachmentStatus (keep it tight)
                let next_is_attachment = chat
                    .get(item_idx + 1)
                    .map_or(false, |next| matches!(next, ChatItem::AttachmentStatus(_)));
                if !next_is_attachment {
                    lines.push(Line::raw(""));
                }
            }
            ChatItem::Thinking { text, active } => {
                if *active {
                    let icon = spinner_char(spin).to_string();
                    let mut title_spans = vec![Span::styled(
                        format!("{} ", icon),
                        Style::default()
                            .fg(theme.info_blue)
                            .add_modifier(Modifier::BOLD),
                    )];
                    title_spans.extend(shimmer_spans("Raciocínio", spin, theme.info_blue));
                    lines.push(Line::from(title_spans));
                } else {
                    lines.push(Line::from(vec![Span::styled(
                        "⁕ Raciocínio",
                        Style::default()
                            .fg(theme.info_blue)
                            .add_modifier(Modifier::BOLD),
                    )]));
                }

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
                        "▸ modo teleprompter (Ctrl+O para expandir)"
                    } else {
                        "▸ Ctrl+O para expandir"
                    };
                    lines.push(rail_line(
                        theme.muted,
                        vec![Span::styled(hint, Style::default().fg(theme.muted))],
                    ));
                } else {
                    lines.push(rail_line(
                        theme.muted,
                        vec![Span::styled(
                            "▾ raciocínio completo visível (Ctrl+O para recolher)",
                            Style::default().fg(theme.muted),
                        )],
                    ));
                }

                lines.push(Line::raw(""));
            }
            ChatItem::Assistant(text) => {
                lines.extend(render_markdown_assistant_lines(text, theme));
                lines.push(Line::raw(""));
            }
            ChatItem::Tool {
                tool_name,
                summary,
                stream,
                output,
                code_path,
                code,
                diff,
                state,
                subagent,
                started_at,
            } => {
                // ── Subagent (Explore) teleprompter-style rendering ──
                if let Some(tracking) = subagent {
                    let is_running = matches!(state, ToolState::Running);
                    let icon = if is_running {
                        format!("{} ", spinner_char(spin))
                    } else {
                        "● ".to_string()
                    };
                    let bullet_color = if is_running {
                        theme.fg
                    } else {
                        theme.info_blue
                    };

                    let (tool_label, tool_suffix) = split_tool_summary(summary);
                    lines.push(Line::from(vec![
                        Span::styled(
                            icon,
                            Style::default()
                                .fg(bullet_color)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            tool_label.to_string(),
                            Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(tool_suffix.to_string(), Style::default().fg(theme.fg)),
                    ]));

                    if is_running {
                        // Show current running tool
                        if let Some(current) = tracking.sub_tools.last() {
                            let current_summary = short_preview(&current.summary, 100);
                            let status_text = if current.done {
                                current_summary
                            } else {
                                format!("Executando…{}", current_summary)
                            };
                            lines.push(explore_detail_line(
                                theme,
                                vec![Span::styled(status_text, Style::default().fg(theme.muted))],
                            ));
                        }

                        // Show count of completed tools (collapsed by default)
                        let done_count = tracking.tools_done as usize;
                        if done_count > 0 {
                            if code_blocks_collapsed {
                                lines.push(explore_detail_line(
                                    theme,
                                    vec![Span::styled(
                                        format!(
                                            "+{} usos de ferramenta (Ctrl+O para expandir)",
                                            done_count
                                        ),
                                        Style::default().fg(theme.muted),
                                    )],
                                ));
                            } else {
                                // Show all completed sub-tools
                                for entry in tracking.sub_tools.iter().filter(|e| e.done) {
                                    lines.push(explore_detail_line(
                                        theme,
                                        vec![Span::styled(
                                            short_preview(&entry.summary, 100),
                                            Style::default().fg(theme.muted),
                                        )],
                                    ));
                                }
                                lines.push(explore_detail_line(
                                    theme,
                                    vec![Span::styled(
                                        "▾ Ctrl+O para recolher",
                                        Style::default().fg(theme.muted),
                                    )],
                                ));
                            }
                        }
                    } else {
                        // Completed: show summary line
                        if let Some(s) = stream {
                            lines.push(explore_detail_line(
                                theme,
                                vec![Span::styled(
                                    s.to_string(),
                                    Style::default().fg(theme.muted),
                                )],
                            ));
                        }

                        // Show sub-tools list (collapsed by default)
                        if !tracking.sub_tools.is_empty() {
                            if code_blocks_collapsed {
                                lines.push(explore_detail_line(
                                    theme,
                                    vec![Span::styled(
                                        "(Ctrl+O para expandir)",
                                        Style::default().fg(theme.muted),
                                    )],
                                ));
                            } else {
                                for entry in &tracking.sub_tools {
                                    let icon = if entry.done { "✓" } else { "✗" };
                                    lines.push(explore_detail_line(
                                        theme,
                                        vec![
                                            Span::styled(
                                                format!("{} ", icon),
                                                Style::default().fg(if entry.done {
                                                    theme.success
                                                } else {
                                                    theme.danger
                                                }),
                                            ),
                                            Span::styled(
                                                short_preview(&entry.summary, 100),
                                                Style::default().fg(theme.muted),
                                            ),
                                        ],
                                    ));
                                }
                                lines.push(explore_detail_line(
                                    theme,
                                    vec![Span::styled(
                                        "▾ Ctrl+O para recolher",
                                        Style::default().fg(theme.muted),
                                    )],
                                ));
                            }
                        }
                    }

                    lines.push(Line::raw(""));
                } else {
                    // ── Regular tool rendering ──
                    let is_running = matches!(state, ToolState::Running);
                    let is_error = matches!(state, ToolState::Error);

                    // Blinking bullet: alternate visibility when running
                    let bullet_visible = if is_running {
                        (spin / 3) % 2 == 0
                    } else {
                        true
                    };
                    let bullet_char = if bullet_visible { "● " } else { "  " };
                    let bullet_color = if is_running {
                        theme.warning
                    } else if is_error {
                        theme.danger
                    } else {
                        theme.accent
                    };

                    let (tool_label, tool_suffix) = split_tool_summary(summary);
                    lines.push(Line::from(vec![
                        Span::styled(
                            bullet_char,
                            Style::default()
                                .fg(bullet_color)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            tool_label.to_string(),
                            Style::default().fg(theme.fg).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(tool_suffix.to_string(), Style::default().fg(theme.fg)),
                    ]));

                    // Running state: show elapsed time
                    if is_running {
                        let elapsed_secs = started_at
                            .map(|t| t.elapsed().as_secs())
                            .unwrap_or(0);
                        let elapsed_str = format_elapsed_clock(elapsed_secs);
                        let hint = format!("Executando… ({})", elapsed_str);
                        for hint_line in hint.lines() {
                            lines.push(tool_detail_line(
                                true,
                                theme,
                                vec![Span::styled(
                                    hint_line.to_string(),
                                    Style::default().fg(theme.muted),
                                )],
                            ));
                        }
                    } else {
                        // Completed: show status line
                        if let Some(s) = stream {
                            lines.push(tool_detail_line(
                                true,
                                theme,
                                vec![Span::styled(
                                    s.to_string(),
                                    Style::default().fg(theme.muted),
                                )],
                            ));
                        }
                    }

                    let has_tool_details = output.is_some() || code.is_some() || diff.is_some();
                    if has_tool_details && !is_running {
                        if tools_collapsed {
                            if let Some(out) = output {
                                // Show preview even when collapsed
                                lines.extend(render_tool_output_lines(out, theme, true));
                            }
                        } else {
                            if let Some(out) = output {
                                lines.extend(render_tool_output_lines(out, theme, false));
                            }
                            if code_blocks_collapsed && (code.is_some() || diff.is_some()) {
                                lines.push(tool_detail_line(
                                    true,
                                    theme,
                                    vec![Span::styled(
                                        "▸ código/diff oculto (ctrl+o para mostrar)",
                                        Style::default().fg(theme.muted),
                                    )],
                                ));
                            } else if let Some(diff_text) = diff {
                                lines.extend(render_diff_lines(
                                    diff_text,
                                    code_path.as_deref(),
                                    theme,
                                ));
                            } else if let Some(code_text) = code {
                                let path = code_path
                                    .as_deref()
                                    .filter(|value| !value.trim().is_empty())
                                    .unwrap_or("arquivo");
                                let section_title = match tool_name.as_str() {
                                    "read_file" => "Código lido",
                                    "write_file" => "Código criado",
                                    _ => "Código",
                                };
                                lines.extend(render_highlighted_code_lines(
                                    section_title,
                                    path,
                                    code_text,
                                    theme,
                                ));
                            }
                        }
                    }

                    lines.push(Line::raw(""));
                }
            }
            ChatItem::AttachmentStatus(text) => {
                lines.push(Line::from(vec![
                    Span::styled("  ⎿ ", Style::default().fg(theme.muted)),
                    Span::styled(text.clone(), Style::default().fg(theme.muted)),
                ]));
                // No trailing blank line — next AttachmentStatus or Thinking follows tightly.
                // Add blank line only if next item is NOT another AttachmentStatus.
                let next_is_attachment = chat
                    .get(item_idx + 1)
                    .map_or(false, |next| matches!(next, ChatItem::AttachmentStatus(_)));
                if !next_is_attachment {
                    lines.push(Line::raw(""));
                }
            }
            ChatItem::Compact {
                active,
                old_tokens,
                new_tokens,
            } => {
                if *active {
                    // Shimmer on icon + text while compacting
                    let blink = (spin / 2) % 2 == 0;
                    let icon_style = if blink {
                        Style::default()
                            .fg(theme.info_blue)
                            .add_modifier(Modifier::BOLD)
                    } else {
                        Style::default().fg(theme.muted).add_modifier(Modifier::BOLD)
                    };
                    let mut spans = vec![Span::styled("✻ ", icon_style)];
                    spans.extend(shimmer_spans(
                        "Compactando conversa",
                        spin,
                        theme.info_blue,
                    ));
                    spans.push(Span::styled("...", Style::default().fg(theme.muted)));
                    lines.push(Line::from(spans));
                } else if let Some(new_tk) = new_tokens {
                    // Completed compact
                    lines.push(Line::from(vec![Span::styled(
                        "✻ Conversa compactada (ctrl+o para ver resumo)",
                        Style::default()
                            .fg(theme.info_blue)
                            .add_modifier(Modifier::BOLD),
                    )]));
                    lines.push(Line::from(vec![
                        Span::styled("  ⎿ ", Style::default().fg(theme.muted)),
                        Span::styled(
                            format!("{} → {} Tokens compactados", old_tokens, new_tk),
                            Style::default().fg(theme.muted),
                        ),
                    ]));
                }
                lines.push(Line::raw(""));
            }
            ChatItem::Error(text) => {
                lines.push(Line::from(vec![
                    Span::styled(
                        "erro ",
                        Style::default()
                            .fg(theme.danger)
                            .add_modifier(Modifier::BOLD),
                    ),
                    Span::styled(text.clone(), Style::default().fg(theme.danger)),
                ]));
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
    let elapsed = format_elapsed_clock(elapsed_secs);

    let mut spans = render_opencode_spinner(app.spinner_idx, theme);
    spans.push(Span::raw(" "));
    spans.extend(gradient_spans("Gerando", app.spinner_idx));
    spans.push(Span::styled(
        "… ",
        Style::default().fg(LOADING_GRADIENT[(app.spinner_idx + 1) % LOADING_GRADIENT.len()]),
    ));
    spans.push(Span::styled(
        format!("({} Esc para interromper)", elapsed),
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

    let mut lines = Vec::with_capacity(visible + 1);
    for (idx, cmd) in app.slash_suggestions[start..end].iter().enumerate() {
        let absolute_idx = start + idx;
        let selected = absolute_idx == app.slash_selected;

        let (alias_style, desc_style) = if selected {
            (
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD | Modifier::REVERSED),
                Style::default()
                    .fg(theme.fg)
                    .add_modifier(Modifier::REVERSED),
            )
        } else {
            (
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
                Style::default().fg(theme.muted),
            )
        };

        lines.push(Line::from(vec![
            Span::styled(format!(" {} ", cmd.alias), alias_style),
            Span::raw(" "),
            Span::styled(
                if cmd.is_skill { "skill" } else { "comando" },
                Style::default().fg(theme.info_blue),
            ),
            Span::raw(" "),
            Span::styled(cmd.description.to_string(), desc_style),
        ]));
    }

    let scroll_hint = if total > visible {
        format!(" [{}/{}]", app.slash_selected + 1, total)
    } else {
        String::new()
    };
    lines.push(Line::from(vec![Span::styled(
        format!("↑↓ selecionar · Tab completar · Enter enviar · Esc fechar{scroll_hint}"),
        Style::default().fg(theme.muted),
    )]));

    lines
}

fn render_mention_suggestion_lines(app: &AppState, theme: UiTheme) -> Vec<Line<'static>> {
    if app.mention_suggestions.is_empty() {
        return vec![Line::raw("")];
    }

    let total = app.mention_suggestions.len();
    let visible = total.min(MAX_SLASH_SUGGESTIONS);
    let start = app
        .mention_selected
        .saturating_add(1)
        .saturating_sub(visible)
        .min(total.saturating_sub(visible));
    let end = (start + visible).min(total);

    let mut lines = Vec::with_capacity(visible + 1);
    for (idx, suggestion) in app.mention_suggestions[start..end].iter().enumerate() {
        let absolute_idx = start + idx;
        let selected = absolute_idx == app.mention_selected;

        let (display_style, detail_style) = if selected {
            (
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD | Modifier::REVERSED),
                Style::default()
                    .fg(theme.fg)
                    .add_modifier(Modifier::REVERSED),
            )
        } else {
            (
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD),
                Style::default().fg(theme.muted),
            )
        };

        let kind = if suggestion.is_directory {
            "diretório"
        } else {
            "arquivo"
        };
        lines.push(Line::from(vec![
            Span::styled(format!(" {} ", suggestion.display), display_style),
            Span::raw(" "),
            Span::styled(
                format!("{} · {}", kind, suggestion.description),
                detail_style,
            ),
        ]));
    }

    let scroll_hint = if total > visible {
        format!(" [{}/{}]", app.mention_selected + 1, total)
    } else {
        String::new()
    };
    lines.push(Line::from(vec![Span::styled(
        format!("↑↓ selecionar · Tab completar · Enter enviar{scroll_hint}"),
        Style::default().fg(theme.muted),
    )]));

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
    let mut welcome_spans = vec![Span::styled(
        "Boas-vindas ao ",
        Style::default().fg(theme.fg),
    )];
    welcome_spans.extend(gradient_spans(WELCOME_HIGHLIGHT, app.spinner_idx));
    welcome_spans.push(Span::styled(
        " - vamos começar!",
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
        "Pressione Enter ↵",
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
        let empty = Paragraph::new("Estado do setup indisponível")
            .alignment(Alignment::Center)
            .style(Style::default().bg(theme.bg).fg(theme.danger));
        frame.render_widget(empty, size);
        return;
    };

    let header = centered_rect(58, 3, size);
    let title = Paragraph::new(vec![Line::from(vec![
        Span::styled(
            "Só mais uma coisa",
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
            Span::styled("Modelo: ", Style::default().fg(theme.muted)),
            Span::styled(
                model.model_display_name.clone(),
                Style::default().fg(theme.info_cyan),
            ),
            Span::raw(" · "),
            Span::styled(
                format!(
                    "{} · raciocínio:{} · visão:{}",
                    model.category_label,
                    if model.supports_thinking {
                        "sim"
                    } else {
                        "não"
                    },
                    if model.supports_vision { "sim" } else { "não" }
                ),
                Style::default().fg(theme.fg),
            ),
        ]));
        info_lines.push(Line::from(vec![
            Span::styled("Contexto: ", Style::default().fg(theme.muted)),
            Span::styled(
                format!(
                    "máx {} · recomendado {} (geral) / {} (código)",
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
        "Selecionar modelo".to_string()
    } else {
        format!(
            "Selecionar quantização · {}",
            setup.current_model_display_name
        )
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
                    "cache: nenhum".to_string()
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
                rows.push(Line::from("Nenhum modelo disponível"));
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
                rows.push(Line::from("Nenhuma quantização disponível"));
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
                    "Erro: ",
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
                            "nenhum".to_string()
                        } else {
                            model.cached_quants.join(", ")
                        };
                        footer_lines.push(Line::from(vec![Span::styled(
                            format!(
                                "Capacidades: raciocínio={} · visão={} · variantes em cache: {}",
                                if model.supports_thinking {
                                    "sim"
                                } else {
                                    "não"
                                },
                                if model.supports_vision { "sim" } else { "não" },
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
                    "↑/↓ selecionar modelo · Enter mostrar variantes · Esc voltar ao chat"
                } else {
                    "↑/↓ selecionar modelo · Enter mostrar variantes · Esc sair"
                }
            }
            ModelSetupView::Variants => {
                "↑/↓ selecionar variante · Enter ativar/baixar · Esc voltar aos modelos"
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

fn render_chat_screen(frame: &mut ratatui::Frame<'_>, app: &AppState, size: Rect, theme: UiTheme) {
    let config_app_active = app.screen == UiScreen::Config && app.config_screen.is_some();
    let bottom_app_active = app.pending_approval.is_some()
        || app.pending_user_question.is_some()
        || app.pending_plan_review.is_some()
        || app.pending_resume_selection.is_some()
        || config_app_active;

    let loading_h = if app.busy
        && app.pending_approval.is_none()
        && app.pending_user_question.is_none()
        && app.pending_plan_review.is_none()
        && app.pending_resume_selection.is_none()
        && !config_app_active
    {
        1
    } else {
        0
    };
    let input_lines = text_lines_or_empty(&app.input).len() as u16;
    let input_h = if let Some(approval) = app.pending_approval.as_ref() {
        // 8 = header(1) + blank(1) + summary(1) + blank(1) + 3 options + hint(1)
        // + 2 for border
        let base: u16 = 10;
        let detail_lines = if approval.details.is_empty() {
            0
        } else {
            approval.details.len() as u16 + 1
        };
        let diff_lines = approval
            .diff_preview
            .as_ref()
            .map(|d| d.lines().count() as u16 + 1) // +1 for blank separator
            .unwrap_or(0);
        let max_h = size.height / 2;
        let min_h = base.min(max_h);
        (base + detail_lines + diff_lines).min(max_h).max(min_h)
    } else if app.pending_plan_review.is_some() {
        let base: u16 = 10;
        let max_h = size.height / 2;
        base.min(max_h).max(4)
    } else if let Some(question) = app.pending_user_question.as_ref() {
        let choices_h = question.choices.len().min(9) as u16;
        let free_text_h = if question.allow_free_text { 3 } else { 0 };
        let base: u16 = 8 + choices_h + free_text_h;
        let max_h = size.height / 2;
        base.min(max_h).max(8)
    } else if let Some(pending_resume) = app.pending_resume_selection.as_ref() {
        let visible_sessions = pending_resume.sessions.len().min(6) as u16;
        let base: u16 = visible_sessions + 5;
        let max_h = size.height / 2;
        base.min(max_h).max(6)
    } else if config_app_active {
        let base: u16 = 14;
        let max_h = size.height.saturating_sub(8).max(8);
        base.min(max_h).max(8)
    } else {
        input_lines.clamp(1, 3) + 2
    };
    let show_slash_suggestions = !bottom_app_active
        && app.input_mode == InputMode::Slash
        && !app.slash_suggestions.is_empty();
    let show_mention_suggestions = !bottom_app_active
        && app.input_mode == InputMode::Default
        && !app.mention_suggestions.is_empty();
    let suggestions_lines = if show_slash_suggestions {
        app.slash_suggestions.len().min(MAX_SLASH_SUGGESTIONS) as u16 + 1
    } else if show_mention_suggestions {
        app.mention_suggestions.len().min(MAX_SLASH_SUGGESTIONS) as u16 + 1
    } else {
        0
    };
    let suggestions_h = if suggestions_lines > 0 {
        suggestions_lines + 2
    } else {
        0
    };

    let root = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),
            Constraint::Length(suggestions_h),
            Constraint::Length(input_h),
            Constraint::Length(loading_h),
            Constraint::Length(1),
            Constraint::Length(2),
        ])
        .split(size);

    let mut chat_lines = render_chat_lines(
        &app.chat,
        app.output_expanded,
        app.spinner_idx,
        &app.model_label,
        app.active_agent,
        app.skills_count,
        app.mcp_servers_count,
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
    let chat_content_len = chat_lines.len();
    // chat_scroll is stored as "distance from bottom", so 0 means follow latest messages.
    let chat_scroll = max_scroll.saturating_sub(app.chat_scroll);

    let show_scrollbar = max_scroll > 0 && root[0].height > 0 && root[0].width > 2;
    let (chat_area, scrollbar_area) = if show_scrollbar {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(1), Constraint::Length(1)])
            .split(root[0]);
        (cols[0], Some(cols[1]))
    } else {
        (root[0], None)
    };

    let chat = Paragraph::new(chat_lines)
        .wrap(Wrap { trim: false })
        .scroll((chat_scroll, 0))
        .style(Style::default().bg(theme.bg).fg(theme.fg));
    frame.render_widget(chat, chat_area);

    if let Some(scroll_area) = scrollbar_area {
        let mut scrollbar_state = ScrollbarState::new(chat_content_len)
            .position(chat_scroll as usize)
            .viewport_content_length(chat_h);
        let scrollbar = Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(None)
            .end_symbol(None)
            .style(Style::default().fg(theme.border));
        frame.render_stateful_widget(scrollbar, scroll_area, &mut scrollbar_state);
    }

    if suggestions_h > 0 {
        let suggestions_box = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.border).bg(theme.bg));
        let suggestions_area = suggestions_box.inner(root[1]);
        frame.render_widget(suggestions_box, root[1]);
        let lines = if show_slash_suggestions {
            render_slash_suggestion_lines(app, theme)
        } else {
            render_mention_suggestion_lines(app, theme)
        };
        let suggestions = Paragraph::new(lines)
            .wrap(Wrap { trim: false })
            .style(Style::default().bg(theme.bg).fg(theme.fg));
        frame.render_widget(suggestions, suggestions_area);
    }

    if let Some(approval) = app.pending_approval.as_ref() {
        let approval_box = Block::default()
            .borders(Borders::ALL)
            .title("permissao")
            .title_alignment(Alignment::Right)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme.border).bg(theme.bg));
        let approval_area = approval_box.inner(root[2]);
        frame.render_widget(approval_box, root[2]);

        let option_line = |idx: usize, label: &str, selected: bool| -> Line<'static> {
            let marker = if selected { ">" } else { " " };
            let style = if selected {
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme.fg)
            };
            Line::from(vec![Span::styled(
                format!("{} {}. {}", marker, idx, label),
                style,
            )])
        };

        let mut lines = Vec::new();
        lines.push(Line::from(vec![Span::styled(
            "Deseja permitir esta ação da ferramenta?",
            Style::default()
                .fg(theme.warning)
                .add_modifier(Modifier::BOLD),
        )]));
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![
            Span::styled(
                "● ",
                Style::default()
                    .fg(theme.info_cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                approval.summary.clone(),
                Style::default()
                    .fg(theme.info_cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        if !approval.details.is_empty() {
            lines.push(Line::raw(""));
            for detail in &approval.details {
                lines.push(Line::from(vec![
                    Span::styled("  - ", Style::default().fg(theme.muted)),
                    Span::styled(detail.clone(), Style::default().fg(theme.fg)),
                ]));
            }
        }
        if let Some(diff_text) = approval.diff_preview.as_deref() {
            lines.push(Line::raw(""));
            for raw_line in diff_text.lines() {
                let (kind, num_str, sign_str, content) = parse_diff_line(raw_line);
                let (num_style, sign_style, content_style) = match kind {
                    DiffLineKind::Added => (
                        Style::default().fg(theme.muted),
                        Style::default().fg(theme.success),
                        Style::default().fg(theme.success),
                    ),
                    DiffLineKind::Removed => (
                        Style::default().fg(theme.muted),
                        Style::default().fg(theme.danger),
                        Style::default().fg(theme.danger),
                    ),
                    DiffLineKind::Context => (
                        Style::default().fg(theme.muted),
                        Style::default().fg(theme.muted),
                        Style::default().fg(theme.fg),
                    ),
                    DiffLineKind::Meta => (
                        Style::default().fg(theme.muted),
                        Style::default().fg(theme.muted),
                        Style::default().fg(theme.muted),
                    ),
                };
                lines.push(Line::from(vec![
                    Span::styled(format!("{} ", num_str), num_style),
                    Span::styled(sign_str.to_string(), sign_style),
                    Span::styled(content.to_string(), content_style),
                ]));
            }
        }
        lines.push(Line::raw(""));
        lines.push(option_line(
            1,
            "Permitir uma vez",
            approval.selected_option == ApprovalOption::ApproveOnce,
        ));
        lines.push(option_line(
            2,
            "Permitir sempre esta ferramenta (sessao)",
            approval.selected_option == ApprovalOption::ApproveAlwaysToolSession,
        ));
        lines.push(option_line(
            3,
            "Negar",
            approval.selected_option == ApprovalOption::Deny,
        ));
        lines.push(Line::from(vec![Span::styled(
            "Use ↑/↓ para selecionar, 1/2/3 para responder, Enter para confirmar e Esc para negar.",
            Style::default().fg(theme.muted),
        )]));

        let approval_widget = Paragraph::new(lines)
            .style(Style::default().bg(theme.bg).fg(theme.fg))
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: false });
        frame.render_widget(approval_widget, approval_area);
    } else if let Some(question) = app.pending_user_question.as_ref() {
        let question_box = Block::default()
            .borders(Borders::ALL)
            .title("pergunta")
            .title_alignment(Alignment::Right)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme.border).bg(theme.bg));
        let question_area = question_box.inner(root[2]);
        frame.render_widget(question_box, root[2]);

        let mut lines = Vec::new();
        lines.push(Line::from(vec![Span::styled(
            "A IA solicitou uma resposta sua.",
            Style::default()
                .fg(theme.info_blue)
                .add_modifier(Modifier::BOLD),
        )]));
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![
            Span::styled(
                "● ",
                Style::default()
                    .fg(theme.info_cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                question.question.clone(),
                Style::default()
                    .fg(theme.info_cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));

        if !question.choices.is_empty() {
            lines.push(Line::raw(""));
            for (idx, choice) in question.choices.iter().take(9).enumerate() {
                let selected = idx == question.selected_choice;
                let marker = if selected { ">" } else { " " };
                let style = if selected {
                    Style::default()
                        .fg(theme.accent)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme.fg)
                };
                lines.push(Line::from(vec![Span::styled(
                    format!("{} {}. {}", marker, idx + 1, choice),
                    style,
                )]));
            }
        }

        if question.allow_free_text {
            lines.push(Line::raw(""));
            let placeholder = question
                .placeholder
                .clone()
                .unwrap_or_else(|| "Digite sua resposta".to_string());
            let text_display = if question.text_input.is_empty() {
                format!("Texto: {}", placeholder)
            } else {
                format!("Texto: {}", question.text_input)
            };
            let text_style = if question.text_input.is_empty() {
                Style::default().fg(theme.muted)
            } else {
                Style::default().fg(theme.fg)
            };
            lines.push(Line::from(vec![Span::styled(text_display, text_style)]));
        }

        lines.push(Line::raw(""));
        lines.push(Line::from(vec![Span::styled(
            "↑↓ opções · 1..9 seleção rápida · Enter confirma · Esc cancela",
            Style::default().fg(theme.muted),
        )]));

        let question_widget = Paragraph::new(lines)
            .style(Style::default().bg(theme.bg).fg(theme.fg))
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: false });
        frame.render_widget(question_widget, question_area);
    } else if let Some(config_screen) = app.config_screen.as_ref() {
        let config_box = Block::default()
            .borders(Borders::ALL)
            .title("configuração")
            .title_alignment(Alignment::Right)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme.border).bg(theme.bg));
        let config_area = config_box.inner(root[2]);
        frame.render_widget(config_box, root[2]);

        let sections = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(3),
                Constraint::Min(3),
                Constraint::Length(2),
            ])
            .split(config_area);

        let header = Paragraph::new(vec![
            Line::from(vec![
                Span::styled("Modelo ", Style::default().fg(theme.muted)),
                Span::styled(
                    config_screen.model_label.clone(),
                    Style::default().fg(theme.info_cyan),
                ),
            ]),
            Line::from(vec![
                Span::styled("Execução ", Style::default().fg(theme.muted)),
                Span::styled(
                    format!(
                        "ctx {} · out {} · {}",
                        config_screen.runtime_context_tokens,
                        config_screen.runtime_max_tokens,
                        config_screen.hardware_display
                    ),
                    Style::default().fg(theme.fg),
                ),
            ]),
            Line::from(vec![
                Span::styled("Configuração ", Style::default().fg(theme.muted)),
                Span::styled(
                    short_preview(&config_screen.config_path, 78),
                    Style::default().fg(theme.fg),
                ),
            ]),
        ])
        .wrap(Wrap { trim: false })
        .style(Style::default().bg(theme.bg).fg(theme.fg));
        frame.render_widget(header, sections[0]);

        let mut rows = Vec::new();
        if sections[1].height > 0 {
            let total = config_screen.fields.len();
            let visible_rows = sections[1].height as usize;
            let mut start = 0usize;
            if total > visible_rows && visible_rows > 0 {
                start = config_screen
                    .selected_idx
                    .saturating_sub(visible_rows.saturating_div(2))
                    .min(total.saturating_sub(visible_rows));
            }
            let end = (start + visible_rows).min(total);
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
                rows.push(Line::from(vec![
                    Span::styled(cursor, Style::default().fg(theme.accent)),
                    Span::raw(" "),
                    Span::styled(dirty, Style::default().fg(theme.warning)),
                    Span::raw(" "),
                    Span::styled(
                        format!("{:<22}", field.label),
                        Style::default().fg(theme.fg),
                    ),
                    Span::raw(" "),
                    Span::styled(
                        field.value.clone(),
                        if selected {
                            Style::default()
                                .fg(theme.fg)
                                .add_modifier(Modifier::BOLD | Modifier::REVERSED)
                        } else {
                            Style::default().fg(theme.info_blue)
                        },
                    ),
                ]));
            }
        }
        if rows.is_empty() {
            rows.push(Line::raw("Nenhuma configuração disponível"));
        }

        let settings = Paragraph::new(rows)
            .wrap(Wrap { trim: false })
            .style(Style::default().bg(theme.bg).fg(theme.fg));
        frame.render_widget(settings, sections[1]);

        let status_lines = if let Some(err) = &config_screen.error_line {
            vec![Line::from(vec![
                Span::styled(
                    "Erro: ",
                    Style::default()
                        .fg(theme.danger)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(err.clone(), Style::default().fg(theme.danger)),
            ])]
        } else if config_screen.is_dirty() {
            vec![Line::from(vec![Span::styled(
                "↑/↓ seleciona · Enter/Space muda valor · Esc salva e fecha",
                Style::default().fg(theme.warning),
            )])]
        } else {
            vec![
                Line::from(vec![Span::styled(
                    "↑/↓ seleciona · Enter/Space muda valor · Esc fecha",
                    Style::default().fg(theme.muted),
                )]),
                Line::from(vec![Span::styled(
                    format!(
                        "models {} · sessions {}",
                        short_preview(&config_screen.models_path, 34),
                        short_preview(&config_screen.sessions_path, 34)
                    ),
                    Style::default().fg(theme.muted),
                )]),
            ]
        };
        let status_widget =
            Paragraph::new(status_lines).style(Style::default().bg(theme.bg).fg(theme.fg));
        frame.render_widget(status_widget, sections[2]);
    } else if let Some(review) = app.pending_plan_review.as_ref() {
        let review_box = Block::default()
            .borders(Borders::ALL)
            .title("revisão do plano")
            .title_alignment(Alignment::Right)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme.border).bg(theme.bg));
        let review_area = review_box.inner(root[2]);
        frame.render_widget(review_box, root[2]);

        let option_line = |idx: usize, label: &str, selected: bool| -> Line<'static> {
            let marker = if selected { ">" } else { " " };
            let style = if selected {
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme.fg)
            };
            Line::from(vec![Span::styled(
                format!("{} {}. {}", marker, idx, label),
                style,
            )])
        };

        let mut lines = Vec::new();
        lines.push(Line::from(vec![Span::styled(
            "Plano concluído. Como deseja continuar?",
            Style::default()
                .fg(theme.info_blue)
                .add_modifier(Modifier::BOLD),
        )]));
        lines.push(Line::from(vec![Span::styled(
            "Selecione uma resposta",
            Style::default().fg(theme.muted),
        )]));
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![
            Span::styled(
                "● ",
                Style::default()
                    .fg(theme.info_cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "Pergunta ativa",
                Style::default()
                    .fg(theme.info_cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]));
        lines.push(Line::raw(""));
        lines.push(option_line(
            1,
            "Sim, mudar para implementação",
            review.selected_option == PlanReviewOption::ApproveAndBuild,
        ));
        lines.push(option_line(
            2,
            "Não, continuar em plano",
            review.selected_option == PlanReviewOption::Disapprove,
        ));
        lines.push(option_line(
            3,
            "Digitar ajustes para refazer o plano",
            review.selected_option == PlanReviewOption::ReworkWithSuggestion,
        ));
        lines.push(Line::from(vec![Span::styled(
            "↑↓ navegar  Enter selecionar  Tab (na opção 3) digitar  Esc rejeitar",
            Style::default().fg(theme.muted),
        )]));

        let review_widget = Paragraph::new(lines)
            .style(Style::default().bg(theme.bg).fg(theme.fg))
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: false });
        frame.render_widget(review_widget, review_area);
    } else if let Some(pending_resume) = app.pending_resume_selection.as_ref() {
        let resume_box = Block::default()
            .borders(Borders::ALL)
            .title("retomar sessão")
            .title_alignment(Alignment::Right)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme.border).bg(theme.bg));
        let resume_area = resume_box.inner(root[2]);
        frame.render_widget(resume_box, root[2]);

        let total = pending_resume.sessions.len();
        let visible_rows = (resume_area.height.saturating_sub(4) as usize).max(1);
        let visible = total.min(visible_rows);
        let start = pending_resume
            .selected_idx
            .saturating_sub(visible.saturating_div(2))
            .min(total.saturating_sub(visible));
        let end = (start + visible).min(total);

        let mut lines = Vec::new();
        lines.push(Line::from(vec![Span::styled(
            "Selecione uma sessão para retomar",
            Style::default()
                .fg(theme.info_blue)
                .add_modifier(Modifier::BOLD),
        )]));
        lines.push(Line::raw(""));

        for idx in start..end {
            let session = &pending_resume.sessions[idx];
            let selected = idx == pending_resume.selected_idx;
            let marker = if selected { ">" } else { " " };
            let entry_style = if selected {
                Style::default()
                    .fg(theme.accent)
                    .add_modifier(Modifier::BOLD | Modifier::REVERSED)
            } else {
                Style::default().fg(theme.fg)
            };
            lines.push(Line::from(vec![Span::styled(
                format!(
                    "{} {} · {} mensagens · {}",
                    marker,
                    session.id,
                    session.message_count,
                    session.start_time.format("%Y-%m-%d %H:%M")
                ),
                entry_style,
            )]));
        }

        lines.push(Line::raw(""));
        lines.push(Line::from(vec![Span::styled(
            "Use ↑/↓ para navegar, Enter para retomar e Esc para cancelar.",
            Style::default().fg(theme.muted),
        )]));

        let resume_widget = Paragraph::new(lines)
            .style(Style::default().bg(theme.bg).fg(theme.fg))
            .alignment(Alignment::Left)
            .wrap(Wrap { trim: false });
        frame.render_widget(resume_widget, resume_area);
    } else {
        let mode_color = agent_mode_color(app.active_agent, app.yolo_mode, theme);
        let input_box = Block::default()
            .borders(Borders::ALL)
            .title(Line::from(Span::styled(
                agent_mode_label(app.active_agent, app.yolo_mode),
                Style::default().fg(mode_color).add_modifier(Modifier::BOLD),
            )))
            .title_alignment(Alignment::Right)
            .border_style(Style::default().fg(mode_color).bg(theme.bg));
        let input_area = input_box.inner(root[2]);
        frame.render_widget(input_box, root[2]);
        let input = Paragraph::new(render_input_lines(app, theme))
            .wrap(Wrap { trim: false })
            .style(Style::default().bg(theme.bg).fg(theme.fg));
        frame.render_widget(input, input_area);
    }

    if loading_h > 0 {
        let loading = Paragraph::new(vec![render_loading_line(app, theme)])
            .style(Style::default().bg(theme.bg).fg(theme.fg));
        frame.render_widget(loading, root[3]);
    }

    // Process viewer overlay
    if app.process_viewer_open && !app.running_bash_call_ids.is_empty() {
        let process_h: u16 = (app.running_bash_call_ids.len() as u16 + 4).min(size.height / 3);
        let process_w = (size.width * 2 / 3).max(40).min(size.width);
        let x = (size.width.saturating_sub(process_w)) / 2;
        let y = (size.height.saturating_sub(process_h)) / 2;
        let area = Rect::new(x, y, process_w, process_h);
        frame.render_widget(Clear, area);
        let process_box = Block::default()
            .borders(Borders::ALL)
            .title("processos ativos")
            .title_alignment(Alignment::Center)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(theme.accent).bg(theme.bg));
        let inner = process_box.inner(area);
        frame.render_widget(process_box, area);

        let mut lines = Vec::new();
        for (_, summary, started) in &app.running_bash_call_ids {
            let elapsed = started.elapsed().as_secs();
            let elapsed_str = format_elapsed_clock(elapsed);
            lines.push(Line::from(vec![
                Span::styled(
                    "● ",
                    Style::default()
                        .fg(theme.warning)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    short_preview(summary, 60),
                    Style::default().fg(theme.fg),
                ),
                Span::styled(
                    format!("  ({})", elapsed_str),
                    Style::default().fg(theme.muted),
                ),
            ]));
        }
        lines.push(Line::raw(""));
        lines.push(Line::from(vec![Span::styled(
            "K encerrar processo  ·  Esc fechar",
            Style::default().fg(theme.muted),
        )]));
        let process_widget = Paragraph::new(lines)
            .style(Style::default().bg(theme.bg).fg(theme.fg))
            .wrap(Wrap { trim: false });
        frame.render_widget(process_widget, inner);
    }

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
    let token_text = format!("{}% de {}k tokens", pct, max_ctx / 1000);
    let thinking_model_label = if app.thinking_enabled {
        "Raciocínio ligado"
    } else {
        "Raciocínio desligado"
    };
    let output_view_label = if app.output_expanded {
        "expandido"
    } else {
        "recolhido"
    };
    let thinking_hint = format!(
        "({}, {}) (Alt+T / Ctrl+O)",
        thinking_model_label, output_view_label
    );
    let right_width = token_text
        .chars()
        .count()
        .max(thinking_hint.chars().count()) as u16
        + 1;
    let footer_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(10), Constraint::Length(right_width)])
        .split(root[5]);
    let left = Paragraph::new(short_preview(&cwd, 90))
        .style(Style::default().bg(theme.bg).fg(theme.muted));
    frame.render_widget(left, footer_cols[0]);
    let thinking_model_color = if app.thinking_enabled {
        theme.success
    } else {
        theme.warning
    };
    let thinking_view_color = if app.output_expanded {
        theme.success
    } else {
        theme.warning
    };
    let right = Paragraph::new(vec![
        Line::from(vec![Span::styled(
            token_text,
            Style::default().fg(theme.muted),
        )]),
        Line::from(vec![
            Span::styled("(", Style::default().fg(theme.muted)),
            Span::styled(
                thinking_model_label,
                Style::default()
                    .fg(thinking_model_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(", ", Style::default().fg(theme.muted)),
            Span::styled(
                output_view_label,
                Style::default()
                    .fg(thinking_view_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(") ", Style::default().fg(theme.muted)),
            Span::styled("(Alt+T / Ctrl+O)", Style::default().fg(theme.accent)),
        ]),
    ])
    .alignment(Alignment::Right)
    .style(Style::default().bg(theme.bg));
    frame.render_widget(right, footer_cols[1]);
}

pub fn draw_ui(terminal: &mut Terminal<CrosstermBackend<Stdout>>, app: &AppState) -> Result<()> {
    terminal.draw(|f| {
        let theme = UI_THEME;
        let size = f.area();
        match app.screen {
            UiScreen::Welcome => render_welcome_screen(f, app, size, theme),
            UiScreen::ModelSetup => render_model_setup_screen(f, app, size, theme),
            UiScreen::Config => render_chat_screen(f, app, size, theme),
            UiScreen::Chat => render_chat_screen(f, app, size, theme),
        }
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span_text(spans: &[Span<'static>]) -> String {
        spans
            .iter()
            .map(|span| span.content.as_ref())
            .collect::<String>()
    }

    #[test]
    fn inline_markdown_renders_strong_sections() {
        let spans = render_inline_markdown_spans(
            "**Status:** validado",
            Style::default().fg(UI_THEME.fg),
            UI_THEME,
        );
        assert_eq!(span_text(&spans), "Status: validado");
        assert!(spans.iter().any(|span| span.content.as_ref() == "Status:"
            && span.style.add_modifier.contains(Modifier::BOLD)));
    }

    #[test]
    fn inline_markdown_preserves_unmatched_backticks() {
        let spans = render_inline_markdown_spans(
            "Use `grep -n",
            Style::default().fg(UI_THEME.fg),
            UI_THEME,
        );
        assert_eq!(span_text(&spans), "Use `grep -n");
    }

    #[test]
    fn inline_markdown_renders_strikethrough_sections() {
        let spans = render_inline_markdown_spans(
            "~~ignorar~~ manter",
            Style::default().fg(UI_THEME.fg),
            UI_THEME,
        );
        assert_eq!(span_text(&spans), "ignorar manter");
        assert!(spans.iter().any(|span| span.content.as_ref() == "ignorar"
            && span.style.add_modifier.contains(Modifier::CROSSED_OUT)));
    }
}
