use base64::engine::general_purpose::STANDARD as BASE64_STD;
use base64::Engine;
use std::collections::HashSet;
use std::ffi::OsStr;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

const MAX_INDEX_ENTRIES: usize = 80_000;
const MAX_SUGGESTIONS: usize = 8;
const MAX_TEXT_BYTES: usize = 256 * 1024;
const MAX_TEXT_LINES: usize = 1_200;
const MAX_LISTED_DIRECTORY_ENTRIES: usize = 200;
const MAX_SEARCH_MATCHES: usize = 120;
const MAX_IMAGE_BYTES: usize = 6 * 1024 * 1024;
const INDEX_REFRESH_EVERY: Duration = Duration::from_secs(4);
const MAX_TOTAL_CONTEXT_CHARS: usize = 360_000;

#[derive(Clone)]
pub struct MentionSuggestion {
    pub replacement: String,
    pub display: String,
    pub description: String,
    pub is_directory: bool,
}

pub struct MentionCompletion {
    pub suggestions: Vec<MentionSuggestion>,
    pub replace_range: Option<(usize, usize)>,
}

pub struct AttachmentExpansion {
    pub prompt_for_model: String,
    pub status_lines: Vec<String>,
    pub errors: Vec<String>,
    pub image_data_urls: Vec<String>,
}

#[derive(Clone)]
struct IndexEntry {
    rel: String,
    rel_lower: String,
    name: String,
    name_lower: String,
    is_dir: bool,
    depth: usize,
}

#[derive(Clone)]
struct ParsedMention {
    raw: String,
    path_part: String,
    query: Option<String>,
}

#[derive(Clone)]
enum ResolvedKind {
    File,
    Directory,
}

#[derive(Clone)]
struct ResolvedMention {
    path: PathBuf,
    rel_display: String,
    query: Option<String>,
    kind: ResolvedKind,
    source: String,
}

struct ActiveMention {
    path_query: String,
    replace_range: (usize, usize),
}

pub struct AttachmentEngine {
    root: PathBuf,
    index: Vec<IndexEntry>,
    last_index_refresh: Instant,
}

impl AttachmentEngine {
    pub fn new(root: PathBuf) -> Self {
        let canonical_root = fs::canonicalize(&root).unwrap_or(root);
        let mut engine = Self {
            root: canonical_root,
            index: Vec::new(),
            last_index_refresh: Instant::now()
                .checked_sub(INDEX_REFRESH_EVERY)
                .unwrap_or_else(Instant::now),
        };
        engine.refresh_index_if_needed(true);
        engine
    }

    pub fn complete_for_input(&mut self, input: &str) -> MentionCompletion {
        let Some(active) = active_mention_at_end(input) else {
            return MentionCompletion {
                suggestions: Vec::new(),
                replace_range: None,
            };
        };
        self.refresh_index_if_needed(false);

        let typed = normalize_slashes(active.path_query.as_str());
        let suggestions = self.suggest_paths(&typed);
        MentionCompletion {
            suggestions,
            replace_range: Some(active.replace_range),
        }
    }

    pub fn expand_prompt(&mut self, prompt: String) -> AttachmentExpansion {
        self.refresh_index_if_needed(false);
        let parsed = parse_mentions(&prompt);
        if parsed.is_empty() {
            return AttachmentExpansion {
                prompt_for_model: prompt,
                status_lines: Vec::new(),
                errors: Vec::new(),
                image_data_urls: Vec::new(),
            };
        }

        let mut resolved: Vec<ResolvedMention> = Vec::new();
        let mut errors = Vec::new();
        let mut seen = HashSet::<String>::new();

        for mention in parsed {
            match self.resolve_mention(&mention) {
                Ok(Some(item)) => {
                    let key = format!(
                        "{}|{}",
                        item.path.display(),
                        item.query.as_deref().unwrap_or("")
                    );
                    if seen.insert(key) {
                        resolved.push(item);
                    }
                }
                Ok(None) => {
                    errors.push(format!(
                        "Anexo ignorado: não encontrei `{}` no workspace.",
                        mention.path_part
                    ));
                }
                Err(err) => errors.push(err),
            }
        }

        if resolved.is_empty() {
            return AttachmentExpansion {
                prompt_for_model: prompt,
                status_lines: Vec::new(),
                errors,
                image_data_urls: Vec::new(),
            };
        }

        let mut total_chars = 0usize;
        let mut context_blocks: Vec<String> = Vec::new();
        let mut status_lines: Vec<String> = Vec::new();
        let mut image_data_urls: Vec<String> = Vec::new();

        for item in resolved {
            match self.render_attachment(&item) {
                Ok(AttachmentRender::Text { block, status }) => {
                    if total_chars + block.len() > MAX_TOTAL_CONTEXT_CHARS {
                        status_lines
                            .push("⎿ Contexto anexado atingiu limite e foi truncado.".into());
                        break;
                    }
                    total_chars += block.len();
                    status_lines.push(format!("⎿ {}", status));
                    context_blocks.push(block);
                }
                Ok(AttachmentRender::Image {
                    data_url,
                    status,
                    prompt_hint,
                }) => {
                    image_data_urls.push(data_url);
                    status_lines.push(format!("⎿ {}", status));
                    context_blocks.push(prompt_hint);
                }
                Err(err) => errors.push(err),
            }
        }

        let cleaned_prompt = strip_attachment_prefix_lines(&prompt);
        let final_prompt = if context_blocks.is_empty() {
            cleaned_prompt
        } else {
            format!(
                "## Contexto anexado\n\n{}\n\n## Solicitação do usuário\n{}",
                context_blocks.join("\n\n"),
                cleaned_prompt.trim()
            )
        };

        AttachmentExpansion {
            prompt_for_model: final_prompt,
            status_lines,
            errors,
            image_data_urls,
        }
    }

    fn refresh_index_if_needed(&mut self, force: bool) {
        if !force && self.last_index_refresh.elapsed() < INDEX_REFRESH_EVERY {
            return;
        }
        self.index = build_index(&self.root, MAX_INDEX_ENTRIES);
        self.last_index_refresh = Instant::now();
    }

    fn suggest_paths(&self, typed_path: &str) -> Vec<MentionSuggestion> {
        if self.index.is_empty() {
            return Vec::new();
        }

        let query = typed_path.trim();
        if query.is_empty() {
            return self
                .index
                .iter()
                .filter(|e| e.depth == 1)
                .filter(|e| !e.name.starts_with('.'))
                .take(MAX_SUGGESTIONS)
                .map(to_suggestion)
                .collect();
        }

        let normalized = query.to_ascii_lowercase();
        let ends_with_sep = query.ends_with('/') || query.ends_with('\\');
        if ends_with_sep {
            let prefix = normalized.trim_end_matches('/').trim_end_matches('\\');
            return self
                .index
                .iter()
                .filter(|e| e.rel_lower.starts_with(prefix))
                .filter(|e| immediate_child_after_prefix(&e.rel_lower, prefix))
                .take(MAX_SUGGESTIONS)
                .map(to_suggestion)
                .collect();
        }

        let mut scored: Vec<(&IndexEntry, i32)> = self
            .index
            .iter()
            .filter_map(|entry| score_entry(&normalized, entry).map(|score| (entry, score)))
            .collect();

        scored.sort_by(|(a_entry, a_score), (b_entry, b_score)| {
            b_score
                .cmp(a_score)
                .then_with(|| b_entry.is_dir.cmp(&a_entry.is_dir))
                .then_with(|| a_entry.depth.cmp(&b_entry.depth))
                .then_with(|| a_entry.rel.cmp(&b_entry.rel))
        });

        scored
            .into_iter()
            .take(MAX_SUGGESTIONS)
            .map(|(entry, _)| to_suggestion(entry))
            .collect()
    }

    fn resolve_mention(&self, mention: &ParsedMention) -> Result<Option<ResolvedMention>, String> {
        let candidate = mention.path_part.trim();
        if candidate.is_empty() {
            return Ok(None);
        }

        if let Some(resolved) = self.resolve_path_exact(candidate)? {
            return Ok(Some(ResolvedMention {
                query: mention.query.clone(),
                source: mention.raw.clone(),
                ..resolved
            }));
        }

        let normalized = normalize_slashes(candidate);
        let mut scored: Vec<(&IndexEntry, i32)> = self
            .index
            .iter()
            .filter_map(|entry| {
                score_entry(&normalized.to_ascii_lowercase(), entry).map(|s| (entry, s))
            })
            .collect();
        scored.sort_by(|(a_entry, a_score), (b_entry, b_score)| {
            b_score
                .cmp(a_score)
                .then_with(|| b_entry.is_dir.cmp(&a_entry.is_dir))
                .then_with(|| a_entry.rel.cmp(&b_entry.rel))
        });

        let Some((best, best_score)) = scored.into_iter().next() else {
            return Ok(None);
        };
        if best_score < 20 {
            return Ok(None);
        }

        let path = self.root.join(&best.rel);
        Ok(Some(ResolvedMention {
            path,
            rel_display: with_dir_suffix(&best.rel, best.is_dir),
            query: mention.query.clone(),
            kind: if best.is_dir {
                ResolvedKind::Directory
            } else {
                ResolvedKind::File
            },
            source: mention.raw.clone(),
        }))
    }

    fn resolve_path_exact(&self, raw: &str) -> Result<Option<ResolvedMention>, String> {
        let path = PathBuf::from(raw);
        let absolute = if path.is_absolute() {
            path
        } else {
            self.root.join(path)
        };

        let canonical = match fs::canonicalize(&absolute) {
            Ok(v) => v,
            Err(_) => return Ok(None),
        };

        if !canonical.starts_with(&self.root) {
            return Err(format!(
                "Anexo bloqueado: `{}` está fora do diretório de trabalho.",
                raw
            ));
        }

        let rel = canonical
            .strip_prefix(&self.root)
            .ok()
            .map(path_to_slash)
            .unwrap_or_else(|| raw.to_string());
        let kind = if canonical.is_dir() {
            ResolvedKind::Directory
        } else {
            ResolvedKind::File
        };

        Ok(Some(ResolvedMention {
            path: canonical,
            rel_display: with_dir_suffix(&rel, matches!(kind, ResolvedKind::Directory)),
            query: None,
            kind,
            source: raw.to_string(),
        }))
    }

    fn render_attachment(&self, mention: &ResolvedMention) -> Result<AttachmentRender, String> {
        match mention.kind {
            ResolvedKind::Directory => self.render_directory_attachment(mention),
            ResolvedKind::File => self.render_file_attachment(mention),
        }
    }

    fn render_directory_attachment(
        &self,
        mention: &ResolvedMention,
    ) -> Result<AttachmentRender, String> {
        if let Some(query) = mention.query.as_deref() {
            let matches = search_in_directory(&mention.path, query, MAX_SEARCH_MATCHES);
            let body = if matches.is_empty() {
                "Nenhuma ocorrência encontrada.".to_string()
            } else {
                matches.join("\n")
            };
            let block = format!(
                "### Diretório {}\n```text\n{}\n```",
                mention.rel_display, body
            );
            let status = format!(
                "Buscou \"{}\" em {} ({} ocorrência(s))",
                query,
                mention.rel_display,
                matches.len()
            );
            return Ok(AttachmentRender::Text { block, status });
        }

        let mut entries = match fs::read_dir(&mention.path) {
            Ok(read_dir) => read_dir
                .flatten()
                .map(|entry| {
                    let is_dir = entry.path().is_dir();
                    let mut name = entry.file_name().to_string_lossy().to_string();
                    if is_dir {
                        name.push('/');
                    }
                    name
                })
                .collect::<Vec<_>>(),
            Err(err) => {
                return Err(format!(
                    "Falha ao listar diretório `{}`: {}",
                    mention.rel_display, err
                ))
            }
        };
        entries.sort();
        let overflow = entries.len().saturating_sub(MAX_LISTED_DIRECTORY_ENTRIES);
        entries.truncate(MAX_LISTED_DIRECTORY_ENTRIES);
        if overflow > 0 {
            entries.push(format!("... (+{} entradas)", overflow));
        }

        let block = format!(
            "### Diretório {}\n```text\n{}\n```",
            mention.rel_display,
            entries.join("\n")
        );
        let status = format!("Listou diretório {}", mention.rel_display);
        Ok(AttachmentRender::Text { block, status })
    }

    fn render_file_attachment(
        &self,
        mention: &ResolvedMention,
    ) -> Result<AttachmentRender, String> {
        let bytes = fs::read(&mention.path)
            .map_err(|err| format!("Falha ao ler `{}`: {}", mention.rel_display, err))?;

        if is_probably_binary(&bytes) {
            if let Some(mime) = detect_image_mime(&mention.path) {
                if bytes.len() > MAX_IMAGE_BYTES {
                    return Err(format!(
                        "Imagem `{}` excede {} MB.",
                        mention.rel_display,
                        MAX_IMAGE_BYTES / (1024 * 1024)
                    ));
                }
                let data_url = format!("data:{};base64,{}", mime, BASE64_STD.encode(&bytes));
                let status = format!("Anexou imagem {}", mention.rel_display);
                let prompt_hint = format!(
                    "### Imagem {}\nArquivo anexado como imagem para o modelo de visão.",
                    mention.rel_display
                );
                return Ok(AttachmentRender::Image {
                    data_url,
                    status,
                    prompt_hint,
                });
            }
            return Err(format!(
                "Arquivo `{}` parece binário e foi ignorado.",
                mention.rel_display
            ));
        }

        let mut text = String::from_utf8(bytes).map_err(|_| {
            format!(
                "Arquivo `{}` não está em UTF-8 e foi ignorado.",
                mention.rel_display
            )
        })?;

        if text.len() > MAX_TEXT_BYTES {
            text.truncate(MAX_TEXT_BYTES);
            text.push_str("\n... [truncado por tamanho]");
        }

        let lines: Vec<&str> = text.lines().collect();
        let rendered = if let Some(query) = mention.query.as_deref() {
            render_query_lines(&lines, query, MAX_SEARCH_MATCHES)
        } else {
            render_head_lines(&lines, MAX_TEXT_LINES)
        };
        let status = if let Some(query) = mention.query.as_deref() {
            format!(
                "Leu {} com filtro \"{}\" ({})",
                mention.rel_display,
                query,
                source_tag(mention)
            )
        } else {
            format!("Leu {} ({} linhas)", mention.rel_display, lines.len())
        };
        let block = format!(
            "### Arquivo {}\n```text\n{}\n```",
            mention.rel_display, rendered
        );
        Ok(AttachmentRender::Text { block, status })
    }
}

enum AttachmentRender {
    Text {
        block: String,
        status: String,
    },
    Image {
        data_url: String,
        status: String,
        prompt_hint: String,
    },
}

fn source_tag(mention: &ResolvedMention) -> String {
    if mention.source == mention.rel_display {
        "match exato".to_string()
    } else {
        format!("via `{}`", mention.source)
    }
}

fn detect_image_mime(path: &Path) -> Option<&'static str> {
    match path
        .extension()
        .and_then(OsStr::to_str)
        .map(|v| v.to_ascii_lowercase())
    {
        Some(ext) if ext == "png" => Some("image/png"),
        Some(ext) if ext == "jpg" || ext == "jpeg" => Some("image/jpeg"),
        Some(ext) if ext == "webp" => Some("image/webp"),
        Some(ext) if ext == "gif" => Some("image/gif"),
        Some(ext) if ext == "bmp" => Some("image/bmp"),
        _ => None,
    }
}

fn render_head_lines(lines: &[&str], max_lines: usize) -> String {
    let mut out: Vec<String> = lines
        .iter()
        .take(max_lines)
        .enumerate()
        .map(|(idx, line)| format!("{:>6} {}", idx + 1, line))
        .collect();
    if lines.len() > max_lines {
        out.push(format!("... (+{} linhas)", lines.len() - max_lines));
    }
    out.join("\n")
}

fn render_query_lines(lines: &[&str], query: &str, max_matches: usize) -> String {
    let query_lower = query.to_ascii_lowercase();
    let mut out = Vec::new();
    for (idx, line) in lines.iter().enumerate() {
        if line.to_ascii_lowercase().contains(&query_lower) {
            out.push(format!("{:>6} {}", idx + 1, line));
            if out.len() >= max_matches {
                out.push(format!("... (+mais ocorrências para \"{}\")", query));
                break;
            }
        }
    }
    if out.is_empty() {
        "Nenhuma ocorrência encontrada.".to_string()
    } else {
        out.join("\n")
    }
}

fn search_in_directory(root: &Path, query: &str, max_matches: usize) -> Vec<String> {
    let mut matches = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    let needle = query.to_ascii_lowercase();

    while let Some(dir) = stack.pop() {
        let read_dir = match fs::read_dir(&dir) {
            Ok(v) => v,
            Err(_) => continue,
        };

        for entry in read_dir.flatten() {
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if file_type.is_symlink() {
                continue;
            }
            if path.is_dir() {
                if should_skip_dir_name(entry.file_name().to_string_lossy().as_ref()) {
                    continue;
                }
                stack.push(path);
                continue;
            }

            let Ok(bytes) = fs::read(&path) else {
                continue;
            };
            if is_probably_binary(&bytes) || bytes.is_empty() || bytes.len() > MAX_TEXT_BYTES {
                continue;
            }
            let Ok(text) = String::from_utf8(bytes) else {
                continue;
            };

            for (line_idx, line) in text.lines().enumerate() {
                if line.to_ascii_lowercase().contains(&needle) {
                    matches.push(format!(
                        "{}:{:>4} {}",
                        path_to_slash(path.strip_prefix(root).unwrap_or(&path)),
                        line_idx + 1,
                        line
                    ));
                    if matches.len() >= max_matches {
                        return matches;
                    }
                }
            }
        }
    }

    matches
}

fn parse_mentions(text: &str) -> Vec<ParsedMention> {
    let mut mentions = Vec::new();
    let mut i = 0usize;
    let bytes = text.as_bytes();

    while i < bytes.len() {
        if bytes[i] != b'@' {
            i += 1;
            continue;
        }

        if i > 0 {
            let prev = text[..i].chars().next_back().unwrap_or(' ');
            if prev.is_ascii_alphanumeric() || prev == '_' || prev == '`' {
                i += 1;
                continue;
            }
        }

        let Some((token, end)) = read_mention_token(text, i + 1) else {
            i += 1;
            continue;
        };

        let token_trimmed = token.trim();
        if token_trimmed.is_empty() {
            i = end;
            continue;
        }

        let (path_part, query) = match token_trimmed.split_once('|') {
            Some((path, q)) => {
                let q = q.trim();
                (
                    path.trim().to_string(),
                    if q.is_empty() {
                        None
                    } else {
                        Some(q.to_string())
                    },
                )
            }
            None => (token_trimmed.to_string(), None),
        };

        if !path_part.is_empty() {
            mentions.push(ParsedMention {
                raw: format!("@{}", token_trimmed),
                path_part,
                query,
            });
        }

        i = end;
    }

    mentions
}

fn read_mention_token(text: &str, start_idx: usize) -> Option<(String, usize)> {
    if start_idx >= text.len() {
        return None;
    }
    let mut out = String::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut pos = chars.iter().position(|(byte, _)| *byte == start_idx)?;

    let quote = chars[pos].1;
    if quote == '"' || quote == '\'' {
        pos += 1;
        while pos < chars.len() {
            let (byte_idx, ch) = chars[pos];
            if ch == quote {
                return Some((out, byte_idx + ch.len_utf8()));
            }
            out.push(ch);
            pos += 1;
        }
        return None;
    }

    while pos < chars.len() {
        let (byte_idx, ch) = chars[pos];
        if !is_mention_char(ch) {
            return Some((out, byte_idx));
        }
        out.push(ch);
        pos += 1;
    }

    Some((out, text.len()))
}

fn is_mention_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric()
        || matches!(
            ch,
            '.' | '_' | '-' | '/' | '\\' | '(' | ')' | '[' | ']' | '{' | '}' | ':' | '|' | '#'
        )
}

fn active_mention_at_end(input: &str) -> Option<ActiveMention> {
    if input.is_empty() {
        return None;
    }
    let at_idx = input.rfind('@')?;

    if at_idx > 0 {
        let prev = input[..at_idx].chars().next_back().unwrap_or(' ');
        if prev.is_ascii_alphanumeric() || prev == '_' || prev == '`' {
            return None;
        }
    }

    let tail = &input[at_idx + 1..];
    if tail.chars().any(char::is_whitespace) {
        return None;
    }

    let path_end_rel = tail.find('|').unwrap_or(tail.len());
    let path_part = &tail[..path_end_rel];
    Some(ActiveMention {
        path_query: path_part.to_string(),
        replace_range: (at_idx + 1, at_idx + 1 + path_end_rel),
    })
}

fn to_suggestion(entry: &IndexEntry) -> MentionSuggestion {
    let replacement = with_dir_suffix(&entry.rel, entry.is_dir);
    MentionSuggestion {
        replacement: replacement.clone(),
        display: format!("@{}", replacement),
        description: entry.rel.clone(),
        is_directory: entry.is_dir,
    }
}

fn immediate_child_after_prefix(path: &str, prefix: &str) -> bool {
    let normalized = prefix.trim_matches('/');
    if normalized.is_empty() {
        return path.split('/').count() == 1;
    }
    let with_sep = format!("{}/", normalized);
    if !path.starts_with(&with_sep) {
        return false;
    }
    let rest = &path[with_sep.len()..];
    !rest.is_empty() && !rest.contains('/')
}

fn score_entry(query: &str, entry: &IndexEntry) -> Option<i32> {
    if query.is_empty() {
        return Some(0);
    }
    if !query.starts_with('.') && entry.name.starts_with('.') {
        return None;
    }

    let mut score = 0i32;
    if entry.rel_lower == query {
        score += 4000;
    }
    if entry.rel_lower.starts_with(query) {
        score += 2800;
    }
    if entry.name_lower.starts_with(query) {
        score += 2300;
    }

    let fuzzy = fuzzy_subsequence_score(query, &entry.rel_lower)?;
    score += fuzzy;
    score -= entry.depth as i32 * 6;
    score -= entry.rel.len() as i32 / 3;
    if entry.is_dir {
        score += 20;
    }
    Some(score)
}

fn fuzzy_subsequence_score(query: &str, text: &str) -> Option<i32> {
    if query.is_empty() {
        return Some(0);
    }
    let mut score = 0i32;
    let mut q_chars = query.chars();
    let mut current = q_chars.next()?;
    let mut last_match = None::<usize>;

    for (idx, ch) in text.chars().enumerate() {
        if ch != current {
            continue;
        }
        score += 10;
        if idx == 0 {
            score += 8;
        }
        if matches!(
            text.chars().nth(idx.saturating_sub(1)),
            Some('/' | '_' | '-' | '.')
        ) {
            score += 6;
        }
        if let Some(prev) = last_match {
            if idx == prev + 1 {
                score += 12;
            } else {
                score -= ((idx - prev) as i32).min(4);
            }
        }
        last_match = Some(idx);
        match q_chars.next() {
            Some(next) => current = next,
            None => return Some(score),
        }
    }

    None
}

fn build_index(root: &Path, limit: usize) -> Vec<IndexEntry> {
    let mut entries = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let read_dir = match fs::read_dir(&dir) {
            Ok(v) => v,
            Err(_) => continue,
        };

        for entry in read_dir.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().to_string();
            let file_type = match entry.file_type() {
                Ok(v) => v,
                Err(_) => continue,
            };
            if file_type.is_symlink() {
                continue;
            }
            let is_dir = file_type.is_dir();

            if is_dir && should_skip_dir_name(&name) {
                continue;
            }
            if should_skip_component_name(&name) {
                continue;
            }

            let Ok(rel) = path.strip_prefix(root) else {
                continue;
            };
            if rel.components().any(|c| matches!(c, Component::ParentDir)) {
                continue;
            }

            let rel_str = path_to_slash(rel);
            if rel_str.is_empty() {
                continue;
            }
            let depth = rel.components().count();
            entries.push(IndexEntry {
                rel: rel_str.clone(),
                rel_lower: rel_str.to_ascii_lowercase(),
                name: name.clone(),
                name_lower: name.to_ascii_lowercase(),
                is_dir,
                depth,
            });

            if entries.len() >= limit {
                break;
            }
            if is_dir {
                stack.push(path);
            }
        }
        if entries.len() >= limit {
            break;
        }
    }

    entries.sort_by(|a, b| a.rel.cmp(&b.rel));
    entries
}

fn should_skip_dir_name(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | "target"
            | "node_modules"
            | ".next"
            | ".idea"
            | ".vscode"
            | ".venv"
            | "venv"
            | "__pycache__"
    )
}

fn should_skip_component_name(name: &str) -> bool {
    name == ".DS_Store"
}

fn normalize_slashes(input: &str) -> String {
    input.replace('\\', "/")
}

fn with_dir_suffix(path: &str, is_dir: bool) -> String {
    if is_dir && !path.ends_with('/') {
        format!("{}/", path)
    } else {
        path.to_string()
    }
}

fn path_to_slash(path: &Path) -> String {
    let mut out = Vec::new();
    for part in path.components() {
        if let Component::Normal(p) = part {
            out.push(p.to_string_lossy().to_string());
        }
    }
    out.join("/")
}

fn is_probably_binary(bytes: &[u8]) -> bool {
    if bytes.is_empty() {
        return false;
    }
    if bytes.iter().take(1024).any(|b| *b == 0) {
        return true;
    }

    let control = bytes
        .iter()
        .take(2048)
        .filter(|b| {
            let c = **b;
            c < 9 || (c > 13 && c < 32)
        })
        .count();
    control > 16
}

fn strip_attachment_prefix_lines(prompt: &str) -> String {
    let mut lines: Vec<&str> = prompt.lines().collect();
    while let Some(first) = lines.first() {
        let trimmed = first.trim();
        if trimmed.is_empty() {
            lines.remove(0);
            continue;
        }
        if trimmed == "---" {
            lines.remove(0);
            continue;
        }
        if line_is_only_mentions(trimmed) {
            lines.remove(0);
            continue;
        }
        break;
    }
    lines.join("\n")
}

fn line_is_only_mentions(line: &str) -> bool {
    line.split_whitespace()
        .all(|token| token.starts_with('@') || token.starts_with("\\@"))
}

#[cfg(test)]
mod tests {
    use super::{active_mention_at_end, parse_mentions};

    #[test]
    fn parse_mentions_with_query() {
        let out = parse_mentions("@README.md|setup e @src/main.rs");
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].path_part, "README.md");
        assert_eq!(out[0].query.as_deref(), Some("setup"));
        assert_eq!(out[1].path_part, "src/main.rs");
        assert_eq!(out[1].query, None);
    }

    #[test]
    fn detects_active_mention_with_pipe() {
        let input = "verifique @nanocode-cli/src/tui.rs|submit";
        let ctx = active_mention_at_end(input).expect("mention");
        assert_eq!(ctx.path_query, "nanocode-cli/src/tui.rs");
        assert!(ctx.replace_range.0 < ctx.replace_range.1);
    }

    #[test]
    fn no_active_mention_after_space() {
        assert!(active_mention_at_end("teste @foo bar").is_none());
    }
}
