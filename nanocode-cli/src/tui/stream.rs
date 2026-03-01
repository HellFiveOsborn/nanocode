#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HiddenStreamState {
    None,
    ThinkTag,
    ThinkBracket,
    ToolCallXml,
}

#[derive(Debug, Clone, Default)]
pub struct StreamParts {
    pub visible: String,
    pub thinking: String,
}

#[derive(Debug, Default)]
pub struct StreamSanitizer {
    pending: String,
    hidden: HiddenStreamState,
}

impl Default for HiddenStreamState {
    fn default() -> Self {
        Self::None
    }
}

fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    haystack
        .to_ascii_lowercase()
        .find(&needle.to_ascii_lowercase())
}

fn split_keep_tail_chars(s: &str, tail_chars: usize) -> (&str, &str) {
    let total_chars = s.chars().count();
    if total_chars <= tail_chars {
        return ("", s);
    }

    let keep_from_char = total_chars - tail_chars;
    let keep_from_byte = s
        .char_indices()
        .nth(keep_from_char)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    (&s[..keep_from_byte], &s[keep_from_byte..])
}

impl StreamSanitizer {
    pub fn is_in_thinking(&self) -> bool {
        matches!(
            self.hidden,
            HiddenStreamState::ThinkTag | HiddenStreamState::ThinkBracket
        )
    }

    pub fn push(&mut self, chunk: &str, finalize: bool) -> StreamParts {
        const START_THINK: &str = "<think>";
        const END_THINK: &str = "</think>";
        const START_THINK_BRACKET: &str = "[start thinking]";
        const END_THINK_BRACKET: &str = "[end thinking]";
        const START_TOOL_XML: &str = "<tool_call>";
        const END_TOOL_XML: &str = "</tool_call>";
        const IM_START_ASSISTANT: &str = "<|im_start|>assistant";
        const IM_END: &str = "<|im_end|>";
        const TAIL_CHARS: usize = 24;

        self.pending.push_str(chunk);
        let mut parts = StreamParts::default();

        loop {
            match self.hidden {
                HiddenStreamState::ThinkTag => {
                    if let Some(end_idx) = find_ci(&self.pending, END_THINK) {
                        parts.thinking.push_str(&self.pending[..end_idx]);
                        let drain_to = end_idx + END_THINK.len();
                        self.pending.drain(..drain_to);
                        self.hidden = HiddenStreamState::None;
                        continue;
                    }

                    if !finalize {
                        let (emit, tail) =
                            split_keep_tail_chars(&self.pending, END_THINK.len() - 1);
                        parts.thinking.push_str(emit);
                        self.pending = tail.to_string();
                    } else {
                        parts.thinking.push_str(&self.pending);
                        self.pending.clear();
                        self.hidden = HiddenStreamState::None;
                    }
                    break;
                }
                HiddenStreamState::ThinkBracket => {
                    if let Some(end_idx) = find_ci(&self.pending, END_THINK_BRACKET) {
                        parts.thinking.push_str(&self.pending[..end_idx]);
                        let drain_to = end_idx + END_THINK_BRACKET.len();
                        self.pending.drain(..drain_to);
                        self.hidden = HiddenStreamState::None;
                        continue;
                    }

                    if !finalize {
                        let (emit, tail) =
                            split_keep_tail_chars(&self.pending, END_THINK_BRACKET.len() - 1);
                        parts.thinking.push_str(emit);
                        self.pending = tail.to_string();
                    } else {
                        parts.thinking.push_str(&self.pending);
                        self.pending.clear();
                        self.hidden = HiddenStreamState::None;
                    }
                    break;
                }
                HiddenStreamState::ToolCallXml => {
                    if let Some(end_idx) = find_ci(&self.pending, END_TOOL_XML) {
                        let drain_to = end_idx + END_TOOL_XML.len();
                        self.pending.drain(..drain_to);
                        self.hidden = HiddenStreamState::None;
                        continue;
                    }

                    if !finalize {
                        let (_, tail) =
                            split_keep_tail_chars(&self.pending, END_TOOL_XML.len() - 1);
                        self.pending = tail.to_string();
                    } else {
                        self.pending.clear();
                        self.hidden = HiddenStreamState::None;
                    }
                    break;
                }
                HiddenStreamState::None => {
                    let mut candidates: Vec<(usize, &'static str, HiddenStreamState)> = Vec::new();

                    if let Some(i) = find_ci(&self.pending, START_THINK) {
                        candidates.push((i, START_THINK, HiddenStreamState::ThinkTag));
                    }
                    if let Some(i) = find_ci(&self.pending, START_THINK_BRACKET) {
                        candidates.push((i, START_THINK_BRACKET, HiddenStreamState::ThinkBracket));
                    }
                    if let Some(i) = find_ci(&self.pending, START_TOOL_XML) {
                        candidates.push((i, START_TOOL_XML, HiddenStreamState::ToolCallXml));
                    }
                    if let Some(i) = find_ci(&self.pending, END_THINK) {
                        candidates.push((i, END_THINK, HiddenStreamState::None));
                    }
                    if let Some(i) = find_ci(&self.pending, END_THINK_BRACKET) {
                        candidates.push((i, END_THINK_BRACKET, HiddenStreamState::None));
                    }
                    if let Some(i) = find_ci(&self.pending, END_TOOL_XML) {
                        candidates.push((i, END_TOOL_XML, HiddenStreamState::None));
                    }
                    if let Some(i) = find_ci(&self.pending, IM_START_ASSISTANT) {
                        candidates.push((i, IM_START_ASSISTANT, HiddenStreamState::None));
                    }
                    if let Some(i) = find_ci(&self.pending, IM_END) {
                        candidates.push((i, IM_END, HiddenStreamState::None));
                    }

                    if let Some((start_idx, marker, next_state)) =
                        candidates.into_iter().min_by_key(|(i, _, _)| *i)
                    {
                        if start_idx > 0 {
                            parts.visible.push_str(&self.pending[..start_idx]);
                        }
                        let drain_to = start_idx + marker.len();
                        self.pending.drain(..drain_to);
                        self.hidden = next_state;
                        continue;
                    }

                    if finalize {
                        parts.visible.push_str(&self.pending);
                        self.pending.clear();
                    } else {
                        let (emit, tail) = split_keep_tail_chars(&self.pending, TAIL_CHARS);
                        parts.visible.push_str(emit);
                        self.pending = tail.to_string();
                    }
                    break;
                }
            }
        }

        parts
    }
}

pub fn extract_thinking_blocks_and_clean(text: &str) -> (Vec<String>, String) {
    let mut thoughts = Vec::new();
    let mut out = text.to_string();

    loop {
        let lower = out.to_ascii_lowercase();
        let Some(start) = lower.find("<think>") else {
            break;
        };

        let Some(end_rel) = lower[start..].find("</think>") else {
            let thought = out[start + "<think>".len()..].trim();
            if !thought.is_empty() {
                thoughts.push(thought.to_string());
            }
            out.truncate(start);
            break;
        };

        let end = start + end_rel;
        let thought = out[start + "<think>".len()..end].trim();
        if !thought.is_empty() {
            thoughts.push(thought.to_string());
        }
        out = format!("{}{}", &out[..start], &out[end + "</think>".len()..]);
    }

    loop {
        let lower = out.to_ascii_lowercase();
        let Some(start) = lower.find("[start thinking]") else {
            break;
        };

        let Some(end_rel) = lower[start..].find("[end thinking]") else {
            let thought = out[start + "[start thinking]".len()..].trim();
            if !thought.is_empty() {
                thoughts.push(thought.to_string());
            }
            out.truncate(start);
            break;
        };

        let end = start + end_rel;
        let thought = out[start + "[start thinking]".len()..end].trim();
        if !thought.is_empty() {
            thoughts.push(thought.to_string());
        }
        out = format!("{}{}", &out[..start], &out[end + "[end thinking]".len()..]);
    }

    while let Some(start) = out.to_ascii_lowercase().find("<tool_call>") {
        if let Some(end_rel) = out.to_ascii_lowercase()[start..].find("</tool_call>") {
            let end = start + end_rel + "</tool_call>".len();
            out.replace_range(start..end, "");
        } else {
            out.truncate(start);
            break;
        }
    }

    (
        thoughts,
        out.replace("<|im_start|>assistant", "")
            .replace("<|im_end|>", "")
            .replace("</think>", "")
            .replace("[end thinking]", "")
            .trim()
            .to_string(),
    )
}
