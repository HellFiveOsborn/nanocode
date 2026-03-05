//! Skill discovery and prompt integration.

use anyhow::{anyhow, Context, Result};
use regex::Regex;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use tracing::{debug, info, warn};

use crate::config::NcConfig;

#[derive(Debug, Clone)]
pub struct SkillInfo {
    pub name: String,
    pub description: String,
    pub skill_path: PathBuf,
    pub user_invocable: bool,
}

#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    name: String,
    description: String,
    #[serde(default = "default_user_invocable", alias = "user-invocable")]
    user_invocable: bool,
}

fn default_user_invocable() -> bool {
    true
}

struct BundledSkill {
    name: &'static str,
    content: &'static str,
}

const BUNDLED_SKILLS: [BundledSkill; 2] = [
    BundledSkill {
        name: "frontend-design",
        content: include_str!("../../skills/frontend-design/SKILL.md"),
    },
    BundledSkill {
        name: "javascript-backend",
        content: include_str!("../../skills/javascript-backend/SKILL.md"),
    },
];

pub fn install_bundled_skills_if_missing() -> Result<usize> {
    let target_root = NcConfig::config_dir().join("skills");
    std::fs::create_dir_all(&target_root).with_context(|| {
        format!(
            "failed to create skills directory at {}",
            target_root.display()
        )
    })?;

    let mut installed = 0usize;
    for skill in BUNDLED_SKILLS {
        let skill_dir = target_root.join(skill.name);
        let skill_file = skill_dir.join("SKILL.md");
        if skill_file.exists() {
            continue;
        }
        std::fs::create_dir_all(&skill_dir).with_context(|| {
            format!("failed to create bundled skill directory {}", skill_dir.display())
        })?;
        std::fs::write(&skill_file, skill.content).with_context(|| {
            format!("failed to write bundled skill file {}", skill_file.display())
        })?;
        installed += 1;
    }

    Ok(installed)
}

pub struct SkillManager {
    search_paths: Vec<PathBuf>,
    discovered: BTreeMap<String, SkillInfo>,
    enabled_patterns: Vec<String>,
    disabled_patterns: Vec<String>,
}

impl SkillManager {
    pub fn new(config: &NcConfig) -> Self {
        let search_paths = compute_search_paths(config);
        let discovered = discover_skills(&search_paths);

        if !discovered.is_empty() {
            info!(
                "discovered {} skill(s) from {} search path(s)",
                discovered.len(),
                search_paths.len()
            );
        }

        Self {
            search_paths,
            discovered,
            enabled_patterns: config.enabled_skills.clone(),
            disabled_patterns: config.disabled_skills.clone(),
        }
    }

    pub fn search_paths(&self) -> &[PathBuf] {
        &self.search_paths
    }

    pub fn available_skills(&self) -> BTreeMap<String, SkillInfo> {
        if !self.enabled_patterns.is_empty() {
            return self
                .discovered
                .iter()
                .filter(|(name, _)| matches_any_pattern(name, &self.enabled_patterns))
                .map(|(name, info)| (name.clone(), info.clone()))
                .collect();
        }

        if !self.disabled_patterns.is_empty() {
            return self
                .discovered
                .iter()
                .filter(|(name, _)| !matches_any_pattern(name, &self.disabled_patterns))
                .map(|(name, info)| (name.clone(), info.clone()))
                .collect();
        }

        self.discovered.clone()
    }

    pub fn skill_count(&self) -> usize {
        self.available_skills().len()
    }

    pub fn get_skill(&self, name: &str) -> Option<SkillInfo> {
        self.available_skills().get(name).cloned()
    }

    pub fn available_skills_prompt_section(&self) -> String {
        let skills = self.available_skills();
        if skills.is_empty() {
            return String::new();
        }

        let mut lines = vec![
            "# Available Skills".to_string(),
            "".to_string(),
            "You have access to the following skills. When a task matches a skill's description,"
                .to_string(),
            "read the full SKILL.md file to load detailed instructions.".to_string(),
            "".to_string(),
            "<available_skills>".to_string(),
        ];

        for (name, info) in skills {
            lines.push("  <skill>".to_string());
            lines.push(format!("    <name>{}</name>", xml_escape(&name)));
            lines.push(format!(
                "    <description>{}</description>",
                xml_escape(&info.description)
            ));
            lines.push(format!(
                "    <path>{}</path>",
                xml_escape(&info.skill_path.display().to_string())
            ));
            lines.push("  </skill>".to_string());
        }

        lines.push("</available_skills>".to_string());
        lines.join("\n")
    }
}

fn compute_search_paths(config: &NcConfig) -> Vec<PathBuf> {
    let mut candidates = Vec::new();

    for configured in &config.skill_paths {
        let expanded = expand_tilde_path(configured);
        if expanded.is_dir() {
            candidates.push(expanded);
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        for relative in [".nanocode/skills", ".agents/skills", ".claude/skills", "skills"] {
            let candidate = cwd.join(relative);
            if candidate.is_dir() {
                candidates.push(candidate);
            }
        }
    }

    let global_skills = NcConfig::config_dir().join("skills");
    if global_skills.is_dir() {
        candidates.push(global_skills);
    }

    dedupe_paths(candidates)
}

fn expand_tilde_path(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if raw == "~" {
        if let Some(home) = dirs::home_dir() {
            return home;
        }
    } else if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    path.to_path_buf()
}

fn dedupe_paths(paths: Vec<PathBuf>) -> Vec<PathBuf> {
    let mut unique: Vec<PathBuf> = Vec::new();
    for path in paths {
        let normalized = path.canonicalize().unwrap_or(path);
        if !unique.iter().any(|p| p == &normalized) {
            unique.push(normalized);
        }
    }
    unique
}

fn discover_skills(search_paths: &[PathBuf]) -> BTreeMap<String, SkillInfo> {
    let mut skills: BTreeMap<String, SkillInfo> = BTreeMap::new();

    for base in search_paths {
        if !base.is_dir() {
            continue;
        }

        let mut skill_dirs: Vec<PathBuf> = Vec::new();
        if base.join("SKILL.md").is_file() {
            skill_dirs.push(base.clone());
        }

        match std::fs::read_dir(base) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() && path.join("SKILL.md").is_file() {
                        skill_dirs.push(path);
                    }
                }
            }
            Err(err) => {
                warn!("failed to read skill directory {}: {}", base.display(), err);
                continue;
            }
        }

        for skill_dir in skill_dirs {
            let skill_file = skill_dir.join("SKILL.md");
            match parse_skill_file(&skill_file) {
                Ok(skill) => {
                    if let Some(existing) = skills.get(&skill.name) {
                        debug!(
                            "skipping duplicate skill '{}' at {} (already loaded from {})",
                            skill.name,
                            skill.skill_path.display(),
                            existing.skill_path.display()
                        );
                    } else {
                        skills.insert(skill.name.clone(), skill);
                    }
                }
                Err(err) => warn!("failed to parse skill {}: {}", skill_file.display(), err),
            }
        }
    }

    skills
}

fn parse_skill_file(skill_path: &Path) -> Result<SkillInfo> {
    let content = std::fs::read_to_string(skill_path)
        .with_context(|| format!("cannot read {}", skill_path.display()))?;
    let frontmatter = extract_frontmatter(&content)?;
    let metadata: SkillFrontmatter =
        serde_yaml::from_str(frontmatter).map_err(|err| anyhow!("invalid frontmatter: {err}"))?;

    if !is_valid_skill_name(&metadata.name) {
        return Err(anyhow!(
            "invalid skill name '{}': expected lowercase letters/numbers/hyphens",
            metadata.name
        ));
    }

    if metadata.name != skill_path.parent().and_then(Path::file_name).and_then(|n| n.to_str()).unwrap_or_default() {
        warn!(
            "skill name '{}' does not match directory name '{}' at {}",
            metadata.name,
            skill_path
                .parent()
                .and_then(Path::file_name)
                .and_then(|n| n.to_str())
                .unwrap_or(""),
            skill_path.display()
        );
    }

    Ok(SkillInfo {
        name: metadata.name,
        description: metadata.description.trim().to_string(),
        skill_path: skill_path
            .canonicalize()
            .unwrap_or_else(|_| skill_path.to_path_buf()),
        user_invocable: metadata.user_invocable,
    })
}

fn extract_frontmatter(content: &str) -> Result<&str> {
    let mut matches = frontmatter_boundary().find_iter(content);
    let Some(first) = matches.next() else {
        return Err(anyhow!(
            "missing YAML frontmatter (expected opening --- at file start)"
        ));
    };
    if first.start() != 0 {
        return Err(anyhow!(
            "invalid YAML frontmatter (opening --- must be at file start)"
        ));
    }
    let Some(second) = matches.next() else {
        return Err(anyhow!(
            "missing YAML frontmatter end boundary (expected closing ---)"
        ));
    };

    Ok(&content[first.end()..second.start()])
}

fn frontmatter_boundary() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?m)^-{3,}\s*$").expect("valid frontmatter regex"))
}

fn is_valid_skill_name(name: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        Regex::new(r"^[a-z0-9]+(?:-[a-z0-9]+)*$").expect("valid skill name regex")
    });
    re.is_match(name)
}

fn matches_any_pattern(name: &str, patterns: &[String]) -> bool {
    patterns
        .iter()
        .filter_map(|pattern| wildcard_to_regex(pattern))
        .any(|regex| regex.is_match(name))
}

fn wildcard_to_regex(pattern: &str) -> Option<Regex> {
    if pattern.trim().is_empty() {
        return None;
    }

    let escaped = regex::escape(pattern.trim()).replace("\\*", ".*");
    Regex::new(&format!("(?i)^{}$", escaped)).ok()
}

fn xml_escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}
