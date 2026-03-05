//! Middleware system

use crate::types::*;
use async_trait::async_trait;
use serde_json::{Map, Value};

/// Action to take after middleware evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MiddlewareAction {
    Continue,
    Stop,
    Compact,
    InjectMessage,
}

/// Reason for middleware reset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResetReason {
    Stop,
    Compact,
}

/// Result returned by middleware before each turn.
#[derive(Debug, Clone)]
pub struct MiddlewareResult {
    pub action: MiddlewareAction,
    pub message: Option<String>,
    pub reason: Option<String>,
    pub metadata: Map<String, Value>,
}

impl Default for MiddlewareResult {
    fn default() -> Self {
        Self {
            action: MiddlewareAction::Continue,
            message: None,
            reason: None,
            metadata: Map::new(),
        }
    }
}

impl MiddlewareResult {
    pub fn stop(reason: impl Into<String>) -> Self {
        Self {
            action: MiddlewareAction::Stop,
            reason: Some(reason.into()),
            ..Self::default()
        }
    }

    pub fn compact(old_tokens: u32, threshold: u32) -> Self {
        let mut metadata = Map::new();
        metadata.insert("old_tokens".to_string(), Value::from(old_tokens));
        metadata.insert("threshold".to_string(), Value::from(threshold));
        Self {
            action: MiddlewareAction::Compact,
            metadata,
            ..Self::default()
        }
    }

    pub fn inject(message: impl Into<String>) -> Self {
        Self {
            action: MiddlewareAction::InjectMessage,
            message: Some(message.into()),
            ..Self::default()
        }
    }
}

/// Middleware trait.
#[async_trait]
pub trait Middleware: Send + Sync {
    /// Called before each turn.
    async fn before_turn(&mut self, ctx: &ConversationContext) -> MiddlewareResult;

    /// Reset middleware state.
    fn reset(&mut self, reason: ResetReason);
}

/// Conversation context for middleware.
#[derive(Debug, Clone)]
pub struct ConversationContext {
    pub messages: Vec<LlmMessage>,
    pub stats: AgentStats,
    pub agent_name: String,
}

/// Turn limit middleware.
pub struct TurnLimitMiddleware {
    max_turns: u32,
}

impl TurnLimitMiddleware {
    pub fn new(max_turns: u32) -> Self {
        Self { max_turns }
    }
}

#[async_trait]
impl Middleware for TurnLimitMiddleware {
    async fn before_turn(&mut self, ctx: &ConversationContext) -> MiddlewareResult {
        if ctx.stats.turns >= self.max_turns {
            MiddlewareResult::stop(format!("Turn limit of {} reached", self.max_turns))
        } else {
            MiddlewareResult::default()
        }
    }

    fn reset(&mut self, _reason: ResetReason) {}
}

/// Auto-compact middleware.
pub struct AutoCompactMiddleware {
    threshold: u32,
}

impl AutoCompactMiddleware {
    pub fn new(threshold: u32) -> Self {
        Self { threshold }
    }
}

#[async_trait]
impl Middleware for AutoCompactMiddleware {
    async fn before_turn(&mut self, ctx: &ConversationContext) -> MiddlewareResult {
        if ctx.stats.context_tokens >= self.threshold {
            MiddlewareResult::compact(ctx.stats.context_tokens, self.threshold)
        } else {
            MiddlewareResult::default()
        }
    }

    fn reset(&mut self, _reason: ResetReason) {}
}

/// Context warning middleware.
pub struct ContextWarningMiddleware {
    threshold: u32,
    warned: bool,
}

impl ContextWarningMiddleware {
    pub fn new(threshold: u32) -> Self {
        Self {
            threshold,
            warned: false,
        }
    }
}

#[async_trait]
impl Middleware for ContextWarningMiddleware {
    async fn before_turn(&mut self, ctx: &ConversationContext) -> MiddlewareResult {
        let warning_threshold = self.threshold / 2;
        if self.warned || warning_threshold == 0 {
            return MiddlewareResult::default();
        }

        if ctx.stats.context_tokens >= warning_threshold {
            self.warned = true;
            let pct = ((ctx.stats.context_tokens as f64 / self.threshold as f64) * 100.0)
                .round()
                .clamp(0.0, 100.0) as u32;
            return MiddlewareResult::inject(format!(
                "[system] context warning: {}% of auto-compact threshold in use ({} / {} tokens)",
                pct, ctx.stats.context_tokens, self.threshold
            ));
        }

        MiddlewareResult::default()
    }

    fn reset(&mut self, _reason: ResetReason) {
        self.warned = false;
    }
}

/// Reminder middleware for `plan` agent profile.
pub struct PlanAgentMiddleware {
    was_plan_agent: bool,
    reminder_message: String,
    exit_message: String,
}

impl Default for PlanAgentMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl PlanAgentMiddleware {
    pub fn new() -> Self {
        Self {
            was_plan_agent: false,
            reminder_message: "[system] Plan mode active: do not execute mutating tools or edit files until the user asks to execute.".to_string(),
            exit_message: "[system] Plan mode ended. To implement, switch to Build (Tab or /agent build).".to_string(),
        }
    }

    fn is_plan(agent_name: &str) -> bool {
        agent_name.trim().eq_ignore_ascii_case("plan")
    }
}

#[async_trait]
impl Middleware for PlanAgentMiddleware {
    async fn before_turn(&mut self, ctx: &ConversationContext) -> MiddlewareResult {
        let is_plan_now = Self::is_plan(&ctx.agent_name);

        if self.was_plan_agent && !is_plan_now {
            self.was_plan_agent = false;
            return MiddlewareResult::inject(self.exit_message.clone());
        }

        if !self.was_plan_agent && is_plan_now {
            self.was_plan_agent = true;
            return MiddlewareResult::inject(self.reminder_message.clone());
        }

        self.was_plan_agent = is_plan_now;
        MiddlewareResult::default()
    }

    fn reset(&mut self, _reason: ResetReason) {
        self.was_plan_agent = false;
    }
}

/// Ordered middleware pipeline.
#[derive(Default)]
pub struct MiddlewarePipeline {
    middlewares: Vec<Box<dyn Middleware>>,
}

impl MiddlewarePipeline {
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    pub fn add<M>(&mut self, middleware: M) -> &mut Self
    where
        M: Middleware + 'static,
    {
        self.middlewares.push(Box::new(middleware));
        self
    }

    pub fn clear(&mut self) {
        self.middlewares.clear();
    }

    pub fn reset(&mut self, reason: ResetReason) {
        for mw in &mut self.middlewares {
            mw.reset(reason);
        }
    }

    pub async fn run_before_turn(&mut self, ctx: &ConversationContext) -> MiddlewareResult {
        let mut injected_messages: Vec<String> = Vec::new();

        for mw in &mut self.middlewares {
            let result = mw.before_turn(ctx).await;
            match result.action {
                MiddlewareAction::InjectMessage => {
                    if let Some(message) = result.message {
                        let trimmed = message.trim();
                        if !trimmed.is_empty() {
                            injected_messages.push(trimmed.to_string());
                        }
                    }
                }
                MiddlewareAction::Stop | MiddlewareAction::Compact => return result,
                MiddlewareAction::Continue => {}
            }
        }

        if !injected_messages.is_empty() {
            return MiddlewareResult::inject(injected_messages.join("\n\n"));
        }

        MiddlewareResult::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context(turns: u32, context_tokens: u32, agent_name: &str) -> ConversationContext {
        ConversationContext {
            messages: Vec::new(),
            stats: AgentStats {
                turns,
                context_tokens,
                ..AgentStats::default()
            },
            agent_name: agent_name.to_string(),
        }
    }

    #[tokio::test]
    async fn turn_limit_stops_when_limit_reached() {
        let mut mw = TurnLimitMiddleware::new(2);
        let result = mw.before_turn(&context(2, 100, "default")).await;
        assert_eq!(result.action, MiddlewareAction::Stop);
        assert!(result.reason.unwrap_or_default().contains("Turn limit"));
    }

    #[tokio::test]
    async fn auto_compact_triggers_on_context_threshold() {
        let mut mw = AutoCompactMiddleware::new(1000);
        let result = mw.before_turn(&context(1, 1200, "default")).await;
        assert_eq!(result.action, MiddlewareAction::Compact);
        assert_eq!(
            result
                .metadata
                .get("old_tokens")
                .and_then(Value::as_u64)
                .unwrap_or_default(),
            1200
        );
    }

    #[tokio::test]
    async fn context_warning_triggers_once_and_resets() {
        let mut mw = ContextWarningMiddleware::new(1000);

        let first = mw.before_turn(&context(0, 600, "default")).await;
        assert_eq!(first.action, MiddlewareAction::InjectMessage);

        let second = mw.before_turn(&context(0, 700, "default")).await;
        assert_eq!(second.action, MiddlewareAction::Continue);

        mw.reset(ResetReason::Stop);

        let third = mw.before_turn(&context(0, 700, "default")).await;
        assert_eq!(third.action, MiddlewareAction::InjectMessage);
    }

    #[tokio::test]
    async fn plan_agent_middleware_injects_entry_and_exit_messages() {
        let mut mw = PlanAgentMiddleware::new();

        let noop = mw.before_turn(&context(0, 100, "default")).await;
        assert_eq!(noop.action, MiddlewareAction::Continue);

        let enter = mw.before_turn(&context(0, 100, "plan")).await;
        assert_eq!(enter.action, MiddlewareAction::InjectMessage);

        let in_plan = mw.before_turn(&context(0, 100, "plan")).await;
        assert_eq!(in_plan.action, MiddlewareAction::Continue);

        let exit = mw.before_turn(&context(0, 100, "default")).await;
        assert_eq!(exit.action, MiddlewareAction::InjectMessage);
    }
}
