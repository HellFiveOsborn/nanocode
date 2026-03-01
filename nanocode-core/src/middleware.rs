//! Middleware system

use crate::types::*;
use async_trait::async_trait;

/// Middleware result
pub type MiddlewareResult = Result<MiddlewareAction, String>;

/// Action to take after middleware
#[derive(Debug)]
pub enum MiddlewareAction {
    Continue,
    Stop,
    Compact,
    InjectMessage(String),
}

/// Reason for middleware reset
#[derive(Debug, Clone, Copy)]
pub enum ResetReason {
    Compact,
    NewSession,
    Error,
}

/// Middleware trait
#[async_trait]
pub trait Middleware: Send + Sync {
    /// Called before each turn
    async fn before_turn(&mut self, ctx: &ConversationContext) -> MiddlewareResult;

    /// Reset the middleware
    fn reset(&mut self, reason: ResetReason);
}

/// Conversation context for middleware
pub struct ConversationContext {
    pub messages: Vec<LlmMessage>,
    pub stats: AgentStats,
    pub agent_name: String,
}

/// Turn limit middleware
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
            Err("Turn limit reached".to_string())
        } else {
            Ok(MiddlewareAction::Continue)
        }
    }

    fn reset(&mut self, _reason: ResetReason) {}
}

/// Auto-compact middleware
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
        if ctx.stats.tokens_used >= self.threshold {
            Ok(MiddlewareAction::Compact)
        } else {
            Ok(MiddlewareAction::Continue)
        }
    }

    fn reset(&mut self, _reason: ResetReason) {}
}

/// Context warning middleware
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
        let half_threshold = self.threshold / 2;

        if ctx.stats.tokens_used >= half_threshold && !self.warned {
            self.warned = true;
            Ok(MiddlewareAction::InjectMessage(format!(
                "[System: Context is at {}% of threshold]",
                (ctx.stats.tokens_used as f32 / self.threshold as f32 * 100.0) as u32
            )))
        } else {
            Ok(MiddlewareAction::Continue)
        }
    }

    fn reset(&mut self, _reason: ResetReason) {
        self.warned = false;
    }
}
