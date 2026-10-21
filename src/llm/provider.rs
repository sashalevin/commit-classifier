// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use crate::llm::{Llm, LlmError};
use serde_json::Value;
use std::fmt;

/// Base configuration for all LLM providers
#[derive(Debug, Clone)]
pub struct BaseLlmConfig {
    pub temperature: f32,
    pub max_tokens: usize,
    pub debug_logging: bool,
    pub max_retries: usize,
    pub initial_backoff_ms: u64,
}

impl Default for BaseLlmConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            max_tokens: 4000,
            debug_logging: false,
            max_retries: 3,
            initial_backoff_ms: 1000,
        }
    }
}

impl BaseLlmConfig {
    /// Parse common configuration from JSON
    pub fn from_json(json: &Value) -> Self {
        let temperature = json.get("temperature")
            .and_then(|v| v.as_f64())
            .map(|v| v as f32)
            .unwrap_or_else(|| Self::default().temperature);

        let max_tokens = json.get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or_else(|| Self::default().max_tokens);

        let debug_logging = json.get("debug_logging")
            .and_then(|v| v.as_bool())
            .unwrap_or_else(|| Self::default().debug_logging);

        let max_retries = json.get("max_retries")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or_else(|| Self::default().max_retries);

        let initial_backoff_ms = json.get("initial_backoff_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or_else(|| Self::default().initial_backoff_ms);

        Self {
            temperature,
            max_tokens,
            debug_logging,
            max_retries,
            initial_backoff_ms,
        }
    }
}

/// Common trait for LLM configurations
pub trait LlmConfig: Sized {
    /// Create configuration from JSON
    fn from_json(json: Option<Value>) -> Result<Self, LlmError>;
}

/// Enum for supported LLM providers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LlmProvider {
    Claude,
    OpenAI,
    Nvidia,
    Ollama,
    Exec,
    Kiro,
    Goose,
}

impl fmt::Display for LlmProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmProvider::Claude => write!(f, "claude"),
            LlmProvider::OpenAI => write!(f, "openai"),
            LlmProvider::Nvidia => write!(f, "nvidia"),
            LlmProvider::Ollama => write!(f, "ollama"),
            LlmProvider::Exec => write!(f, "exec"),
            LlmProvider::Kiro => write!(f, "kiro"),
            LlmProvider::Goose => write!(f, "goose"),
        }
    }
}

impl LlmProvider {
    pub fn from_name(name: &str) -> Result<Self, LlmError> {
        match name.to_lowercase().as_str() {
            "claude" => Ok(LlmProvider::Claude),
            "openai" | "gpt4" | "gpt4o" => Ok(LlmProvider::OpenAI),
            "nvidia" => Ok(LlmProvider::Nvidia),
            "ollama" => Ok(LlmProvider::Ollama),
            "exec" => Ok(LlmProvider::Exec),
            "kiro" => Ok(LlmProvider::Kiro),
            "goose" => Ok(LlmProvider::Goose),
            _ => Err(LlmError::Config(format!("Unsupported LLM provider: {}", name))),
        }
    }
}

/// Factory for creating LLM instances
pub struct LlmFactory {
    provider: LlmProvider,
}

impl LlmFactory {
    pub fn new(provider: LlmProvider) -> Self {
        Self { provider }
    }

    pub fn from_provider_name(name: &str) -> Result<Self, LlmError> {
        let provider = LlmProvider::from_name(name)?;
        Ok(Self::new(provider))
    }

    pub fn create_llm(&self, config: Option<Value>) -> Result<Box<dyn Llm>, LlmError> {
        match self.provider {
            LlmProvider::Claude => {
                use crate::llm::claude::{ClaudeConfig, ClaudeLlm};
                let config = ClaudeConfig::from_json(config)?;
                Ok(Box::new(ClaudeLlm::new(config)?))
            },
            LlmProvider::OpenAI => {
                use crate::llm::openai::{OpenAIConfig, OpenAILlm};
                let config = OpenAIConfig::from_json(config)?;
                Ok(Box::new(OpenAILlm::new(config)?))
            },
            LlmProvider::Nvidia => {
                use crate::llm::nvidia::{NvidiaConfig, NvidiaLlm};
                let config = NvidiaConfig::from_json(config)?;
                Ok(Box::new(NvidiaLlm::new(config)?))
            },
            LlmProvider::Ollama => {
                use crate::llm::ollama::{OllamaConfig, OllamaLlm};
                let config = OllamaConfig::from_json(config)?;
                Ok(Box::new(OllamaLlm::new(config)?))
            },
            LlmProvider::Exec => {
                use crate::llm::exec::{ExecConfig, ExecLlm};
                let config = ExecConfig::from_json(config)?;
                Ok(Box::new(ExecLlm::new(config)?))
            },
            LlmProvider::Kiro => {
                use crate::llm::kiro::{KiroConfig, KiroLlm};
                let config = KiroConfig::from_json(config)?;
                Ok(Box::new(KiroLlm::new(config)?))
            },
            LlmProvider::Goose => {
                use crate::llm::goose::{GooseConfig, GooseLlm};
                let config = GooseConfig::from_json(config)?;
                Ok(Box::new(GooseLlm::new(config)?))
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_provider_display() {
        assert_eq!(format!("{}", LlmProvider::Claude), "claude");
        assert_eq!(format!("{}", LlmProvider::OpenAI), "openai");
        assert_eq!(format!("{}", LlmProvider::Nvidia), "nvidia");
        assert_eq!(format!("{}", LlmProvider::Ollama), "ollama");
        assert_eq!(format!("{}", LlmProvider::Exec), "exec");
        assert_eq!(format!("{}", LlmProvider::Kiro), "kiro");
        assert_eq!(format!("{}", LlmProvider::Goose), "goose");
    }

    #[test]
    fn test_llm_provider_from_name() {
        // Test valid providers
        assert_eq!(LlmProvider::from_name("claude").unwrap(), LlmProvider::Claude);
        assert_eq!(LlmProvider::from_name("openai").unwrap(), LlmProvider::OpenAI);
        assert_eq!(LlmProvider::from_name("nvidia").unwrap(), LlmProvider::Nvidia);
        assert_eq!(LlmProvider::from_name("ollama").unwrap(), LlmProvider::Ollama);
        assert_eq!(LlmProvider::from_name("exec").unwrap(), LlmProvider::Exec);
        assert_eq!(LlmProvider::from_name("kiro").unwrap(), LlmProvider::Kiro);
        assert_eq!(LlmProvider::from_name("goose").unwrap(), LlmProvider::Goose);

        // Test case insensitivity
        assert_eq!(LlmProvider::from_name("Claude").unwrap(), LlmProvider::Claude);
        assert_eq!(LlmProvider::from_name("OPENAI").unwrap(), LlmProvider::OpenAI);
        assert_eq!(LlmProvider::from_name("NvIdIa").unwrap(), LlmProvider::Nvidia);
        assert_eq!(LlmProvider::from_name("OlLaMa").unwrap(), LlmProvider::Ollama);
        assert_eq!(LlmProvider::from_name("ExEc").unwrap(), LlmProvider::Exec);
        assert_eq!(LlmProvider::from_name("Kiro").unwrap(), LlmProvider::Kiro);
        assert_eq!(LlmProvider::from_name("GoOsE").unwrap(), LlmProvider::Goose);

        // Test aliases
        assert_eq!(LlmProvider::from_name("gpt4").unwrap(), LlmProvider::OpenAI);
        assert_eq!(LlmProvider::from_name("gpt4o").unwrap(), LlmProvider::OpenAI);

        // Test that llama and llama3 are no longer Ollama aliases
        let llama_result = LlmProvider::from_name("llama");
        assert!(llama_result.is_err());
        let llama3_result = LlmProvider::from_name("llama3");
        assert!(llama3_result.is_err());

        // Test invalid provider
        let result = LlmProvider::from_name("unknown");
        assert!(result.is_err());
        match result {
            Err(LlmError::Config(msg)) => {
                assert!(msg.contains("Unsupported LLM provider"));
            },
            _ => panic!("Expected Config error"),
        }
    }

    #[test]
    fn test_llm_factory_new() {
        let factory = LlmFactory::new(LlmProvider::Claude);
        assert_eq!(factory.provider, LlmProvider::Claude);
    }

    #[test]
    fn test_llm_factory_from_provider_name() {
        // Test valid provider
        let result = LlmFactory::from_provider_name("claude");
        assert!(result.is_ok());
        let factory = result.unwrap();
        assert_eq!(factory.provider, LlmProvider::Claude);

        // Test invalid provider
        let result = LlmFactory::from_provider_name("unknown");
        assert!(result.is_err());
    }
}
