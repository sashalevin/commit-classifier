// SPDX-License-Identifier: GPL-2.0
// (c) 2025, Sasha Levin <sashal@kernel.org>

use std::error::Error;
use std::fmt;
use async_trait::async_trait;
use serde::Serialize;

use crate::http::HttpError;

// Sub-modules
mod provider;
mod openai;
mod claude;
mod claude_model;
mod nvidia;
mod ollama;
mod client;
mod response;
mod models;
mod exec;
mod kiro;
mod goose;

// Re-exports
pub use provider::LlmFactory;
pub use response::{ResponseExtractor, NvidiaResponse};
pub use models::{ModelType, ModelHelper};

// Common error type for LLM operations
#[derive(Debug)]
pub enum LlmError {
    Api(String),
    Config(String),
    Http(HttpError),
}

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LlmError::Api(msg) => write!(f, "API error: {}", msg),
            LlmError::Config(msg) => write!(f, "Configuration error: {}", msg),
            LlmError::Http(err) => write!(f, "HTTP error: {}", err),
        }
    }
}

impl Error for LlmError {}

impl From<HttpError> for LlmError {
    fn from(err: HttpError) -> Self {
        LlmError::Http(err)
    }
}

impl From<reqwest::Error> for LlmError {
    fn from(err: reqwest::Error) -> Self {
        LlmError::Http(HttpError::from(err))
    }
}

// Message structure that can be used across providers
#[derive(Serialize, Clone, Debug)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn user(content: String) -> Self {
        Self {
            role: "user".to_string(),
            content,
        }
    }
}

// Universal request for LLM API calls
#[derive(Serialize, Debug)]
pub struct UniversalLlmRequest {
    // Common required fields
    pub model: String,
    pub messages: Vec<Message>,
    pub temperature: f32,
    pub max_tokens: usize,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
}

impl UniversalLlmRequest {
    pub fn new(model: String, messages: Vec<Message>, temperature: f32, max_tokens: usize) -> Self {
        Self {
            model,
            messages,
            temperature,
            max_tokens,
            top_p: None,
            stream: None,
        }
    }

    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn with_stream(mut self, stream: bool) -> Self {
        self.stream = Some(stream);
        self
    }
}


// Base LLM trait
#[async_trait]
pub trait Llm: Send + Sync {
    async fn invoke(&self, prompt: &str) -> Result<String, LlmError>;
}

// Factory function to create an LLM based on provider name
pub fn create_llm(provider: &str, config: Option<serde_json::Value>) -> Result<Box<dyn Llm>, LlmError> {
    let factory = LlmFactory::from_provider_name(provider)?;
    factory.create_llm(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llm_error_display() {
        let api_err = LlmError::Api("API rate limit exceeded".to_string());
        let config_err = LlmError::Config("Missing API key".to_string());
        let http_err = LlmError::Http(HttpError::InvalidHeader("Invalid header name".to_string()));

        assert_eq!(format!("{}", api_err), "API error: API rate limit exceeded");
        assert_eq!(format!("{}", config_err), "Configuration error: Missing API key");
        assert!(format!("{}", http_err).contains("HTTP error: Invalid header"));
    }

    #[test]
    fn test_message_user() {
        let content = "Hello, world!".to_string();
        let message = Message::user(content.clone());

        assert_eq!(message.role, "user");
        assert_eq!(message.content, content);
    }

    #[test]
    fn test_create_llm_invalid_provider() {
        // Try to create an LLM with an invalid provider name
        let result = create_llm("invalid_provider", None);

        assert!(result.is_err());
        if let Err(err) = result {
            match err {
                LlmError::Config(msg) => {
                    // The actual error message comes from LlmProvider::from_name and contains "Unsupported LLM provider"
                    assert!(msg.contains("Unsupported LLM provider"));
                },
                _ => panic!("Expected Config error"),
            }
        }
    }

    #[test]
    fn test_response_handling() {
        // Tests moved to response.rs module
    }
}
