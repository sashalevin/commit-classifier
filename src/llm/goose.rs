// SPDX-License-Identifier: GPL-2.0
// (c) 2026, Yunseong Kim <yunseong.kim@est.tech>

//! goose CLI provider.
//!
//! Runs `goose run --no-session --quiet --output-format json --instructions -`,
//! feeds the prompt on stdin and collects the assistant text from the JSON
//! transcript goose prints on stdout. goose owns the model backend, so any
//! provider goose supports works here, including OpenAI-compatible local
//! servers such as vLLM (`GOOSE_PROVIDER=openai OPENAI_HOST=...`).

use async_trait::async_trait;
use log::debug;
use serde_json::Value;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::llm::provider::{BaseLlmConfig, LlmConfig};
use crate::llm::{Llm, LlmError};

/// Default upper bound for one `goose run`. Agent runs that use tools can
/// take a while, so this is more generous than a single API round trip.
const DEFAULT_TIMEOUT_SECS: u64 = 600;

/// goose CLI configuration
#[derive(Debug, Clone)]
pub struct GooseConfig {
    pub base: BaseLlmConfig,
    /// Path to the goose binary
    pub binary: PathBuf,
    /// goose provider (`--provider`), e.g. "openai", "anthropic", "ollama".
    /// When unset goose falls back to GOOSE_PROVIDER or its config file.
    pub provider: Option<String>,
    /// goose model (`--model`). Falls back to GOOSE_MODEL or the config file.
    pub model: Option<String>,
    /// Builtin extensions to enable (`--with-builtin`), e.g. "developer"
    /// to let the agent run git in the kernel tree. Empty by default.
    pub builtins: Vec<String>,
    /// Pass `--no-profile` so the user's default extensions are not loaded.
    pub no_profile: bool,
    /// Agent turn limit (`--max-turns`)
    pub max_turns: Option<u64>,
    /// Working directory for the goose process
    pub cwd: Option<PathBuf>,
    /// Extra environment for the goose process (OPENAI_HOST, GOOSE_MODE, ...)
    pub env: BTreeMap<String, String>,
    /// Forward base.temperature as GOOSE_TEMPERATURE
    pub set_temperature: bool,
    /// Forward base.max_tokens as GOOSE_MAX_TOKENS
    pub set_max_tokens: bool,
    /// Timeout for a single goose run
    pub timeout: Duration,
}

impl Default for GooseConfig {
    fn default() -> Self {
        Self {
            base: BaseLlmConfig::default(),
            binary: PathBuf::from("goose"),
            provider: None,
            model: None,
            builtins: Vec::new(),
            no_profile: true,
            max_turns: None,
            cwd: None,
            env: BTreeMap::new(),
            set_temperature: false,
            set_max_tokens: false,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
        }
    }
}

fn json_str(json: &Value, key: &str) -> Option<String> {
    json.get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
}

/// Accept either a JSON array of names or a comma-separated string.
fn json_list(value: Option<&Value>) -> Vec<String> {
    let items: Vec<String> = match value {
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(Value::as_str)
            .map(String::from)
            .collect(),
        Some(Value::String(s)) => s.split(',').map(String::from).collect(),
        _ => Vec::new(),
    };
    items
        .into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

impl LlmConfig for GooseConfig {
    fn from_json(json: Option<Value>) -> Result<Self, LlmError> {
        let json = json.unwrap_or(Value::Object(serde_json::Map::new()));
        let base = BaseLlmConfig::from_json(&json);
        let defaults = Self::default();

        let binary = json_str(&json, "binary")
            .or_else(|| json_str(&json, "executable_path"))
            .map(PathBuf::from)
            .unwrap_or(defaults.binary);

        let builtins = json_list(json.get("builtins").or_else(|| json.get("with_builtin")));

        let mut env = BTreeMap::new();
        if let Some(obj) = json.get("env").and_then(Value::as_object) {
            for (k, v) in obj {
                let v = match v {
                    Value::String(s) => s.clone(),
                    Value::Null => continue,
                    other => other.to_string(),
                };
                env.insert(k.clone(), v);
            }
        }
        // GOOSE_MODE is common enough to deserve a top-level key
        if let Some(mode) = json_str(&json, "mode") {
            env.insert("GOOSE_MODE".to_string(), mode);
        }

        let timeout = json
            .get("timeout_secs")
            .and_then(Value::as_u64)
            .filter(|&s| s > 0)
            .map(Duration::from_secs)
            .unwrap_or(defaults.timeout);

        Ok(Self {
            base,
            binary,
            provider: json_str(&json, "provider"),
            model: json_str(&json, "model"),
            builtins,
            no_profile: json
                .get("no_profile")
                .and_then(Value::as_bool)
                .unwrap_or(defaults.no_profile),
            max_turns: json.get("max_turns").and_then(Value::as_u64),
            cwd: json_str(&json, "cwd").map(PathBuf::from),
            env,
            // Only override goose's own sampling settings when asked to
            set_temperature: json.get("temperature").is_some(),
            set_max_tokens: json.get("max_tokens").is_some(),
            timeout,
        })
    }
}

/// goose LLM implementation driving `goose run`
pub struct GooseLlm {
    config: GooseConfig,
}

impl GooseLlm {
    pub fn new(config: GooseConfig) -> Result<Self, LlmError> {
        Ok(Self { config })
    }

    /// Arguments for `goose run`, prompt excluded (it goes over stdin)
    fn build_args(&self) -> Vec<String> {
        let c = &self.config;
        let mut args: Vec<String> = [
            "run",
            "--no-session",
            "--quiet",
            "--output-format",
            "json",
            "--instructions",
            "-",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();

        if c.no_profile {
            args.push("--no-profile".into());
        }
        if !c.builtins.is_empty() {
            args.push("--with-builtin".into());
            args.push(c.builtins.join(","));
        }
        if let Some(provider) = &c.provider {
            args.push("--provider".into());
            args.push(provider.clone());
        }
        if let Some(model) = &c.model {
            args.push("--model".into());
            args.push(model.clone());
        }
        if let Some(max_turns) = c.max_turns {
            args.push("--max-turns".into());
            args.push(max_turns.to_string());
        }
        args
    }

    /// Environment overrides for the goose process
    fn build_env(&self) -> BTreeMap<String, String> {
        let c = &self.config;
        let mut env = BTreeMap::new();
        if c.set_temperature {
            env.insert("GOOSE_TEMPERATURE".into(), c.base.temperature.to_string());
        }
        if c.set_max_tokens {
            env.insert("GOOSE_MAX_TOKENS".into(), c.base.max_tokens.to_string());
        }
        // Explicit env entries win over the derived ones
        env.extend(c.env.clone());
        env
    }

    async fn invoke_once(&self, prompt: &str) -> Result<String, LlmError> {
        let args = self.build_args();
        if self.config.base.debug_logging {
            debug!("goose: {} {}", self.config.binary.display(), args.join(" "));
            debug!("goose: prompt length {} chars", prompt.len());
        }

        let mut cmd = Command::new(&self.config.binary);
        cmd.args(&args)
            .envs(self.build_env())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        if let Some(cwd) = &self.config.cwd {
            cmd.current_dir(cwd);
        }

        let mut child = cmd.spawn().map_err(|e| {
            LlmError::Config(format!(
                "Failed to spawn {}: {e}. Is goose installed?",
                self.config.binary.display()
            ))
        })?;

        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| LlmError::Config("goose stdin unavailable".into()))?;
        // Write from a separate task so a large prompt cannot deadlock against
        // goose filling the stdout pipe before it has read all of stdin
        let prompt_owned = prompt.to_string();
        let writer = tokio::spawn(async move {
            stdin.write_all(prompt_owned.as_bytes()).await?;
            stdin.shutdown().await
        });

        let output = child
            .wait_with_output()
            .await
            .map_err(|e| LlmError::Api(format!("Failed to wait for goose: {e}")))?;

        if let Ok(Err(e)) = writer.await {
            debug!("goose: writing prompt to stdin failed: {e}");
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        for line in stderr.lines() {
            debug!("goose stderr: {line}");
        }

        match parse_goose_output(&stdout) {
            Ok(text) if output.status.success() => {
                debug!("goose: response length {} chars", text.len());
                Ok(text)
            }
            Ok(_) => Err(LlmError::Api(format!(
                "goose exited with {}: {}",
                output.status,
                tail(&stderr)
            ))),
            Err(e) if output.status.success() => Err(e),
            Err(e) => {
                let detail = if stderr.trim().is_empty() {
                    e.to_string()
                } else {
                    tail(&stderr)
                };
                Err(LlmError::Api(format!(
                    "goose exited with {}: {detail}",
                    output.status
                )))
            }
        }
    }
}

#[async_trait]
impl Llm for GooseLlm {
    async fn invoke(&self, prompt: &str) -> Result<String, LlmError> {
        let max_retries = self.config.base.max_retries;
        let mut delay_ms = self.config.base.initial_backoff_ms;
        let timeout = self.config.timeout;

        for attempt in 0..=max_retries {
            match tokio::time::timeout(timeout, self.invoke_once(prompt)).await {
                Ok(Ok(s)) => return Ok(s),
                // Missing binary / bad path won't fix itself on retry
                Ok(Err(LlmError::Config(e))) => return Err(LlmError::Config(e)),
                Ok(Err(e)) if attempt < max_retries => {
                    debug!("goose: attempt {attempt} failed: {e}; retrying in {delay_ms}ms");
                }
                Ok(Err(e)) => return Err(e),
                Err(_) if attempt < max_retries => {
                    debug!(
                        "goose: attempt {attempt} timed out after {}s; retrying in {delay_ms}ms",
                        timeout.as_secs()
                    );
                }
                Err(_) => {
                    return Err(LlmError::Api(format!(
                        "goose timed out after {}s ({} attempts)",
                        timeout.as_secs(),
                        max_retries + 1
                    )));
                }
            }
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            delay_ms = delay_ms.saturating_mul(2);
        }
        unreachable!("retry loop exits via return")
    }
}

/// Last few lines of a diagnostic stream, for error messages
fn tail(s: &str) -> String {
    let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty()).collect();
    let start = lines.len().saturating_sub(5);
    lines[start..].join(" | ")
}

/// Extract the assistant's answer from goose's `--output-format json` output.
///
/// The transcript is `{"messages": [...], "metadata": {"status": ...}}`.
/// Text parts of every assistant message are joined in order, so an agent
/// that talks between tool calls still ends with its final verdict. Thinking
/// parts are kept as `<think>` blocks placed before the text of their message;
/// tool request and tool response parts are dropped.
fn parse_goose_output(stdout: &str) -> Result<String, LlmError> {
    let doc: Value = serde_json::from_str(stdout.trim())
        .or_else(|_| {
            // Tolerate stray log lines around the JSON document
            let start = stdout.find('{').ok_or(())?;
            let end = stdout.rfind('}').ok_or(())?;
            serde_json::from_str(&stdout[start..=end]).map_err(|_| ())
        })
        .map_err(|_| {
            LlmError::Api(format!(
                "goose output is not JSON: {}",
                stdout.chars().take(200).collect::<String>()
            ))
        })?;

    let messages = doc
        .get("messages")
        .and_then(Value::as_array)
        .ok_or_else(|| LlmError::Api("goose output has no messages array".into()))?;

    let mut out = Vec::new();
    let mut has_text = false;
    let mut errors = Vec::new();
    for msg in messages {
        if msg.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(parts) = msg.get("content").and_then(Value::as_array) else {
            continue;
        };
        let mut thoughts = Vec::new();
        let mut texts = Vec::new();
        for part in parts {
            let field = |key: &str| {
                part.get(key)
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|t| !t.is_empty())
                    .map(String::from)
            };
            match part.get("type").and_then(Value::as_str) {
                Some("text") => texts.extend(field("text")),
                // Reasoning models (Qwen3, DeepSeek-R1, ...) put their analysis
                // here and leave little more than the verdict in the text part.
                // Keep it as a <think> block ahead of the text: verdict parsers
                // already skip those, and saved explanations stay useful
                Some("thinking") => {
                    thoughts.extend(field("thinking").map(|t| format!("<think>\n{t}\n</think>")))
                }
                Some("error") => errors.extend(field("message")),
                _ => {}
            }
        }
        has_text |= !texts.is_empty();
        out.extend(thoughts);
        out.extend(texts);
    }

    let status = doc
        .get("metadata")
        .and_then(|m| m.get("status"))
        .and_then(Value::as_str);

    if !errors.is_empty() || status == Some("error") {
        let detail = if errors.is_empty() {
            "status=error".to_string()
        } else {
            errors.join(" | ")
        };
        return Err(LlmError::Api(format!("goose reported an error: {detail}")));
    }
    if !has_text {
        return Err(LlmError::Api("goose returned no assistant text".into()));
    }
    Ok(out.join("\n\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_goose_config_defaults() {
        let config = GooseConfig::from_json(None).unwrap();
        assert_eq!(config.binary, PathBuf::from("goose"));
        assert!(config.provider.is_none());
        assert!(config.model.is_none());
        assert!(config.builtins.is_empty());
        assert!(config.no_profile);
        assert!(config.cwd.is_none());
        assert!(!config.set_temperature);
        assert!(!config.set_max_tokens);
        assert_eq!(config.timeout, Duration::from_secs(DEFAULT_TIMEOUT_SECS));

        let llm = GooseLlm::new(config).unwrap();
        assert_eq!(
            llm.build_args(),
            [
                "run",
                "--no-session",
                "--quiet",
                "--output-format",
                "json",
                "--instructions",
                "-",
                "--no-profile"
            ]
        );
        assert!(llm.build_env().is_empty());
    }

    #[test]
    fn test_goose_config_from_json() {
        let json = json!({
            "executable_path": "/opt/goose",
            "provider": "openai",
            "model": "OpenVINO/Qwen3-8B-int4-ov",
            "builtins": "developer, todo",
            "no_profile": false,
            "max_turns": 20,
            "cwd": "/src/linux",
            "mode": "chat",
            "env": {"OPENAI_HOST": "http://localhost:8000", "GOOSE_TOOLSHIM": true},
            "temperature": 0.2,
            "timeout_secs": 30
        });
        let config = GooseConfig::from_json(Some(json)).unwrap();
        assert_eq!(config.binary, PathBuf::from("/opt/goose"));
        assert_eq!(config.builtins, ["developer", "todo"]);
        assert_eq!(config.cwd, Some(PathBuf::from("/src/linux")));
        assert_eq!(config.timeout, Duration::from_secs(30));

        let llm = GooseLlm::new(config).unwrap();
        let args = llm.build_args();
        assert!(!args.contains(&"--no-profile".to_string()));
        let joined = args.join(" ");
        assert!(joined.contains("--with-builtin developer,todo"));
        assert!(joined.contains("--provider openai"));
        assert!(joined.contains("--model OpenVINO/Qwen3-8B-int4-ov"));
        assert!(joined.contains("--max-turns 20"));

        let env = llm.build_env();
        assert_eq!(env["OPENAI_HOST"], "http://localhost:8000");
        assert_eq!(env["GOOSE_TOOLSHIM"], "true");
        assert_eq!(env["GOOSE_MODE"], "chat");
        assert_eq!(env["GOOSE_TEMPERATURE"], "0.2");
        assert!(!env.contains_key("GOOSE_MAX_TOKENS"));
    }

    #[test]
    fn test_parse_goose_output_joins_assistant_text() {
        let out = json!({
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "prompt **NO**"}]},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hmm **NO**"},
                    {"type": "text", "text": "Let me check the tree."},
                    {"type": "toolRequest", "id": "1"}
                ]},
                {"role": "user", "content": [{"type": "toolResponse", "id": "1"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "\n\nFixes a UAF.\n\n**YES**"}]}
            ],
            "metadata": {"status": "completed"}
        });
        let text = parse_goose_output(&out.to_string()).unwrap();
        assert_eq!(
            text,
            "<think>\nhmm **NO**\n</think>\n\nLet me check the tree.\n\nFixes a UAF.\n\n**YES**"
        );
    }

    #[test]
    fn test_parse_goose_output_thinking_before_text() {
        // goose emits the thinking part after the streamed text; it is moved
        // in front so the verdict stays on the last line
        let out = json!({
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "text", "text": "\n\n**YES**"},
                    {"type": "thinking", "thinking": "Phase 1 ... looks like a UAF fix"}
                ]}
            ],
            "metadata": {"status": "completed"}
        });
        let text = parse_goose_output(&out.to_string()).unwrap();
        assert_eq!(
            text,
            "<think>\nPhase 1 ... looks like a UAF fix\n</think>\n\n**YES**"
        );

        // Thinking alone is not an answer
        let out = json!({
            "messages": [
                {"role": "assistant", "content": [{"type": "thinking", "thinking": "hmm"}]}
            ],
            "metadata": {"status": "completed"}
        });
        assert!(parse_goose_output(&out.to_string()).is_err());
    }

    #[test]
    fn test_parse_goose_output_error() {
        let out = json!({
            "messages": [
                {"role": "assistant", "content": [
                    {"type": "error", "message": "Request failed: Bad request (400)"}
                ]}
            ],
            "metadata": {"status": "error"}
        });
        let err = parse_goose_output(&out.to_string()).unwrap_err();
        assert!(err.to_string().contains("Bad request (400)"));
    }

    #[test]
    fn test_parse_goose_output_tolerates_noise_and_rejects_empty() {
        let doc = json!({
            "messages": [{"role": "assistant", "content": [{"type": "text", "text": "**NO**"}]}],
            "metadata": {"status": "completed"}
        });
        let noisy = format!("starting session\n{doc}\n");
        assert_eq!(parse_goose_output(&noisy).unwrap(), "**NO**");

        let empty = json!({"messages": [], "metadata": {"status": "completed"}});
        assert!(parse_goose_output(&empty.to_string()).is_err());
        assert!(parse_goose_output("not json").is_err());
    }

    #[tokio::test]
    async fn test_goose_invoke_with_fake_binary() {
        use std::os::unix::fs::PermissionsExt;

        // Stand-in for goose: checks argv and env, echoes stdin back as the
        // assistant text in goose's JSON transcript format
        let dir = tempfile::tempdir().unwrap();
        let script = dir.path().join("goose");
        std::fs::write(
            &script,
            r#"#!/bin/sh
[ "$1" = run ] || exit 3
case "$*" in *"--instructions -"*) ;; *) exit 4 ;; esac
[ "$OPENAI_HOST" = "http://localhost:8000" ] || exit 5
prompt=$(cat | tr -d '"\\')
printf '{"messages":[{"role":"assistant","content":[{"type":"text","text":"%s in %s\\n\\n**YES**"}]}],"metadata":{"status":"completed"}}\n' "$prompt" "$(basename "$PWD")"
"#,
        )
        .unwrap();
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();

        let workdir = dir.path().join("linux");
        std::fs::create_dir(&workdir).unwrap();

        let config = GooseConfig::from_json(Some(json!({
            "binary": script,
            "cwd": workdir,
            "env": {"OPENAI_HOST": "http://localhost:8000"},
            "max_retries": 0
        })))
        .unwrap();
        let llm = GooseLlm::new(config).unwrap();
        let out = llm.invoke("analyze this").await.unwrap();
        assert_eq!(out, "analyze this in linux\n\n**YES**");
    }

    #[tokio::test]
    async fn test_goose_missing_binary_is_config_error() {
        let config = GooseConfig::from_json(Some(json!({
            "binary": "/nonexistent/goose",
            "max_retries": 2
        })))
        .unwrap();
        let llm = GooseLlm::new(config).unwrap();
        match llm.invoke("x").await {
            Err(LlmError::Config(msg)) => assert!(msg.contains("Is goose installed")),
            other => panic!("expected config error, got {other:?}"),
        }
    }
}
