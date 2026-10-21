# commit-classifier

A Rust library for classifying git commits with large language models and
embeddings. It provides the LLM provider abstraction, the embeddings model
wrapper and the vector store used by
[autosel](https://github.com/sashalevin/autosel), which decides whether Linux
kernel commits should be backported to stable trees.

The code was originally part of autosel and was split out so the classification
machinery could be reused independently, so autosel's history contains earlier
copies of these files.

## Providers

| Provider | Selector | Credentials |
|---|---|---|
| Anthropic Claude | `claude` | `ANTHROPIC_API_KEY` |
| OpenAI | `openai` | `OPENAI_API_KEY` |
| NVIDIA | `nvidia` | `NVIDIA_API_KEY` |
| Kiro CLI | `kiro` | `KIRO_API_KEY`, or an authenticated `kiro-cli` session |
| goose CLI | `goose` | whatever the goose provider needs (see below) |
| Ollama | `ollama` | none, talks to a local server |
| Exec | `exec` | none, runs a local binary as the model |

### goose

The `goose` provider runs `goose run --no-session --quiet --output-format json`
with the prompt on stdin and returns the assistant's text. goose picks the
model backend, so its usual environment applies (`GOOSE_PROVIDER`,
`GOOSE_MODEL`, `OPENAI_HOST`, ...). An OpenAI-compatible local server such as
vLLM works through goose's `openai` provider:

```
GOOSE_PROVIDER=openai GOOSE_MODEL=OpenVINO/Qwen3-8B-int4-ov \
OPENAI_HOST=http://localhost:8000 OPENAI_API_KEY=unused ...
```

goose always offers the model at least one tool, so the server must accept
tool calls (`vllm serve ... --enable-auto-tool-choice --tool-call-parser hermes`
for Qwen3). For a server without tool calling set `GOOSE_TOOLSHIM=true`.

JSON config keys, all optional: `binary` (default `goose`), `provider`,
`model`, `builtins` (extensions passed to `--with-builtin`, e.g.
`["developer"]` to let the agent run shell commands; none by default),
`no_profile` (default `true`, so the user's default extensions are not
loaded), `max_turns`, `cwd`, `mode` (sets `GOOSE_MODE`), `env` (extra
environment for the goose process) and `timeout_secs` (default 600).
`temperature` and `max_tokens` are forwarded as `GOOSE_TEMPERATURE` and
`GOOSE_MAX_TOKENS` only when set explicitly.

## Embeddings

`HuggingFaceEmbeddings` wraps [candle_embed](https://github.com/ShelbyJenkins/candle_embed)
and defaults to `BAAI/bge-large-en-v1.5` (1024 dimensions). The model is
downloaded from HuggingFace on first use.

## Building

```
cargo build
cargo test
```

The test suite is 74 unit tests. None of them need network access or API keys;
credentials are only used for real classification runs.

Optional CUDA support for embeddings:

```
cargo build --features cuda
```

## License

SPDX-License-Identifier: GPL-2.0-only
(c) 2025, Sasha Levin <sashal@kernel.org>

## Contributing

Open a pull request against `main`. Sign your work with `git commit -s`
(Developer Certificate of Origin). Commit messages use imperative subjects,
optionally prefixed with the subsystem (`llm/kiro: ...`); explain *why* in the
body, wrapped at 72 columns.
