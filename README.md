# bedrock-mantle

A CLI for exploring [Amazon Bedrock OpenAI-compatible APIs](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html).

Amazon Bedrock now supports **OpenAI-compatible APIs** through Project Mantle:

- **Use familiar APIs** - Same endpoints and formats you know from OpenAI
- **Drop-in compatibility** - Existing OpenAI SDK code works with minimal changes
- **New capabilities** - Stateful conversations and background processing for long-running tasks

This CLI is an **exploration tool** to help you understand and test these new endpoints.

## Quick Start

```bash
# Install
uv tool install .

# Configure (get your API key from AWS console)
export OPENAI_BASE_URL=https://bedrock-mantle.us-east-1.api.aws/v1
export OPENAI_API_KEY=your-amazon-bedrock-api-key

# Start chatting
bedrock-mantle chat --model openai.gpt-oss-120b
```

Example session:
```
Starting chat session
  Model: openai.gpt-oss-120b
  API: Responses API
  Streaming: enabled
  Background: disabled

Type /quit or /q to exit, /clear to reset conversation
------------------------------------------------------------

You: What is the capital of France?
Assistant: The capital of France is Paris.

You: /quit
Goodbye!
```

## Two APIs, Different Strengths

Amazon Bedrock Mantle offers two OpenAI-compatible APIs:

### Responses API (Default)

**Stateful conversations** - The server remembers context automatically.

```bash
bedrock-mantle chat --model openai.gpt-oss-120b
```

The Responses API maintains conversation state server-side using `previous_response_id`. You don't need to send the full conversation history with each request.

**Background processing** - For complex tasks that take time:

```bash
bedrock-mantle chat --model openai.gpt-oss-120b --background
```

When enabled, requests are queued and processed asynchronously. The CLI polls for completion, which is useful for:
- Complex reasoning tasks that may take minutes
- Avoiding connection timeouts
- Building reliable async workflows

### Chat Completions API

**Stateless and simple** - You manage conversation history client-side.

```bash
bedrock-mantle chat --model openai.gpt-oss-120b --completions
```

Choose this when you need:
- Full control over conversation context
- Zero Data Retention (ZDR) compliance
- Maximum compatibility with existing OpenAI code

### Feature Comparison

| Feature | Responses API | Chat Completions API |
|---------|---------------|---------------------|
| Model Support | OpenAI OSS GPT models only | All Bedrock models |
| State Management | Server-side (stateful) | Client-side (stateless) |
| Background Processing | ✓ Supported | ✗ Not available |
| ZDR Compatible | ✗ Stores data ~30 days | ✓ No data stored |
| Conversation Context | Automatic via ID | Manual history |
| Cancel Request | ✓ Supported | ✗ Not available |

## Installation

```bash
# Recommended: install as a uv tool
uv tool install .

# Or for development
uv sync

# Or using pip
pip install -e .
```

## Configuration

Set these two required environment variables:

```bash
export OPENAI_BASE_URL=https://bedrock-mantle.us-east-1.api.aws/v1
export OPENAI_API_KEY=your-amazon-bedrock-api-key
```

Or create a `.env` file (the CLI loads it automatically):

```bash
cp .env.example .env
# Edit .env with your values
```

Get your API key from the [Amazon Bedrock console](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html).

## Commands

### List Models

```bash
bedrock-mantle list-models
```

### Chat Options

```bash
# Responses API with streaming (default)
bedrock-mantle chat --model openai.gpt-oss-120b

# Chat Completions API
bedrock-mantle chat --model openai.gpt-oss-120b --completions

# Background processing
bedrock-mantle chat --model openai.gpt-oss-120b --background

# Disable streaming
bedrock-mantle chat --model openai.gpt-oss-120b --no-stream

# Custom system prompt
bedrock-mantle chat --model openai.gpt-oss-120b --system "You are a pirate"
```

### In-Chat Commands

- `/quit` or `/q` - Exit
- `/clear` - Reset conversation
- `/status` - Show current settings

### API Info

```bash
bedrock-mantle info
```

## Available Models

The **Chat Completions API** supports all Bedrock models.

The **Responses API** currently supports OpenAI OSS GPT models.

Use `list-models` to see all available models in your region.

## Development

```bash
# Install dependencies
uv sync

# Run tests (requires credentials in .env)
uv run pytest tests/ -v

# Lint and format
uv run ruff check . --fix
uv run ruff format .

# Run all pre-commit checks
uv run pre-commit run --all-files
```

## License

MIT
