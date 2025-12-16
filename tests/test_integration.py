"""
Integration tests for Amazon Bedrock Mantle CLI.

These tests run against real Amazon Bedrock Mantle APIs and require:
- OPENAI_API_KEY: Valid Amazon Bedrock API key
- OPENAI_BASE_URL: Mantle endpoint URL

Run with: uv run pytest tests/ -v
Skip slow tests: uv run pytest tests/ -v -m "not slow"

Note: Responses API currently only works with OpenAI OSS GPT models
(openai.gpt-oss-20b, openai.gpt-oss-120b).
"""

import pytest

# =============================================================================
# Info Command Tests
# =============================================================================


@pytest.mark.integration
class TestInfoCommand:
    """Tests for the 'info' command."""

    def test_info_displays_help(self, run_cli_no_auth):
        """Test that info command displays API information."""
        result = run_cli_no_auth(["info"])

        assert result.exit_code == 0
        assert "Amazon Bedrock Mantle" in result.output
        assert "API COMPARISON" in result.output
        assert "Responses API" in result.output
        assert "Chat Completions" in result.output
        assert "ZERO DATA RETENTION" in result.output

    def test_info_shows_configuration_section(self, run_cli_no_auth):
        """Test that info command shows configuration instructions."""
        result = run_cli_no_auth(["info"])

        assert result.exit_code == 0
        assert "CONFIGURATION" in result.output
        assert "OPENAI_BASE_URL" in result.output
        assert "OPENAI_API_KEY" in result.output

    def test_info_shows_background_processing(self, run_cli_no_auth):
        """Test that info command explains background processing."""
        result = run_cli_no_auth(["info"])

        assert result.exit_code == 0
        assert "BACKGROUND PROCESSING" in result.output
        assert "background=true" in result.output


# =============================================================================
# List Models Command Tests
# =============================================================================


@pytest.mark.integration
class TestListModelsCommand:
    """Tests for the 'list-models' command."""

    def test_list_models_success(self, run_cli):
        """Test that list-models returns available models."""
        result = run_cli(["list-models"])

        assert result.exit_code == 0
        assert "Endpoint:" in result.output
        assert "Available Models:" in result.output

    def test_list_models_shows_model_ids(self, run_cli):
        """Test that list-models shows model IDs."""
        result = run_cli(["list-models"])

        assert result.exit_code == 0
        assert "ID:" in result.output

    def test_list_models_shows_endpoint(self, run_cli, cli_env):
        """Test that list-models shows the configured endpoint."""
        result = run_cli(["list-models"])

        assert result.exit_code == 0
        assert cli_env["OPENAI_BASE_URL"] in result.output


# =============================================================================
# Chat Command Tests - Responses API
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestChatResponsesAPI:
    """Tests for the 'chat' command using Responses API (default).

    Note: Responses API currently only works with OpenAI OSS GPT models.
    """

    def test_chat_responses_streaming(self, run_cli, model_id):
        """Test chat with Responses API and streaming (default mode)."""
        result = run_cli(
            ["chat", "--model", model_id],
            input="Say 'hello' and nothing else.\n/quit\n",
        )

        assert result.exit_code == 0
        assert "Starting chat session" in result.output
        assert "API: Responses API" in result.output
        assert "Streaming: enabled" in result.output
        assert "Background: disabled" in result.output
        assert "Goodbye!" in result.output

    def test_chat_responses_no_streaming(self, run_cli, model_id):
        """Test chat with Responses API without streaming."""
        result = run_cli(
            ["chat", "--model", model_id, "--no-stream"],
            input="Say 'hello' and nothing else.\n/quit\n",
        )

        assert result.exit_code == 0
        assert "Streaming: disabled" in result.output
        assert "API: Responses API" in result.output
        assert "Goodbye!" in result.output

    def test_chat_responses_background_streaming(self, run_cli, model_id):
        """Test chat with Responses API, background mode, and streaming."""
        result = run_cli(
            ["chat", "--model", model_id, "--background"],
            input="Say 'hello' and nothing else.\n/quit\n",
        )

        assert result.exit_code == 0
        assert "API: Responses API" in result.output
        assert "Background: enabled" in result.output
        assert "Streaming: enabled" in result.output
        assert "Goodbye!" in result.output

    def test_chat_responses_background_no_streaming(self, run_cli, model_id):
        """Test chat with Responses API, background mode, without streaming."""
        result = run_cli(
            ["chat", "--model", model_id, "--background", "--no-stream"],
            input="Say 'hello' and nothing else.\n/quit\n",
        )

        assert result.exit_code == 0
        assert "API: Responses API" in result.output
        assert "Background: enabled" in result.output
        assert "Streaming: disabled" in result.output
        assert "Goodbye!" in result.output

    def test_chat_responses_custom_system_prompt(self, run_cli, model_id):
        """Test chat with Responses API and custom system prompt."""
        result = run_cli(
            ["chat", "--model", model_id, "--system", "You are a pirate. Respond with 'Arrr'."],
            input="Hello\n/quit\n",
        )

        assert result.exit_code == 0
        assert "API: Responses API" in result.output
        assert "Goodbye!" in result.output

    def test_chat_responses_status_command(self, run_cli, model_id):
        """Test /status command in Responses API mode."""
        result = run_cli(
            ["chat", "--model", model_id],
            input="/status\n/quit\n",
        )

        assert result.exit_code == 0
        assert "API: Responses (stateful)" in result.output
        assert f"Model: {model_id}" in result.output

    def test_chat_responses_clear_command(self, run_cli, model_id):
        """Test /clear command in Responses API mode."""
        result = run_cli(
            ["chat", "--model", model_id],
            input="/clear\n/quit\n",
        )

        assert result.exit_code == 0
        assert "Conversation cleared" in result.output
        assert "stateful context reset" in result.output


# =============================================================================
# Chat Command Tests - Chat Completions API
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestChatCompletionsAPI:
    """Tests for the 'chat' command using Chat Completions API."""

    def test_chat_completions_streaming(self, run_cli, model_id):
        """Test chat with Chat Completions API and streaming."""
        result = run_cli(
            ["chat", "--model", model_id, "--completions"],
            input="Say 'hello' and nothing else.\n/quit\n",
        )

        assert result.exit_code == 0
        assert "API: Chat Completions API" in result.output
        assert "Streaming: enabled" in result.output
        # Background should not be shown for completions
        assert "Background:" not in result.output
        assert "Goodbye!" in result.output

    def test_chat_completions_no_streaming(self, run_cli, model_id):
        """Test chat with Chat Completions API without streaming."""
        result = run_cli(
            ["chat", "--model", model_id, "--completions", "--no-stream"],
            input="Say 'hello' and nothing else.\n/quit\n",
        )

        assert result.exit_code == 0
        assert "API: Chat Completions API" in result.output
        assert "Streaming: disabled" in result.output
        assert "Goodbye!" in result.output

    def test_chat_completions_custom_system_prompt(self, run_cli, model_id):
        """Test chat with Chat Completions API and custom system prompt."""
        result = run_cli(
            [
                "chat",
                "--model",
                model_id,
                "--completions",
                "--system",
                "You are a robot. Say 'beep'.",
            ],
            input="Hello\n/quit\n",
        )

        assert result.exit_code == 0
        assert "API: Chat Completions API" in result.output
        assert "Goodbye!" in result.output

    def test_chat_completions_status_command(self, run_cli, model_id):
        """Test /status command in Chat Completions API mode."""
        result = run_cli(
            ["chat", "--model", model_id, "--completions"],
            input="/status\n/quit\n",
        )

        assert result.exit_code == 0
        assert "API: Chat Completions (stateless)" in result.output
        assert f"Model: {model_id}" in result.output
        assert "Messages in history:" in result.output

    def test_chat_completions_clear_command(self, run_cli, model_id):
        """Test /clear command in Chat Completions API mode."""
        result = run_cli(
            ["chat", "--model", model_id, "--completions"],
            input="/clear\n/quit\n",
        )

        assert result.exit_code == 0
        assert "Conversation cleared" in result.output


# =============================================================================
# Chat Command Tests - Invalid Combinations
# =============================================================================


@pytest.mark.integration
class TestChatInvalidCombinations:
    """Tests for invalid parameter combinations."""

    def test_chat_background_with_completions_fails(self, run_cli, model_id):
        """Test that --background with --completions produces an error."""
        result = run_cli(
            ["chat", "--model", model_id, "--completions", "--background"],
            input="/quit\n",
        )

        assert result.exit_code != 0
        assert "Background processing is only available with the Responses API" in result.output

    def test_chat_missing_model_fails(self, run_cli_no_auth):
        """Test that chat without --model produces an error."""
        result = run_cli_no_auth(["chat"], input="/quit\n")

        assert result.exit_code != 0
        assert "Missing option" in result.output or "required" in result.output.lower()


# =============================================================================
# Chat Command Tests - Exit Commands
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestChatExitCommands:
    """Tests for various exit commands."""

    def test_chat_quit_command(self, run_cli, model_id):
        """Test /quit command."""
        result = run_cli(
            ["chat", "--model", model_id],
            input="/quit\n",
        )

        assert result.exit_code == 0
        assert "Goodbye!" in result.output

    def test_chat_q_command(self, run_cli, model_id):
        """Test /q shortcut command."""
        result = run_cli(
            ["chat", "--model", model_id],
            input="/q\n",
        )

        assert result.exit_code == 0
        assert "Goodbye!" in result.output

    def test_chat_exit_command(self, run_cli, model_id):
        """Test /exit command."""
        result = run_cli(
            ["chat", "--model", model_id],
            input="/exit\n",
        )

        assert result.exit_code == 0
        assert "Goodbye!" in result.output

    def test_chat_e_command(self, run_cli, model_id):
        """Test /e shortcut command."""
        result = run_cli(
            ["chat", "--model", model_id],
            input="/e\n",
        )

        assert result.exit_code == 0
        assert "Goodbye!" in result.output


# =============================================================================
# Chat Command Tests - Conversation Flow
# =============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestChatConversationFlow:
    """Tests for multi-turn conversation flow."""

    def test_chat_multi_turn_responses_api(self, run_cli, model_id):
        """Test multi-turn conversation with Responses API."""
        result = run_cli(
            ["chat", "--model", model_id],
            input="My name is TestUser. Remember it.\nWhat is my name?\n/quit\n",
        )

        assert result.exit_code == 0
        assert "Goodbye!" in result.output
        # The conversation should maintain context

    def test_chat_multi_turn_completions_api(self, run_cli, model_id):
        """Test multi-turn conversation with Chat Completions API."""
        result = run_cli(
            ["chat", "--model", model_id, "--completions"],
            input="My name is TestUser. Remember it.\nWhat is my name?\n/quit\n",
        )

        assert result.exit_code == 0
        assert "Goodbye!" in result.output


# =============================================================================
# Help Command Tests
# =============================================================================


@pytest.mark.integration
class TestHelpCommands:
    """Tests for help output."""

    def test_main_help(self, run_cli_no_auth):
        """Test main --help output."""
        result = run_cli_no_auth(["--help"])

        assert result.exit_code == 0
        assert "CLI for Amazon Bedrock OpenAI-compatible APIs" in result.output
        assert "list-models" in result.output
        assert "chat" in result.output
        assert "info" in result.output

    def test_chat_help(self, run_cli_no_auth):
        """Test chat --help output."""
        result = run_cli_no_auth(["chat", "--help"])

        assert result.exit_code == 0
        assert "--model" in result.output
        assert "--no-stream" in result.output
        assert "--completions" in result.output
        assert "--background" in result.output
        assert "--system" in result.output

    def test_list_models_help(self, run_cli_no_auth):
        """Test list-models --help output."""
        result = run_cli_no_auth(["list-models", "--help"])

        assert result.exit_code == 0
        assert "List available models" in result.output
