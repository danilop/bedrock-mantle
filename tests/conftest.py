"""
Pytest configuration and fixtures for integration tests.

These tests run against real Amazon Bedrock Mantle APIs and require:
- OPENAI_API_KEY: Valid Amazon Bedrock API key
- OPENAI_BASE_URL: Mantle endpoint URL

Note: Responses API currently only works with OpenAI OSS GPT models
(openai.gpt-oss-20b, openai.gpt-oss-120b).
"""

import os

import pytest
from click.testing import CliRunner
from dotenv import load_dotenv

from main import cli

# Load .env file for credentials
load_dotenv()

# Default model for testing (supports both Responses API and Chat Completions API)
DEFAULT_TEST_MODEL = "openai.gpt-oss-20b"


def get_env_or_skip():
    """Get required environment variables or skip the test."""
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")

    if not api_key or not base_url:
        pytest.skip(
            "Integration tests require OPENAI_API_KEY and OPENAI_BASE_URL environment variables"
        )

    return {"OPENAI_API_KEY": api_key, "OPENAI_BASE_URL": base_url}


@pytest.fixture
def runner():
    """Fixture that provides a Click CliRunner."""
    return CliRunner()


@pytest.fixture
def cli_env():
    """Fixture that provides environment variables for CLI execution (requires credentials)."""
    return get_env_or_skip()


@pytest.fixture
def run_cli(runner, cli_env):
    """
    Fixture that returns a function to run CLI commands using CliRunner.

    Usage:
        result = run_cli(["list-models"])
        result = run_cli(["chat", "--model", "openai.gpt-oss-20b"], input="Hello\n/quit\n")
    """

    def _run_cli(args, input=None):
        """
        Run the CLI with given arguments.

        Args:
            args: List of CLI arguments (e.g., ["list-models"])
            input: Optional input string to send to stdin (for interactive commands)

        Returns:
            click.testing.Result with output, exit_code, exception
        """
        result = runner.invoke(cli, args, input=input, env=cli_env, catch_exceptions=False)
        return result

    return _run_cli


@pytest.fixture
def run_cli_no_auth(runner):
    """
    Fixture that returns a function to run CLI commands without requiring auth.

    Use for commands that don't need API credentials (--help, info).
    """

    def _run_cli(args, input=None):
        result = runner.invoke(cli, args, input=input, catch_exceptions=False)
        return result

    return _run_cli


@pytest.fixture
def model_id(cli_env):
    """
    Fixture that returns a valid model ID for testing.

    Uses openai.gpt-oss-20b which supports both Responses API and Chat Completions API.
    """
    return DEFAULT_TEST_MODEL


# Markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "slow: marks tests as slow running")
