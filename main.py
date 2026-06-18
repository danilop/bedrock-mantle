#!/usr/bin/env python3
"""
CLI for Amazon Bedrock OpenAI-compatible APIs (Mantle).

This CLI provides access to the Bedrock Mantle APIs which are compatible with OpenAI's
API format, including the Responses API and Chat Completions API.

Key differences between APIs:
- Responses API: Stateful, supports background processing, maintains conversation context
- Chat Completions API: Stateless, simpler but requires manual context management

Both APIs support the same models through the Mantle endpoint.
"""

import os
import re
import time
from collections.abc import Iterator
from typing import Any

import click
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Command constants
EXIT_COMMANDS = frozenset({"/quit", "/q", "/exit", "/e"})


def process_streaming_events(stream_response: Iterator[Any]) -> tuple[str, str | None]:
    """
    Process streaming events from the Responses API.

    Handles different event types including text deltas, status updates,
    and completion events.

    Args:
        stream_response: Iterator of streaming events from the API.

    Returns:
        Tuple of (accumulated_text, response_id).
    """
    response_text = ""
    response_id = None

    for event in stream_response:
        if hasattr(event, "type"):
            if event.type == "response.output_text.delta":
                if hasattr(event, "delta"):
                    click.echo(event.delta, nl=False)
                    response_text += event.delta
            elif event.type == "response.completed":
                if hasattr(event, "response") and hasattr(event.response, "id"):
                    response_id = event.response.id
            elif event.type == "response.queued":
                click.echo("[Queued...]", nl=False)
            elif event.type == "response.in_progress":
                click.echo("[Processing...]", nl=False)
        elif hasattr(event, "delta"):
            # Fallback for delta content
            click.echo(event.delta, nl=False)
            response_text += event.delta
        elif hasattr(event, "id"):
            response_id = event.id

    click.echo()  # New line after response
    return response_text, response_id


def extract_response_text(response: Any) -> str:
    """
    Extract text content from a Responses API response object.

    Handles different response formats including output_text attribute
    and nested output content structures.

    Args:
        response: Response object from the Responses API.

    Returns:
        Extracted text content as a string.
    """
    if hasattr(response, "output_text"):
        return response.output_text

    if hasattr(response, "output") and response.output:
        text_parts = []
        for output in response.output:
            if hasattr(output, "content"):
                for content in output.content:
                    if hasattr(content, "text"):
                        text_parts.append(content.text)
        if text_parts:
            return "".join(text_parts)

    return str(response.output)


def region_from_base_url(base_url: str) -> str | None:
    """Best-effort extraction of an AWS region from a Mantle endpoint URL.

    Example: https://bedrock-mantle.us-east-1.api.aws/v1 -> us-east-1
    """
    match = re.search(r"bedrock-mantle\.([a-z0-9-]+)\.api\.aws", base_url)
    return match.group(1) if match else None


def mantle_base_url(region: str) -> str:
    """Construct the Mantle endpoint URL for a region."""
    return f"https://bedrock-mantle.{region}.api.aws/v1"


def build_aws_session(profile: str | None):
    """Build a boto3 session for the given profile (or the default chain).

    Imports boto3 lazily to keep CLI startup fast for the static-API-key path.
    """
    try:
        import boto3
    except ImportError as e:
        raise click.ClickException(
            f"Could not import '{e.name}'. Reinstall the CLI to restore AWS dependencies:\n"
            "  uv tool install . --force   (or: pip install -e .)"
        ) from None

    return boto3.Session(profile_name=profile)


def mint_bedrock_token(session, region: str) -> str:
    """Mint a short-term Bedrock bearer token from a boto3 session's credentials.

    Args:
        session: A boto3 Session whose credentials should be used.
        region: AWS region for the token (required by the signer).

    Returns:
        A short-term bearer token string.
    """
    try:
        from aws_bedrock_token_generator import provide_token
    except ImportError as e:
        raise click.ClickException(
            f"Could not import '{e.name}'. Reinstall the CLI to restore AWS dependencies:\n"
            "  uv tool install . --force   (or: pip install -e .)"
        ) from None

    try:
        credentials = session.get_credentials()
        if credentials is None:
            raise click.ClickException(
                "No AWS credentials found"
                + (
                    f" for profile '{session.profile_name}'."
                    if session.profile_name
                    else " in the default credential chain."
                )
            )

        # provide_token() expects a provider exposing .load() -> credentials.
        # Wrap the session's (frozen) credentials so the profile is honored.
        class _CredentialProvider:
            def load(self):
                return credentials.get_frozen_credentials()

        return provide_token(region=region, aws_credentials_provider=_CredentialProvider())
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to mint Bedrock token: {e}") from None


def create_client(profile: str | None = None) -> OpenAI:
    """Create an OpenAI client configured for Bedrock Mantle.

    Authentication resolves in this order:
    1. If --profile is given, mint a short-term token from that AWS profile.
    2. Else use OPENAI_API_KEY if set (static Bedrock API key).
    3. Else fall back to the default AWS credential chain (env/AWS_PROFILE/SSO/etc).

    OPENAI_BASE_URL is optional when a region can be determined (from the env,
    the AWS profile config, or the URL itself) -- the endpoint is built from it.
    """
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")

    region = (
        os.environ.get("AWS_REGION")
        or os.environ.get("AWS_DEFAULT_REGION")
        or (region_from_base_url(base_url) if base_url else None)
    )

    # Use AWS credentials when a profile is requested, or when no static key is set.
    use_aws = bool(profile) or not api_key

    if use_aws:
        session = build_aws_session(profile)
        # Fall back to the region configured for the profile (e.g. ~/.aws/config).
        if not region:
            region = session.region_name
        if not region:
            raise click.ClickException(
                "Could not determine an AWS region.\n"
                "Set AWS_REGION/AWS_DEFAULT_REGION, configure a region for the profile,\n"
                "or set OPENAI_BASE_URL (e.g. https://bedrock-mantle.us-east-1.api.aws/v1)."
            )
        api_key = mint_bedrock_token(session, region)

    if not base_url:
        if not region:
            raise click.ClickException(
                "OPENAI_BASE_URL is required when no AWS region can be determined.\n"
                "Set it in a .env file or as an environment variable.\n"
                "Example: https://bedrock-mantle.us-east-1.api.aws/v1"
            )
        base_url = mantle_base_url(region)

    return OpenAI(base_url=base_url, api_key=api_key)


@click.group()
def cli():
    """
    CLI for Amazon Bedrock OpenAI-compatible APIs (Mantle).

    This CLI provides access to:
    - Models API: List available models
    - Responses API: Stateful conversations with background processing support
    - Chat Completions API: Stateless chat completions

    Both Responses API and Chat Completions API support the same models.
    The Responses API adds stateful conversation management and async background processing.

    Configuration is done via environment variables (or .env file):
    - OPENAI_BASE_URL: Mantle endpoint URL (optional if a region can be determined)
    - OPENAI_API_KEY: Your Bedrock API key (optional if using AWS credentials)

    Authentication resolves in this order:
    1. --profile flag: mint a short-term token from that AWS profile
    2. OPENAI_API_KEY: a static Bedrock API key
    3. Default AWS credential chain (honors AWS_PROFILE, SSO, env vars, etc.)

    When using AWS credentials, the endpoint URL is built automatically from the
    region (AWS_REGION/AWS_DEFAULT_REGION, the profile's configured region, or
    OPENAI_BASE_URL).
    """
    pass


@cli.command("list-models")
@click.option(
    "--profile",
    default=None,
    help="AWS named profile to authenticate with (mints a short-term Bedrock token).",
)
def list_models(profile: str | None):
    """
    List available models for Bedrock Mantle.

    Models listed here are available for both the Responses API and Chat Completions API.
    The same set of models is supported by both APIs.
    """
    try:
        client = create_client(profile)
    except click.ClickException:
        raise  # Let credential errors propagate with their original message

    click.echo(f"Endpoint: {client.base_url}")
    click.echo()

    try:
        models = client.models.list()

        click.echo("Available Models:")
        click.echo("-" * 60)

        for model in models.data:
            click.echo(f"  ID: {model.id}")
            if hasattr(model, "created"):
                click.echo(f"      Created: {model.created}")
            if hasattr(model, "owned_by"):
                click.echo(f"      Owner: {model.owned_by}")
            click.echo()

    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Failed to list models: {e}") from None


@cli.command("chat")
@click.option(
    "--model",
    "-m",
    required=True,
    help="Model ID or inference profile to use",
)
@click.option(
    "--no-stream",
    is_flag=True,
    default=False,
    help="Disable streaming (streaming is enabled by default)",
)
@click.option(
    "--completions",
    is_flag=True,
    default=False,
    help="Use Chat Completions API instead of Responses API",
)
@click.option(
    "--background",
    is_flag=True,
    default=False,
    help="Enable background processing (Responses API only). Demonstrates async inference.",
)
@click.option(
    "--system",
    "-s",
    default="You are a helpful assistant.",
    help="System prompt for the conversation",
)
@click.option(
    "--profile",
    default=None,
    help="AWS named profile to authenticate with (mints a short-term Bedrock token).",
)
def chat(
    model: str,
    no_stream: bool,
    completions: bool,
    background: bool,
    system: str,
    profile: str | None,
):
    """
    Start an interactive chat session.

    By default, uses the Responses API with streaming enabled.

    \b
    API Comparison:
    - Responses API (default): Stateful, supports background processing,
      maintains conversation context automatically via previous_response_id
    - Chat Completions API (--completions): Stateless, simpler interface,
      requires manual conversation history management

    \b
    Commands during chat:
      /quit or /q  - Exit the chat
      /exit or /e  - Exit the chat
      /clear       - Clear conversation history
      /status      - Show current API mode and settings
    """
    stream = not no_stream

    if background and completions:
        raise click.ClickException(
            "Background processing is only available with the Responses API.\n"
            "Remove --completions to use background mode."
        )

    if background and not no_stream:
        click.echo(
            "Note: Background mode with streaming - events will stream as processing completes."
        )
        click.echo()

    api_mode = "Chat Completions" if completions else "Responses"
    click.echo("Starting chat session")
    click.echo(f"  Model: {model}")
    click.echo(f"  API: {api_mode} API")
    click.echo(f"  Streaming: {'disabled' if no_stream else 'enabled'}")
    if not completions:
        click.echo(f"  Background: {'enabled' if background else 'disabled'}")
    click.echo()
    click.echo("Type /quit or /q to exit, /clear to reset conversation")
    click.echo("-" * 60)
    click.echo()

    try:
        client = create_client(profile)
    except click.ClickException:
        raise  # Let credential errors propagate with their original message

    try:
        if completions:
            run_chat_completions(client, model, stream, system)
        else:
            run_responses_api(client, model, stream, background, system)

    except KeyboardInterrupt:
        click.echo("\n\nChat session ended.")
    except click.ClickException:
        raise
    except Exception as e:
        raise click.ClickException(f"Chat error: {e}") from None


def run_chat_completions(client: OpenAI, model: str, stream: bool, system: str) -> None:
    """
    Run an interactive chat using the Chat Completions API.

    This API is stateless - we must maintain the full conversation history
    and send it with each request.
    """
    messages = [{"role": "system", "content": system}]

    while True:
        try:
            user_input = click.prompt("You", prompt_suffix=": ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in EXIT_COMMANDS:
            click.echo("Goodbye!")
            break
        elif user_input.lower() == "/clear":
            messages = [{"role": "system", "content": system}]
            click.echo("Conversation cleared.\n")
            continue
        elif user_input.lower() == "/status":
            click.echo("API: Chat Completions (stateless)")
            click.echo(f"Model: {model}")
            click.echo(f"Messages in history: {len(messages)}")
            click.echo()
            continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        click.echo()
        click.echo("Assistant: ", nl=False)

        try:
            if stream:
                response_text = ""
                stream_response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=True,
                )

                for chunk in stream_response:
                    if chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        click.echo(content, nl=False)
                        response_text += content

                click.echo()  # New line after response
                messages.append({"role": "assistant", "content": response_text})

            else:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )

                assistant_message = response.choices[0].message.content
                click.echo(assistant_message)
                messages.append({"role": "assistant", "content": assistant_message})

        except Exception as e:
            click.echo(f"\nError: {e}")
            # Remove the failed user message
            messages.pop()

        click.echo()


def run_responses_api(
    client: OpenAI, model: str, stream: bool, background: bool, system: str
) -> None:
    """
    Run an interactive chat using the Responses API.

    This API is stateful - it can maintain conversation context via previous_response_id.
    It also supports background processing for async inference.
    """
    previous_response_id = None

    while True:
        try:
            user_input = click.prompt("You", prompt_suffix=": ").strip()
        except EOFError:
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in EXIT_COMMANDS:
            click.echo("Goodbye!")
            break
        elif user_input.lower() == "/clear":
            previous_response_id = None
            click.echo("Conversation cleared (stateful context reset).\n")
            continue
        elif user_input.lower() == "/status":
            click.echo("API: Responses (stateful)")
            click.echo(f"Model: {model}")
            click.echo(f"Background: {'enabled' if background else 'disabled'}")
            click.echo(f"Previous response ID: {previous_response_id or 'None (new conversation)'}")
            click.echo()
            continue

        click.echo()
        click.echo("Assistant: ", nl=False)

        try:
            # Build the input - first message includes system prompt if no previous context
            input_messages = []

            if previous_response_id is None:
                # Include system message on first turn
                input_messages.append({"role": "system", "content": system})

            input_messages.append({"role": "user", "content": user_input})

            # Build request parameters
            request_params = {
                "model": model,
                "input": input_messages,
            }

            # Add previous response ID for conversation continuity
            if previous_response_id:
                request_params["previous_response_id"] = previous_response_id

            if background:
                # Background mode - async processing
                request_params["background"] = True

                if stream:
                    # Background + Streaming: stream events as they become available
                    request_params["stream"] = True
                    stream_response = client.responses.create(**request_params)
                    _, response_id = process_streaming_events(stream_response)
                    if response_id:
                        previous_response_id = response_id
                else:
                    # Background + Polling (non-streaming)
                    response = client.responses.create(**request_params)
                    response_id = response.id

                    # Poll for completion
                    click.echo("[Background processing started...]", nl=False)
                    poll_count = 0

                    while response.status in ["queued", "in_progress"]:
                        time.sleep(1)
                        poll_count += 1
                        if poll_count % 5 == 0:
                            click.echo(".", nl=False)
                        response = client.responses.retrieve(response_id)

                    click.echo()  # Clear the polling line
                    click.echo("Assistant: ", nl=False)

                    if response.status == "completed":
                        click.echo(extract_response_text(response))
                        previous_response_id = response_id
                    elif response.status == "failed":
                        click.echo(f"[Background task failed: {response.status}]")
                    elif response.status == "cancelled":
                        click.echo("[Background task was cancelled]")
                    else:
                        click.echo(f"[Unexpected status: {response.status}]")

            elif stream:
                # Streaming mode
                request_params["stream"] = True
                stream_response = client.responses.create(**request_params)
                _, response_id = process_streaming_events(stream_response)
                if response_id:
                    previous_response_id = response_id

            else:
                # Non-streaming mode
                response = client.responses.create(**request_params)
                click.echo(extract_response_text(response))
                previous_response_id = response.id

        except Exception as e:
            click.echo(f"\nError: {e}")

        click.echo()


@cli.command("info")
def info():
    """
    Show information about API differences and limitations.

    Displays a comparison between the Responses API and Chat Completions API,
    including model availability and feature support.
    """
    click.echo("""
Amazon Bedrock Mantle - OpenAI-Compatible APIs
==============================================

CONFIGURATION
-------------
Set these environment variables (or use a .env file):

  OPENAI_BASE_URL  Mantle endpoint (required)
                   Example: https://bedrock-mantle.us-east-1.api.aws/v1

  OPENAI_API_KEY   Your Bedrock API key (required)
                   Generate at: https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html

Supported Regions:
  us-east-1, us-east-2, us-west-2, ap-southeast-3, ap-south-1,
  ap-northeast-1, eu-central-1, eu-west-1, eu-west-2, eu-south-1,
  eu-north-1, sa-east-1

API COMPARISON
--------------
┌─────────────────────────┬─────────────────────┬────────────────────────┐
│ Feature                 │ Responses API       │ Chat Completions API   │
├─────────────────────────┼─────────────────────┼────────────────────────┤
│ State Management        │ Stateful            │ Stateless              │
│ Conversation Context    │ Automatic (ID)      │ Manual (history)       │
│ Background Processing   │ ✓ Supported         │ ✗ Not supported        │
│ Response Storage        │ ~30 days            │ Temporary              │
│ Streaming               │ ✓ Supported         │ ✓ Supported            │
│ Tool/Function Calling   │ ✓ Supported         │ ✓ Supported            │
│ Cancel Request          │ ✓ Supported         │ ✗ Not supported        │
└─────────────────────────┴─────────────────────┴────────────────────────┘

MODEL AVAILABILITY
------------------
Both APIs access the same set of models through the Mantle endpoint.
Use 'list-models' to see available models.

Known models include:
  - openai.gpt-oss-20b: Smaller model, optimized for lower latency
  - openai.gpt-oss-120b: Larger model, optimized for production use

BACKGROUND PROCESSING
---------------------
The Responses API supports async background processing for long-running tasks:
  1. Set background=true in your request (use --background flag)
  2. Receive immediate response with ID and status="queued"
  3. Poll for completion using the response ID
  4. Retrieve results when status="completed"

This is useful for:
  - Complex reasoning tasks that may take minutes
  - Avoiding connection timeouts
  - Building reliable async workflows

LIMITATIONS
-----------
- Chat Completions API does not support background processing
- Background mode has higher time-to-first-token latency

ZERO DATA RETENTION (ZDR)
-------------------------
ZDR is a policy where API inputs/outputs are not stored beyond immediate processing.
By default, the API retains data for 30 days for safety monitoring.

The Responses API is NOT ZDR-compatible because it stores data for:
  - Background processing (~10 minutes for polling)
  - Stateful conversations (~30 days for previous_response_id)

The Chat Completions API is stateless and can be ZDR-compatible.

See: https://platform.openai.com/docs/guides/your-data
""")


PROG_NAME = "bedrock-mantle"
COMPLETE_VAR = "_BEDROCK_MANTLE_COMPLETE"

# Per-shell config: the file to write the script to and the profile to source it from.
SHELL_CONFIG = {
    "bash": ("~/.config/bedrock-mantle/complete.bash", "~/.bashrc"),
    "zsh": ("~/.config/bedrock-mantle/complete.zsh", "~/.zshrc"),
    "fish": ("~/.config/fish/completions/bedrock-mantle.fish", None),
}


def generate_completion(shell: str) -> str:
    """Render the shell completion script for the given shell."""
    from click.shell_completion import get_completion_class

    comp_cls = get_completion_class(shell)
    if comp_cls is None:
        raise click.ClickException(f"Unsupported shell: {shell}")
    return comp_cls(cli, {}, PROG_NAME, COMPLETE_VAR).source()


@cli.command("completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
@click.option(
    "--install",
    is_flag=True,
    default=False,
    help="Write the completion script and wire it into your shell profile.",
)
def completion(shell: str, install: bool):
    """
    Generate (or install) shell completion for bedrock-mantle.

    \b
    Print the script to evaluate in the current shell:
      eval "$(bedrock-mantle completion bash)"

    \b
    Or install it permanently for new shells:
      bedrock-mantle completion bash --install
    """
    script = generate_completion(shell)

    if not install:
        click.echo(script)
        return

    script_path, profile = SHELL_CONFIG[shell]
    script_file = os.path.expanduser(script_path)
    os.makedirs(os.path.dirname(script_file), exist_ok=True)
    with open(script_file, "w") as f:
        f.write(script + "\n")
    click.echo(f"Wrote completion script to {script_path}")

    # fish auto-loads from its completions dir; nothing else to wire up.
    if profile is None:
        click.echo("Restart your shell (or open a new one) to enable completion.")
        return

    profile_file = os.path.expanduser(profile)
    source_line = f"[ -f {script_path} ] && source {script_path}"
    marker = "# bedrock-mantle shell completion"

    existing = ""
    if os.path.exists(profile_file):
        with open(profile_file) as f:
            existing = f.read()

    if marker in existing:
        click.echo(f"{profile} already sources the completion script. Nothing to do.")
    else:
        with open(profile_file, "a") as f:
            f.write(f"\n{marker}\n{source_line}\n")
        click.echo(f"Added source line to {profile}")

    click.echo(f"Run 'source {profile}' or open a new shell to enable completion.")


if __name__ == "__main__":
    cli()
