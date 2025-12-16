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


def create_client() -> OpenAI:
    """Create an OpenAI client configured for Bedrock Mantle."""
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")

    if not api_key:
        raise click.ClickException(
            "OPENAI_API_KEY is required. Set it in .env file or as environment variable.\n"
            "See: https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html"
        )

    if not base_url:
        raise click.ClickException(
            "OPENAI_BASE_URL is required. Set it in .env file or as environment variable.\n"
            "Example: https://bedrock-mantle.us-east-1.api.aws/v1"
        )

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
    - OPENAI_API_KEY: Your Bedrock API key (required)
    - OPENAI_BASE_URL: Mantle endpoint URL (required)
    """
    pass


@cli.command("list-models")
def list_models():
    """
    List available models for Bedrock Mantle.

    Models listed here are available for both the Responses API and Chat Completions API.
    The same set of models is supported by both APIs.
    """
    try:
        client = create_client()
    except click.ClickException:
        raise  # Let credential errors propagate with their original message

    base_url = os.environ.get("OPENAI_BASE_URL", "")
    click.echo(f"Endpoint: {base_url}")
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
def chat(model: str, no_stream: bool, completions: bool, background: bool, system: str):
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
        client = create_client()
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


if __name__ == "__main__":
    cli()
