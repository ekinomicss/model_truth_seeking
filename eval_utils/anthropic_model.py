import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from anthropic import Anthropic, AsyncAnthropic
from anthropic._types import NOT_GIVEN
from anthropic.types import Message, MessageParam
from .env_vars import ENV

def get_anthropic_client_sync() -> Anthropic:
    """Get a synchronous Anthropic client."""
    if ENV.ANTHROPIC_API_KEY is None:
        raise Exception("Missing Anthropic API key.")
    return Anthropic(api_key=ENV.ANTHROPIC_API_KEY)

def get_anthropic_client_async() -> AsyncAnthropic:
    """Get an asynchronous Anthropic client."""
    if ENV.ANTHROPIC_API_KEY is None:
        raise Exception("Missing Anthropic API key.")
    return AsyncAnthropic(api_key=ENV.ANTHROPIC_API_KEY)

def get_anthropic_chat_completion(
    client: Anthropic,
    messages: list[dict],
    model_name: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
) -> Message:
    """
    Makes a single message request to Anthropic's API.
    """
    return client.messages.create(
        messages=messages,
        model=model_name,
        max_tokens=max_new_tokens,
        temperature=temperature
    )

async def get_anthropic_chat_completion_async(
    client: AsyncAnthropic,
    messages: list[dict],
    model_name: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
) -> Message:
    """
    Makes a single message async request to Anthropic's API.
    """
    return await client.messages.create(
        messages=messages,
        model=model_name,
        max_tokens=max_new_tokens,
        temperature=temperature
    )

def parse_anthropic_completion(response: Message | None) -> str | None:
    """
    Parse the response from Anthropic API to extract the text content.
    """
    if not response:
        return None
    try:
        message_content = response.content[0]
        return message_content.text if message_content.type == "text" else None
    except (AttributeError, IndexError):
        return None