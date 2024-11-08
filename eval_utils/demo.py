from anthropic_model import get_anthropic_chat_completion, parse_anthropic_completion
from anthropic import Anthropic, AsyncAnthropic
import anthropic

client = anthropic.Anthropic()

messages = [
    {
        "role": "user",
        "content": "Hello model"
    }
]

res = get_anthropic_chat_completion(
    client=client, 
    messages=messages, 
    model_name="claude-3-sonnet-20240229",
    max_new_tokens=1000  # This maps to max_tokens in the API
)

print(parse_anthropic_completion(res))