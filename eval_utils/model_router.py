import asyncio
import datetime
from typing import List, Dict, Optional
from anthropic import Anthropic, AsyncAnthropic
import re
from eval_utils.anthropic_model import (
    get_anthropic_client_sync,
    get_anthropic_client_async,
    get_anthropic_chat_completion,
    get_anthropic_chat_completion_async,
    parse_anthropic_completion
)


class GetModel:
    """
    Given a model, load env and get chat completion.
    """
    
    def __init__(
        self,
        max_tokens_per_response: int = 1000,
        temperature: float = 0.7,
        model_name: str = "claude-3-sonnet-20240229",
        judge_criteria: Optional[str] = None,
        async_mode: bool = True
    ):
        self.max_tokens = max_tokens_per_response
        self.temperature = temperature
        self.model_name = model_name
        self.judge_criteria = judge_criteria or self._default_judge_criteria()
        self.debate_history = []
        self.async_mode = async_mode
        self.topic = None  
        self.judgment_result = None 
        
        if async_mode:
            self.agent_a = self.get_client_async()
            self.agent_b = self.get_client_async()
            self.judge = self.get_client_async()
        else:
            self.agent_a = self.get_client_sync()
            self.agent_b = self.get_client_sync()
            self.judge = self.get_client_sync()

    def get_client_async(self, model_name, async_mode):
        if model_name.contains("claude"): 
                self.agent_a = get_anthropic_client_async()
                self.agent_b = get_anthropic_client_async()
                self.judge = get_anthropic_client_async()
           
    def get_client_sync(self, model_name, async_mode):
        if model_name.contains("claude"): 
                self.agent_a = get_anthropic_client_sync()
                self.agent_b = get_anthropic_client_sync()
                self.judge = get_anthropic_client_sync()
           
