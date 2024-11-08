import pandas as pd
import numpy as np
import anthropic
from dotenv import load_dotenv
from eval_utils.anthropic_model import get_anthropic_chat_completion, parse_anthropic_completion

class GetModel():

    def __init__(self, model):
        self.model = model 


class EvaluateModel():
    """
    Evaluate a language model for a given truth-seeking method. 
    """

    def __init__(self, model, method, second_model=False):
        self.model = model
        self.method = method 
        self.second_model = second_model

    def run_eval(self, message):
        if self.model == "Anthropic":
            client = anthropic.Anthropic()
            messages = [
                {
                    "role": "user",
                    "content": message
                }
            ]
            response = get_anthropic_chat_completion(
                            client=client, 
                            messages=messages, 
                            model_name="claude-3-sonnet-20240229",
                            max_new_tokens=1000  # This maps to max_tokens in the API
                        )
            print(parse_anthropic_completion(response))


if __name__ == "__main__":
    eval_model = EvaluateModel(model="Anthropic", method="debate", second_model=False)
    eval_model.run_eval("Hello are you ready to debate?")
    




