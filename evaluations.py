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

    def __init__(self, model: str, method: str, second_model: bool = False):
        self.model = model
        self.method = method 
        self.second_model = second_model

    def run_eval(self, message: str):
        if self.model == "Anthropic":
            mainDebater = anthropic.Anthropic()
            otherDebater = anthropic.Anthropic()
            messages = [
                {
                    "role": "user",
                    "content": message
                }
            ]
            response = get_anthropic_chat_completion(
                            client=mainDebater, 
                            messages=messages, 
                            model_name="claude-3-sonnet-20240229",
                            max_new_tokens=1000  
                        )
            print("-->MAIN MODEL:\n----------\n" + parse_anthropic_completion(response) + "\n")
            responses = [
                {
                    "role": "user",
                    "content": "Can you debate this response?" + parse_anthropic_completion(response)
                }
            ]
            response = get_anthropic_chat_completion(
                            client=otherDebater, 
                            messages=responses, 
                            model_name="claude-3-sonnet-20240229",
                            max_new_tokens=1000 
                        )
            print("-->OTHER MODEL:\n-----------\n" + parse_anthropic_completion(response))


if __name__ == "__main__":
    eval_model = EvaluateModel(model="Anthropic", method="debate", second_model=False)
    eval_model.run_eval("Hi! We will be debating about whether November is a rainy season month in Costa Rica, based on information from the internet and form an argument.")