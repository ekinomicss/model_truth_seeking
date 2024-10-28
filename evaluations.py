import pandas as pd
import numpy as np
import openAI
from typing import Optional


class EvaluateModel():
    """
    Evaluate a language model for a given truth-seeking method. 
    """

    def __init__(self, model, method, second_model=False):
        self.model = model
        self.method = method 
        self.second_model = second_model

    def run_eval(self):
        pass
    
    def plots(self):
        pass



if __name__ == "__main__":
    eval_model = EvaluateModel(model="OpenAI", method="debate", second_model=True)
    pass 
    




