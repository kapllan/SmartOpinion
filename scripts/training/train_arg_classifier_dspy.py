# Based on this code: https://github.com/saifulhaq95/DSPy-Indic/blob/main/indicxlni.ipynb
# There some examples from the community in the documentation page:https://dspy-docs.vercel.app/docs/tutorials/examples
import os
import openai

import glob
import os
import pandas as pd
import random

import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from opinion_analyzer.inference.client_handler import ClientHandler


if __name__ == "__main__":
    turbo = dspy.HFModel(model="VAGOsolutions/SauerkrautLM-Phi-3-medium")

    dspy.settings.configure(lm=turbo)

    # This is not required but it helps to understand what is happening
    my_example = {
        "question": "What game was Super Mario Bros. 2 based on?",
        "answer": "Doki Doki Panic",
    }

    # This is the signature for the predictor. It is a simple question and answer model.
    class BasicQA(dspy.Signature):
        """Answer questions about classic video games."""

        question = dspy.InputField(desc="a question about classic video games")
        answer = dspy.OutputField(desc="often between 1 and 5 words")

    # Define the predictor.
    generate_answer = dspy.Predict(BasicQA)

    # Call the predictor on a particular input.
    pred = generate_answer(question=my_example["question"])

    # Print the answer...profit :)
    print(pred.answer)
