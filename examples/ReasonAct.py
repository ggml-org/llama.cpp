#!/bin/python
import sys, os, datetime
from common import GptParams
from low_level_api_chat_cpp import LLaMAInteract

def env_or_def(env, default):
    if (env in os.environ):
        return os.environ[env]
    return default

MODEL = env_or_def("MODEL", "./models/llama-13B/ggml-model.bin")

prompt=f"""You run in a loop of Thought, Action, Observation.
At the end of the loop either Answer or restate your Thought and Action.
Use Thought to describe your thoughts about the question you have been asked.
Use Action to run one of these actions available to you:
- calculate[python math expression]
Observation will be the result of running those actions


Question: What is 4 * 7 / 3?
Thought: Do I need to use an action? Yes, I use calculate to do math
Action: calculate[4 * 7 / 3]
Observation: 9.3333333333
Thought: Do I need to use an action? No, have the result
Answer: The calculate tool says it is 9.3333333333
Question: What is capital of france?
Thought: Do I need to use an action? No, I know the answer
Answer: Paris is the capital of France
Question:""" + " ".join(sys.argv[1:])

print("Loading model...")
params = GptParams(
    interactive=True,
    interactive_start=True,
    top_k=10000,
    temp=0.2,
    repeat_penalty=1,
    n_threads=7,
    n_ctx=2048,
    antiprompt=["Question:","Observation:"],
    model=MODEL,
    input_prefix=" ",
    n_predict=-1,
    prompt=prompt,
)

with LLaMAInteract(params) as m:
    m.interact()
