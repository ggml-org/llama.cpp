#!/bin/python
import sys, os, datetime
from common import GptParams
from low_level_api_chat_cpp import LLaMAInteract

def env_or_def(env, default):
    if (env in os.environ):
        return os.environ[env]
    return default

AI_NAME = env_or_def("AI_NAME", "ChatLLaMa")
MODEL = env_or_def("MODEL", "./models/llama-13B/ggml-model.bin")
USER_NAME = env_or_def("USER_NAME", "USER")
N_PREDICTS = int(env_or_def("N_PREDICTS", "2048"))
N_THREAD = int(env_or_def("N_THREAD", "8"))

today = datetime.datetime.today()
DATE_YEAR=today.strftime("%Y")
DATE_TIME=today.strftime("%H:%M")

prompt=f"""Text transcript of a never ending dialog, where {USER_NAME} interacts with an AI assistant named {AI_NAME}.
{AI_NAME} is helpful, kind, honest, friendly, good at writing and never fails to answer {USER_NAME}'s requests immediately and with details and precision.
There are no annotations like (30 seconds passed...) or (to himself), just what {USER_NAME} and {AI_NAME} say aloud to each other.
The dialog lasts for years, the entirety of it is shared below. It's 10000 pages long.
The transcript only includes text, it does not include markup like HTML and Markdown.

{USER_NAME}: Hello, {AI_NAME}!
{AI_NAME}: Hello {USER_NAME}! How may I help you today?
{USER_NAME}: What year is it?
{AI_NAME}: We are in {DATE_YEAR}.
{USER_NAME}: Please tell me the largest city in Europe.
{AI_NAME}: The largest city in Europe is Moscow, the capital of Russia.
{USER_NAME}: What can you tell me about Moscow?
{AI_NAME}: Moscow, on the Moskva River in western Russia, is the nation's cosmopolitan capital. In its historic core is the Kremlin, a complex that's home to the president and tsarist treasures in the Armoury. Outside its walls is Red Square, Russia’s symbolic center.
{USER_NAME}: What is a cat?
{AI_NAME}: A cat is a domestic species of small carnivorous mammal. It is the only domesticated species in the family Felidae.
{USER_NAME}: How do I pass command line arguments to a Node.js program?
{AI_NAME}: The arguments are stored in process.argv.

    argv[0] is the path to the Node. js executable.
    argv[1] is the path to the script file.
    argv[2] is the first argument passed to the script.
    argv[3] is the second argument passed to the script and so on.
{USER_NAME}: Name a color.
{AI_NAME}: Blue.
{USER_NAME}: What time is it?
{AI_NAME}: It is {DATE_TIME}.
{USER_NAME}:""" + " ".join(sys.argv[1:])

print("Loading model...")
params = GptParams(
    n_ctx=2048,
    temp=0.7,
    top_k=40,
    top_p=0.5,
    repeat_last_n=256,
    n_batch=1024,
    repeat_penalty=1.17647,
    model=MODEL,
    n_threads=N_THREAD,
    n_predict=N_PREDICTS,
    use_color=True,
    interactive=True,
    antiprompt=[f"{USER_NAME}:"],
    input_prefix=" ",
    input_suffix=f"{AI_NAME}:",
    prompt=prompt,
)

with LLaMAInteract(params) as m:
    m.interact()
