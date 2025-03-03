#!/usr/bin/env python3
# function calling using llama-cli

import subprocess
import sys
import select
import os
import re

import json

import functions
from function_tool import get_function_tool_json, get_chat_tool_format

function_name_list = [ name for name in dir(functions) if not name.startswith('_') ]
function_lookup = { name: getattr(functions, name) for name in function_name_list }
tools = [ get_function_tool_json(f) for (n, f) in function_lookup.items() ]

def main():
    import argparse

    parser = argparse.ArgumentParser(epilog='For more options: llama-cli --help')
    parser.add_argument('--display-prompt', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--special', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--reverse-prompt', type=str)
    parser.add_argument('-m', '--model', type=str, default='model.gguf')
    parser.add_argument('--ctx-size', type=int, default=1024)
    args, other_args = parser.parse_known_args()

    tool_format = get_chat_tool_format(args, tools)
    if args.reverse_prompt is None: args.reverse_prompt = tool_format['user_start']

    if args.display_prompt: print(tool_format['prompt'])

    command = [ './llama-cli', '-i', '-p', tool_format['prompt'], '--model', args.model, '--reverse-prompt', args.reverse_prompt, '--escape', '--special', '--no-display-prompt', '--log-disable', '--simple-io', '--ctx-size',  str(args.ctx_size), *other_args]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if process.stdout is not None: os.set_blocking(process.stdout.fileno(), False)

    try:
        run_loop(process, args, tool_format)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        process.terminate()
        process.wait()

def run_loop(process, args, tool_format):
    pbuffer = ''
    skip_output_until_result = False
    while True:
        readable, _, _ = select.select([process.stdout, process.stderr, sys.stdin], [], [])

        for stream in readable:
            if stream == process.stdout:
                pdata = process.stdout.read()
                if not pdata: continue
                pbuffer += pdata

                if(match := re.search(tool_format['function_re'], pbuffer, re.S)):
                    if not args.special:
                        pdata = pdata[:match.pos]
                    pbuffer = ''
                    skip_output_until_result = False
                    try:
                        if 1 < len(match.groups()):
                            tool_name = match.group(1)
                            tool_args = json.loads(match.group(2))
                        else:
                            tool = json.loads(match.group(1))
                            tool_name = tool['name']
                            tool_args = tool['arguments']

                        if tool_name == 'python':
                            result = functions._run_python(tool_args);
                        else:
                            result = function_lookup[tool_name](**tool_args)
                    except ValueError as e:
                        result = {'error': 'unknown'}

                    result = tool_format['tool_start'] + json.dumps(result) + tool_format['tool_end']
                    process.stdin.write(result + '\n')
                    process.stdin.flush()
                    if(args.special): pdata += '\n' + result
                elif (n := pdata.find(tool_format['function_marker'])) >= 0:
                    if not args.special:
                        pdata = pdata[:n]
                        skip_output_until_result = True
                elif skip_output_until_result:
                    pdata = ''

                if not args.special:
                    pdata = re.sub(r'<\|[^\|>]*\|>', '', pdata)
                sys.stdout.write(pdata)
                sys.stdout.flush()

            elif stream == sys.stdin:
                user_input = sys.stdin.readline()
                if user_input:
                    user_input = user_input.rstrip()
                    process.stdin.write(user_input + tool_format['user_end'] + '\n')
                    process.stdin.flush()

if __name__ == '__main__':
    main()

