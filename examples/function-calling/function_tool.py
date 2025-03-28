# Generate function calling definitions function schemas

import inspect
import re

import json

# Extract OpenAI function calling style definitions from functions
#
# Generated with: Create a python function to to generate the OpenAI function calling definition from a given function, getting the description, parameter type and parameter description from the function documentation, assuming the function documentation contains sphynx style parameter descriptions, marked with :param.
def get_function_tool_json(func):
    typemap = { 'str': 'string' };
    def get_type(s):
        return typemap[s] if s in typemap else s

    function_name = func.__name__
    doc_parts = re.split(r'\n\s*:param[^:]*\s+', func.__doc__.rstrip());

    function_description = doc_parts[0]
    params_doc = [ re.split(r'\:\s*', param_doc, maxsplit=1) for param_doc in doc_parts[1:] ]
    params_doc = { param: desc for param, desc in params_doc }

    function_def = {
        'name': function_name,
        'description': function_description,
        'parameters': { 'type': 'object', 'properties': {}, 'required': [] }
    }

    for param_name, param in inspect.signature(func).parameters.items():
        function_def['parameters']['properties'][param_name] = {
            'type' : get_type(param.annotation.__name__) if param.annotation is not param.empty else '',
            'description': params_doc[param_name] if param_name in params_doc else ''
        }
        function_def['parameters']['required'].append(param_name);

    return function_def

# Generate function definition schema from function definitions
#
# This is from llama-cpp-python, llama_chat_format.py
def generate_functionary_schema_from_functions(functions, namespace="functions") -> str:
    schema = (
        "// Supported function definitions that should be called when necessary.\n"
    )
    schema += f"namespace {namespace} {{\n\n"

    for function in functions:
        function_name = function["name"]
        description = function.get("description", "")
        parameters = function.get("parameters", {})
        required_params = parameters.get("required", [])

        schema += f"// {description}\n"
        schema += f"type {function_name} = (_: {{\n"

        for param_name, param in parameters.get("properties", {}).items():
            param_description = param.get("description", "")
            param_type = param.get("type", "any")
            optional_indicator = "" if param_name in required_params else "?"
            schema += f"// {param_description}\n"
            schema += f"{param_name}{optional_indicator}: {param_type},\n"
        schema += "}) => any;\n\n"

    schema += "}} // namespace {}".format(namespace)
    return schema

def generate_simple_schema_from_functions(functions) -> str:
    return '\n'.join([json.dumps(function).replace('{', '{ ').replace('}', ' }') for function in functions])

functionary_prompt_start = """<|start_header_id|>system<|end_header_id|>

You are capable of executing available function(s) if required.
Execute function(s) as needed.
The function calls are not shown in the conversation and should be called covertly to answer questions.
Ask for the required input to:recipient==all
Use JSON for function arguments.
Respond in this format:
>>>${recipient}
${content}
Available functions:
"""
functionary_prompt_end = """<|eot_id|><|start_header_id|>system<|end_header_id|>

When you send a message containing Python code to python, it will be executed in a stateful Jupyter notebook environment. python will respond with the output of the execution or time out after 60.0 seconds. The drive at '/mnt/data' can be used to save and persist user files.<|eot_id|><|start_header_id|>user<|end_header_id|>
"""

simple_prompt_start = """<s><|user|> You are a helpful assistant with access to the following functions. Use them if required - """
simple_prompt_end = """<|end|>"""

def get_chat_tool_format(args, tools):
    if 'functionary' in args.model.lower():
        return {
            'prompt': functionary_prompt_start + generate_functionary_schema_from_functions(tools) + functionary_prompt_end,
            'function_marker': '>>>',
            'function_re': r'>>>([^\n]*)\n(.*)<\|eot_id\|>',
            'user_start': '<|start_header_id|>user<|end_header_id|>\n',
            'user_end': '<|eot_id|><|start_header_id|>assistant<|end_header_id|>' + '\n',
            'tool_start': '',
            'tool_end': '<|eot_id|><|start_header_id|>assistant<|end_header_id|>'
        }
    else:
        return {
            'prompt': simple_prompt_start + generate_simple_schema_from_functions(tools) + simple_prompt_end,
            'function_marker': '<functioncall>',
            'function_re': r'<functioncall> \n?(.*)<\|end\|>',
            'user_start': '<|user|> ',
            'user_end': '<|end|>' + '\n',
            'tool_start': '<|user|>',
            'tool_end': '<|end|> <|assistant|>'
        }
