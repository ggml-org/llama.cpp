#!/bin/bash
    # todo - auto generate settings, fix toggle stream response

curl http://localhost:8080/v1/chat/completions -d '{
"model": "gpt-3.5-turbo",
"tools": [
    {
    "type":"function",
    "function":{
        "name":"python",
        "description":"Runs code in an ipython interpreter and returns the result of the execution after 60 seconds.",
        "parameters":{
        "type":"object",
        "properties":{
            "code":{
            "type":"string",
            "description":"The code to run in the ipython interpreter."
            }
        },
        "required":["code"]
        }
    }
    }
],
"messages": [
    {
    "role": "user",
    "content": "What is 34245 * 3452484 - 12874387 + 24125^2?"
    },
    {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "type": "function",
            "call_id": "1",
            "function": {
              "name": "python",
              "arguments": "{\"code\":\"print(34245 * 3452484 - 12874387 + 24125^2)\"}"
            }
          }
        ]
    },
    {
      "role": "tool",
      "call_id": "1",
      "content": "118799455818"
    }
]
}'

#zed editor uses "system" instead of "tool"
