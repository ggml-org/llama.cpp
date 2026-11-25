// Golden inputs and expectations from Miles #1233 at 13aff7f82d3d98298c28ea05ba1f70459833b16f.
// Source: tests/fast/test_flint_grammar_backend.py and test_llg_proxy_stream.py.
// Fixture data is independent of the C++ grammar and parser.
#pragma once

static constexpr const char * flint_python_golden = R"golden(
{
  "reference": "https://github.com/recursive-org/miles/pull/1233",
  "revision": "13aff7f82d3d98298c28ea05ba1f70459833b16f",
  "tools": {
    "read": {
      "type": "object",
      "properties": {
        "filePath": {
          "type": "string"
        }
      },
      "required": [
        "filePath"
      ]
    },
    "editor": {
      "type": "object",
      "properties": {
        "path": {
          "type": "string",
          "minLength": 1
        },
        "old_text": {
          "type": "string"
        },
        "new_text": {
          "type": "string"
        }
      },
      "required": [
        "path",
        "new_text"
      ],
      "additionalProperties": false
    },
    "run_commands": {
      "type": "object",
      "properties": {
        "commands": {
          "type": "array",
          "items": {
            "type": "string"
          }
        }
      },
      "required": [
        "commands"
      ],
      "additionalProperties": false
    },
    "tool_search": {
      "type": "object",
      "properties": {
        "queries": {
          "anyOf": [
            {
              "type": "array",
              "items": {
                "type": "object",
                "required": [
                  "query"
                ],
                "properties": {
                  "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 512
                  }
                }
              },
              "maxItems": 16
            },
            {
              "type": "null"
            }
          ]
        }
      }
    }
  },
  "markers": [
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<arg_key>",
    "</arg_key>",
    "<arg_value>",
    "</arg_value>",
    "<|begin_of_text|>",
    "<|end_of_text|>",
    "<|system|>",
    "<|developer|>",
    "<|user|>",
    "<|assistant|>",
    "<|observation|>"
  ],
  "valid": [
    {
      "id": "plain-answer",
      "text": "The median bug is fixed."
    },
    {
      "id": "reasoning-then-call",
      "text": "<think>look at the file</think><tool_call>read<arg_key>filePath</arg_key><arg_value>calc/stats.py</arg_value></tool_call>"
    },
    {
      "id": "json-list",
      "text": "<tool_call>run_commands<arg_key>commands</arg_key><arg_value>[\"ls -la\", \"pytest -q\"]</arg_value></tool_call>"
    },
    {
      "id": "all-keys-in-order",
      "text": "<tool_call>editor<arg_key>path</arg_key><arg_value>a.py</arg_value><arg_key>old_text</arg_key><arg_value>x</arg_value><arg_key>new_text</arg_key><arg_value>y</arg_value></tool_call>"
    },
    {
      "id": "two-calls",
      "text": "<tool_call>read<arg_key>filePath</arg_key><arg_value>a</arg_value></tool_call><tool_call>read<arg_key>filePath</arg_key><arg_value>b</arg_value></tool_call>"
    },
    {
      "id": "at-maxLength",
      "text": "<tool_call>tool_search<arg_key>queries</arg_key><arg_value>[{\"query\": \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}]</arg_value></tool_call>"
    }
  ],
  "invalid": [
    {
      "id": "wrong-key",
      "text": "<tool_call>read<arg_key>path</arg_key><arg_value>calc/stats.py</arg_value></tool_call>"
    },
    {
      "id": "key-not-in-schema",
      "text": "<tool_call>run_commands<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
    },
    {
      "id": "unknown-tool",
      "text": "<tool_call>shell<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call>"
    },
    {
      "id": "truncated-tool-name",
      "text": "<tool_call>edito<arg_key>path</arg_key><arg_value>a.py</arg_value><arg_key>new_text</arg_key><arg_value>y</arg_value></tool_call>"
    },
    {
      "id": "tool-name-with-extra-suffix",
      "text": "<tool_call>read_files<arg_key>filePath</arg_key><arg_value>a.py</arg_value></tool_call>"
    },
    {
      "id": "tool-name-wrong-case",
      "text": "<tool_call>Read<arg_key>filePath</arg_key><arg_value>a.py</arg_value></tool_call>"
    },
    {
      "id": "bare-string-for-list",
      "text": "<tool_call>run_commands<arg_key>commands</arg_key><arg_value>ls -la</arg_value></tool_call>"
    },
    {
      "id": "whitespace-padding",
      "text": "<tool_call>run_commands<arg_key>commands</arg_key><arg_value>[   \n\n  ]</arg_value></tool_call>"
    },
    {
      "id": "below-minLength",
      "text": "<tool_call>editor<arg_key>path</arg_key><arg_value></arg_value><arg_key>new_text</arg_key><arg_value>y</arg_value></tool_call>"
    },
    {
      "id": "above-maxLength",
      "text": "<tool_call>tool_search<arg_key>queries</arg_key><arg_value>[{\"query\": \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}]</arg_value></tool_call>"
    },
    {
      "id": "missing-required",
      "text": "<tool_call>editor<arg_key>path</arg_key><arg_value>a.py</arg_value><arg_key>old_text</arg_key><arg_value>x</arg_value></tool_call>"
    },
    {
      "id": "marker-token-in-json-string",
      "text": "<tool_call>run_commands<arg_key>commands</arg_key><arg_value>[\"ls</arg_value></tool_call><tool_call>shell</arg_value></tool_call>"
    }
  ],
  "ordinary_markers": [
    {
      "id": "json-string",
      "prefix": "<tool_call>run_commands<arg_key>commands</arg_key><arg_value>[\"ls",
      "marker": "</arg_value>"
    },
    {
      "id": "string-value",
      "prefix": "<tool_call>read<arg_key>filePath</arg_key><arg_value>/repo/a",
      "marker": "</arg_value>"
    },
    {
      "id": "text",
      "prefix": "Let me look",
      "marker": "<tool_call>"
    },
    {
      "id": "reasoning",
      "prefix": "<think>hmm",
      "marker": "</think>"
    }
  ],
  "stream_tools": [
    {
      "type": "function",
      "function": {
        "name": "view",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string"
            }
          },
          "required": [
            "path"
          ]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "bash",
        "parameters": {
          "type": "object",
          "properties": {
            "command": {
              "type": "string"
            },
            "timeout": {
              "type": "integer"
            }
          },
          "required": [
            "command"
          ]
        }
      }
    }
  ],
  "stream_output": "<think>The user wants the tests run. I should read main.py first.</think>\n\nLet me look at the file and run the tests.\n\n<tool_call>read<arg_key>path</arg_key><arg_value>/repo/main.py</arg_value></tool_call><tool_call>bash<arg_key>command</arg_key><arg_value>pytest -q</arg_value><arg_key>timeout</arg_key><arg_value>30</arg_value></tool_call><|observation|>",
  "stream_expected": {
    "reasoning_content": "The user wants the tests run. I should read main.py first.",
    "content": "Let me look at the file and run the tests.",
    "tool_calls": [
      {
        "name": "view",
        "arguments": {
          "path": "/repo/main.py"
        }
      },
      {
        "name": "bash",
        "arguments": {
          "command": "pytest -q",
          "timeout": 30
        }
      }
    ]
  },
  "truncated": [
    {
      "id": "splitter-cut-key",
      "output": "  Running it.\n<tool_call>bash<arg_key>comm",
      "content": "Running it.",
      "reasoning_content": ""
    },
    {
      "id": "token-limit-content",
      "output": "<think>Reading main.py first.</think>\n\nThe file defines three funct",
      "content": "The file defines three funct",
      "reasoning_content": "Reading main.py first."
    },
    {
      "id": "token-limit-call",
      "output": "<think>Reading main.py first.</think>\n\nLet me look.\n\n<tool_call>read<arg_key>path</arg_key><arg_value>/repo/ma",
      "content": "Let me look.",
      "reasoning_content": "Reading main.py first."
    }
  ]
}
)golden";
