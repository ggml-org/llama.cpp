@llama.cpp
@ctx_shift
Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   a model file test-model.gguf
    And   a model alias tinyllama-2
    And   BOS token is 1
    And   42 as server seed
    And   256 KV cache size
    And   32 as batch size
    And   2 slots

    # the prompt is 301 tokens
    # the slot context is 256/2 = 128 tokens
    # the prompt is truncated to keep the last 109 tokens
    # 64 tokens are generated thanks to shifting the context when it gets full
  Scenario: Inference with context shift
    And   64 server max tokens to predict
    Then  the server is starting
    Then  the server is healthy
    Given a prompt:
    """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
    Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    """
    And   a completion request with no api error
    Then  64 tokens are predicted matching fun|Annaks|popcorns|pictry|bowl
    And   the completion is  truncated
    And   109 prompt tokens are processed

  Scenario Outline: Inference without context shift
    And   <n_predict> server max tokens to predict
    And   disable context shifting
    Then  the server is starting
    Then  the server is healthy
    Given a prompt:
    """
    Hi how are you
    """
    And   a completion request with no api error
    Then  <n_token_output> tokens are predicted matching twind|Anna
    And   the completion is <truncated> truncated
    And   8 prompt tokens are processed
    Examples:
      | n_predict | n_token_output | truncated |
      | 64        | 64             | not       |
      | -1        | 120            |           |

  Scenario: Inference without context shift (expected error: prompt too long)
    And   disable context shifting
    Then  the server is starting
    Then  the server is healthy
    Given a prompt:
    """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
    Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    """
    And   a completion request with 400 api error

