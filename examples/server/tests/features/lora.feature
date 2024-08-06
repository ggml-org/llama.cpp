@llama.cpp
@lora
Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model url https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/stories15M_MOE-F16.gguf
    And   a model file stories15M_MOE-F16.gguf
    And   a model alias stories15M_MOE
    And   a lora adapter file from https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf
    And   42 as server seed
    And   1024 as batch size
    And   1024 as ubatch size
    And   2048 KV cache size
    And   64 max tokens to predict
    And   0.0 temperature
    Then  the server is starting
    Then  the server is healthy

  Scenario: Completion LoRA disabled
    Given switch off lora adapter 0
    Given a prompt:
    """
    Look in thy glass
    """
    And   a completion request with no api error
    Then  64 tokens are predicted matching little|girl|three|years|old

  Scenario: Completion LoRA enabled
    Given switch on lora adapter 0
    Given a prompt:
    """
    Look in thy glass
    """
    And   a completion request with no api error
    Then  64 tokens are predicted matching eye|love|glass|sun
