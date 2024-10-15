# run with: ./tests.sh --no-skipped --tags wrong_usage
@wrong_usage
Feature: Wrong usage of llama.cpp server

  #3969 The user must always set --n-predict option
  # to cap the number of tokens any completion request can generate
  # or pass n_predict/max_tokens in the request.
  Scenario: Infinite loop
    Given a server listening on localhost:8080
    And   a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   42 as server seed
    And   2048 KV cache size
    # Uncomment below to fix the issue
    #And   64 server max tokens to predict
    Then  the server is starting
    Then  the server is healthy
    Given a prompt:
      """
      Go to: infinite loop
      """
    # Uncomment below to fix the issue
    #And   128 max tokens to predict
    Given concurrent completion requests
    Then the server is idle
    Then all prompts are predicted
