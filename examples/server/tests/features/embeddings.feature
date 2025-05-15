@llama.cpp
@embeddings
Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model url https://huggingface.co/ggml-org/models/resolve/main/bert-bge-small/ggml-model-f16.gguf
    And   a model file bert-bge-small.gguf
    And   a model alias bert-bge-small
    And   42 as server seed
    And   2 slots
    # the bert-bge-small model has context size of 512
    # since the generated prompts are as big as the batch size, we need to set the batch size to <= 512
    # ref: https://huggingface.co/BAAI/bge-small-en-v1.5/blob/5c38ec7c405ec4b44b94cc5a9bb96e735b38267a/config.json#L20
    And   128 as batch size
    And   128 as ubatch size
    And   512 KV cache size
    And   enable embeddings endpoint
    Then  the server is starting
    Then  the server is healthy

  Scenario: Embedding
    When embeddings are computed for:
    """
    What is the capital of Bulgaria ?
    """
    Then embeddings are generated

  Scenario: Embedding (error: prompt too long)
    When embeddings are computed for:
    """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
    Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
    Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
    Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
    Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
    """
    And  embeddings request with 500 api error

  Scenario: OAI Embeddings compatibility
    Given a model bert-bge-small
    When an OAI compatible embeddings computation request for:
    """
    What is the capital of Spain ?
    """
    Then embeddings are generated

  Scenario: OAI Embeddings compatibility with multiple inputs
    Given a model bert-bge-small
    Given a prompt:
      """
      In which country Paris is located ?
      """
    And a prompt:
      """
      Is Madrid the capital of Spain ?
      """
    When an OAI compatible embeddings computation request for multiple inputs
    Then embeddings are generated

  Scenario: Multi users embeddings
    Given a prompt:
      """
      Write a very long story about AI.
      """
    And a prompt:
      """
      Write another very long music lyrics.
      """
    And a prompt:
      """
      Write a very long poem.
      """
    And a prompt:
      """
      Write a very long joke.
      """
    Given concurrent embedding requests
    Then the server is busy
    Then the server is idle
    Then all embeddings are generated

  Scenario: Multi users OAI compatibility embeddings
    Given a prompt:
      """
      In which country Paris is located ?
      """
    And a prompt:
      """
      Is Madrid the capital of Spain ?
      """
    And a prompt:
      """
      What is the biggest US city ?
      """
    And a prompt:
      """
      What is the capital of Bulgaria ?
      """
    And   a model bert-bge-small
    Given concurrent OAI embedding requests
    Then the server is busy
    Then the server is idle
    Then all embeddings are generated

  Scenario: All embeddings should be the same
    Given 10 fixed prompts
    And   a model bert-bge-small
    Given concurrent OAI embedding requests
    Then all embeddings are the same
