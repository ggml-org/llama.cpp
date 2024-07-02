# run with: ./tests.sh --no-skipped --tags passkey
@passkey
@slow
Feature: Passkey / Self-extend with context shift

  Background: Server startup
    Given a server listening on localhost:8080

  # Generates a long text of junk and inserts a secret passkey number inside it.
  # Then we query the LLM for the secret passkey.
  # see #3856 and #4810
  Scenario Outline: Passkey
    Given a model file <hf_file> from HF repo <hf_repo>
    And   <n_batch> as batch size
    And   <n_junk> as number of junk
    And   <n_predicted> server max tokens to predict
    And   42 as seed
    And   <n_ctx> KV cache size
    And   1 slots
    And   <n_ga> group attention factor to extend context size through self-extend
    And   <n_ga_w> group attention width to extend context size through self-extend
    # Can be override with N_GPU_LAYERS
    And   <ngl> GPU offloaded layers
    Then  the server is starting
    Then  the server is healthy
    Given available models
    Then  model 0 is trained on <n_ctx_train> tokens context
    Given a prefix prompt:
    """
    here is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there.
    """
    And a passkey prompt template:
    """
    The pass key is <passkey> Remember it. <passkey> is the pass key.
    """
    And a junk suffix prompt:
    """
    The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.
    """
    And a suffix prompt:
    """
    What is the pass key? The pass key is
    """
    Given a "<passkey>" passkey challenge prompt with the passkey inserted every <i_pos> junk
    And  a completion request with no api error
    Then <n_predicted> tokens are predicted matching <re_content>

    Examples:
      | hf_repo                         | hf_file                     | n_ctx_train | ngl | n_ctx | n_batch | n_ga | n_ga_w | n_junk | i_pos | passkey | n_predicted | re_content     |
      | TheBloke/phi-2-GGUF             | phi-2.Q4_K_M.gguf           | 2048        | 5   | 8192  | 512     | 4    | 512    | 250    | 50    | 42      | 1           | 42             |
      | TheBloke/phi-2-GGUF             | phi-2.Q4_K_M.gguf           | 2048        | 5   | 8192  | 512     | 2    | 512    | 250    | 50    | 42      | 1           | \b((?!42)\w)+\b  |
      #| TheBloke/Llama-2-7B-GGUF        | llama-2-7b.Q2_K.gguf        | 4096        | 3   | 16384 | 512     | 4    | 512    | 500    | 300   | 1234    | 5           | 1234           |
      #| TheBloke/Mixtral-8x7B-v0.1-GGUF | mixtral-8x7b-v0.1.Q2_K.gguf | 32768       | 2   | 16384 | 512     | 4    | 512    | 500    | 100   | 0987    | 5           | 0
      # 987           |
