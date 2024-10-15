@llama.cpp
@rerank
Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model url https://huggingface.co/ggml-org/models/resolve/main/jina-reranker-v1-tiny-en/ggml-model-f16.gguf
    And   a model file jina-reranker-v1-tiny-en.gguf
    And   a model alias jina-reranker-v1-tiny-en
    And   42 as server seed
    And   2 slots
    And   512 as batch size
    And   512 as ubatch size
    And   512 KV cache size
    And   enable reranking endpoint
    Then  the server is starting
    Then  the server is healthy

  Scenario: Rerank
    Given a rerank query:
      """
      Machine learning is
      """
    And   a rerank document:
      """
      A machine is a physical system that uses power to apply forces and control movement to perform an action. The term is commonly applied to artificial devices, such as those employing engines or motors, but also to natural biological macromolecules, such as molecular machines.
      """
    And   a rerank document:
      """
      Learning is the process of acquiring new understanding, knowledge, behaviors, skills, values, attitudes, and preferences. The ability to learn is possessed by humans, non-human animals, and some machines; there is also evidence for some kind of learning in certain plants.
      """
    And   a rerank document:
      """
      Machine learning is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions.
      """
    And   a rerank document:
      """
      Paris, capitale de la France, est une grande ville européenne et un centre mondial de l'art, de la mode, de la gastronomie et de la culture. Son paysage urbain du XIXe siècle est traversé par de larges boulevards et la Seine.
      """
    When  reranking request
    Then  reranking results are returned
    Then  reranking highest score is index 2 and lowest score is index 3
