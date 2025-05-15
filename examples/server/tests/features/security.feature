@llama.cpp
@security
Feature: Security

  Background: Server startup with an api key defined
    Given a server listening on localhost:8080
    And   a model file tinyllamas/stories260K.gguf from HF repo ggml-org/models
    And   a server api key THIS_IS_THE_KEY
    Then  the server is starting
    Then  the server is healthy

  Scenario Outline: Completion with some user api key
    Given a prompt test
    And   a user api key <api_key>
    And   4 max tokens to predict
    And   a completion request with <api_error> api error

    Examples: Prompts
      | api_key         | api_error |
      | THIS_IS_THE_KEY | no        |
      | THIS_IS_THE_KEY | no        |
      | hackeme         | raised    |
      |                 | raised    |

  Scenario Outline: OAI Compatibility
    Given a system prompt test
    And   a user prompt test
    And   a model test
    And   2 max tokens to predict
    And   streaming is disabled
    And   a user api key <api_key>
    Given an OAI compatible chat completions request with <api_error> api error

    Examples: Prompts
      | api_key         | api_error |
      | THIS_IS_THE_KEY | no        |
      | THIS_IS_THE_KEY | no        |
      | hackme          | raised    |

  Scenario Outline: OAI Compatibility (invalid response formats)
    Given a system prompt test
    And   a user prompt test
    And   a response format <response_format>
    And   a model test
    And   2 max tokens to predict
    And   streaming is disabled
    Given an OAI compatible chat completions request with raised api error

    Examples: Prompts
      | response_format                                       |
      | {"type": "sound"}                                     |
      | {"type": "json_object", "schema": 123}                |
      | {"type": "json_object", "schema": {"type": 123}}      |
      | {"type": "json_object", "schema": {"type": "hiccup"}} |


  Scenario Outline: CORS Options
    Given a user api key THIS_IS_THE_KEY
    When  an OPTIONS request is sent from <origin>
    Then  CORS header <cors_header> is set to <cors_header_value>

    Examples: Headers
      | origin          | cors_header                      | cors_header_value |
      | localhost       | Access-Control-Allow-Origin      | localhost         |
      | web.mydomain.fr | Access-Control-Allow-Origin      | web.mydomain.fr   |
      | origin          | Access-Control-Allow-Credentials | true              |
      | web.mydomain.fr | Access-Control-Allow-Methods     | POST              |
      | web.mydomain.fr | Access-Control-Allow-Headers     | *                 |
