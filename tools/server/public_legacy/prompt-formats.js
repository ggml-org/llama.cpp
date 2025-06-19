// extended list
export const promptFormats = {
  "alpaca": {
  template: `{{prompt}}\n\n{{history}}\n\n{{char}}:`,

  historyTemplate: `### {{name}}:\n{{message}}`,

  char: "Response",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "Instruction",
  userMsgPrefix: "",
  userMsgSuffix: "",

  stops: ""
  },

  // ----------------------------

  "chatml": {
  template: `<|im_start|>system\n{{prompt}}<|im_end|>\n{{history}}{{char}}`,

  historyTemplate: `<|im_start|>{{name}}\n{{message}}`,

  char: "assistant",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "user",
  userMsgPrefix: "",
  userMsgSuffix: "<|im_end|>\n",

  stops: ""
  },

  // ----------------------------

  "commandr": {
  template: `<BOS_TOKEN><|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{{prompt}}\n<|END_OF_TURN_TOKEN|>{{history}}{{char}}`,

  historyTemplate: `<|START_OF_TURN_TOKEN|><|{{name}}|> {{message}}`,

  char: "CHATBOT_TOKEN",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "USER_TOKEN",
  userMsgPrefix: "",
  userMsgSuffix: "<|END_OF_TURN_TOKEN|>",

  stops: ""
  },
  // ref: https://docs.cohere.com/docs/prompting-command-r

  // ----------------------------

  "llama2": {
  template: `<s>[INST] <<SYS>>\n{{prompt}}\n<</SYS>>\n\nTest Message [/INST] Test Successfull </s>{{history}}{{char}}`,

  historyTemplate: `{{name}}: {{message}}`,

  char: "Assistant",
  charMsgPrefix: "",
  charMsgSuffix: "</s>",

  user: "User",
  userMsgPrefix: "<s>[INST] ",
  userMsgSuffix: " [/INST]",

  stops: ""
  },
  // ref: https://huggingface.co/blog/llama2#how-to-prompt-llama-2

  // ----------------------------

  "llama3": {
  template: `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{{prompt}}{{history}}{{char}}`,

  historyTemplate: `<|start_header_id|>{{name}}<|end_header_id|>\n\n{{message}}<|eot_id|>`,

  char: "assistant",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "user",
  userMsgPrefix: "",
  userMsgSuffix: "",

  stops: "<|eot_id|>"
  },
  // ref: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/#special-tokens-used-with-meta-llama-3

  // ----------------------------

  "openchat": {
  template: `{{history}}{{char}}`,

  historyTemplate: `GPT4 Correct {{name}}: {{message}}<|end_of_turn|>`,

  char: "Assistant",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "User",
  userMsgPrefix: "",
  userMsgSuffix: "",

  stops: ""
  },

  // ----------------------------

  "phi3": {
  template: `{{history}}{{char}}`,

  historyTemplate: `<|{{name}}|>\n{{message}}<|end|>\n`,

  char: "assistant",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "user",
  userMsgPrefix: "",
  userMsgSuffix: "",

  stops: "<|end|>"
  },
  // ref: https://huggingface.co/microsoft/Phi-3-mini-4k-instruct#chat-format

  // ----------------------------

  "vicuna": {
  template: `{{prompt}}\n{{history}}{{char}}`,

  historyTemplate: `{{name}}: {{message}}\n`,

  char: "ASSISTANT",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "USER",
  userMsgPrefix: "",
  userMsgSuffix: "",

  stops: ""
  },
  // ref: https://huggingface.co/lmsys/vicuna-33b-v1.3/discussions/1

  // ----------------------------

  "deepseekCoder": {
  template: `{{prompt}}{{history}}{{char}}:`,

  historyTemplate: `### {{name}}:\n{{message}}`,

  char: "Response",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "Instruction",
  userMsgPrefix: "",
  userMsgSuffix: "",

  stops: "<|EOT|>"
  },

  // ----------------------------

  "med42": {
  template: `<|system|>: {{prompt}}\n{{history}}{{char}}`,

  historyTemplate: `<|{{name}}|>: {{message}}\n`,

  char: "assistant",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "prompter",
  userMsgPrefix: "",
  userMsgSuffix: "",

  stops: ""
  },

  // ----------------------------

  "neuralchat": {
  template: `### System:\n{{prompt}}\n{{history}}{{char}}:`,

  historyTemplate: `### {{name}}:\n{{message}}\n`,

  char: "Assistant",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "User",
  userMsgPrefix: "",
  userMsgSuffix: "",

  stops: ""
  },

  // ----------------------------

  "nousHermes": {
  template: `### Instruction: {{prompt}}\n\n{{history}}\n\n{{char}}:`,

  historyTemplate: `### {{name}}:\n{{message}}`,

  char: "Response",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "Input",
  userMsgPrefix: "",
  userMsgSuffix: "",

  stops: ""
  },

  // ----------------------------

  "openchatMath": {
  template: `{{history}}{{char}}`,

  historyTemplate: `Math Correct {{name}}: {{message}}<|end_of_turn|>`,

  char: "Assistant",
  charMsgPrefix: "",
  charMsgSuffix: "",


  user: "User",
  userMsgPrefix: "",
  userMsgSuffix: "",

  stops: ""
  },

  // ----------------------------

  "orion": {
  template: `<s>Human: Test Message\n\nAssistant: </s>Test Successful</s>{{history}}{{char}}:`,

  historyTemplate: `{{name}}: {{message}}`,

  char: "Assistant </s>",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "Human",
  userMsgPrefix: "",
  userMsgSuffix: "\n\n",

  stops: ""
  },

  // ----------------------------

  "sauerkraut": {
  template: `{{prompt}}\n{{history}}{{char}}`,

  historyTemplate: `
  {{name}}: {{message}}\n`,

  char: "Assistant",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "User",
  userMsgPrefix: "",
  userMsgSuffix: "",

  stops: ""
  },

  // ----------------------------

  "starlingCode": {
  template: `{{history}}{{char}}`,

  historyTemplate: `Code {{name}}: {{message}}<|end_of_turn|>`,

  char: "Assistant",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "User",
  userMsgPrefix: "",
  userMsgSuffix: "",

  stops: ""
  },

  // ----------------------------

  "yi34b": {
  template: `{{history}} {{char}}`,

  historyTemplate: `{{name}}: {{message}}`,

  char: "Assistant",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "Human",
  userMsgPrefix: "",
  userMsgSuffix: "",

  stops: ""
  },

  // ----------------------------

  "zephyr": {
  template: `<|system|>\n{{prompt}}</s>\n{{history}}{{char}}`,

  historyTemplate: `<|{{name}}|>\n{{message}}</s>\n`,

  char: "assistant",
  charMsgPrefix: "",
  charMsgSuffix: "",

  user: "user",
  userMsgPrefix: "",
  userMsgSuffix: "",

  stops: ""
  }
  };
