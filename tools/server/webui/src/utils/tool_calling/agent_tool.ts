import {
  ToolCall,
  ToolCallOutput,
  ToolCallParameters,
  ToolCallSpec,
} from '../types';

export abstract class AgentTool {
  id: string;
  isEnabled: () => boolean;
  toolDescription: string;
  parameters: ToolCallParameters;

  constructor(
    id: string,
    enabled: () => boolean,
    toolDescription: string,
    parameters: ToolCallParameters
  ) {
    this.id = id;
    this.isEnabled = enabled;
    this.toolDescription = toolDescription;
    this.parameters = parameters;
  }

  /**
   * "Public" wrapper for the tool call processing logic.
   * @param call The tool call object from the API response.
   * @returns The tool call output or undefined if the tool is not enabled.
   */
  public processCall(call: ToolCall): ToolCallOutput | undefined {
    if (this.enabled()) {
      return this._process(call);
    }

    return undefined;
  }

  /**
   * Whether calling this tool is enabled.
   * User can toggle the status from the settings panel.
   * @returns enabled status.
   */
  public enabled(): boolean {
    return this.isEnabled();
  }

  /**
   * Specifications for the tool call.
   * https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
   * https://platform.openai.com/docs/guides/function-calling?api-mode=responses#defining-functions
   */
  public specs(): ToolCallSpec {
    return {
      type: 'function',
      function: {
        name: this.id,
        description: this.toolDescription,
        parameters: this.parameters,
      },
    };
  }

  /**
   * The actual tool call processing logic.
   * @param call: The tool call object from the API response.
   */
  protected abstract _process(call: ToolCall): ToolCallOutput;
}
