import {
  ToolCallRequest,
  ToolCallOutput,
  ToolCallParameters,
  ToolCallSpec,
} from '../types';

export abstract class AgentTool {
  constructor(
    public readonly id: string,
    private readonly isEnabledCallback: () => boolean,
    public readonly toolDescription: string,
    public readonly parameters: ToolCallParameters
  ) {}

  /**
   * "Public" wrapper for the tool call processing logic.
   * @param call The tool call object from the API response.
   * @returns The tool call output or undefined if the tool is not enabled.
   */
  public processCall(call: ToolCallRequest): ToolCallOutput | undefined {
    if (this.enabled) {
      return this._process(call);
    }

    return undefined;
  }

  /**
   * Whether calling this tool is enabled.
   * User can toggle the status from the settings panel.
   * @returns enabled status.
   */
  public get enabled(): boolean {
    return this.isEnabledCallback();
  }

  /**
   * Specifications for the tool call.
   * https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md
   * https://platform.openai.com/docs/guides/function-calling?api-mode=responses#defining-functions
   */
  public get specs(): ToolCallSpec {
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
  protected abstract _process(call: ToolCallRequest): ToolCallOutput;
}
