import {
  ToolCallRequest,
  ToolCallOutput,
  ToolCallParameters,
  ToolCallSpec,
  AvailableToolId,
} from '../types';

export abstract class AgentTool {
  constructor(
    public readonly id: AvailableToolId,
    public readonly name: string,
    public readonly toolDescription: string,
    public readonly parameters: ToolCallParameters,
    private readonly _enabled: () => boolean
  ) {}

  /**
   * "Public" wrapper for the tool call processing logic.
   * @param call The tool call object from the API response.
   * @returns The tool call output or undefined if the tool is not enabled.
   */
  public async processCall(
    call: ToolCallRequest
  ): Promise<ToolCallOutput | undefined> {
    if (this.enabled) {
      try {
        return await this._process(call);
      } catch (error) {
        console.error(`Error processing tool call for ${this.id}:`, error);
        return {
          type: 'function_call_output',
          call_id: call.call_id,
          output: `Error during tool execution: ${(error as Error).message}`,
        } as ToolCallOutput;
      }
    }
    return undefined;
  }

  /**
   * Whether calling this tool is enabled.
   * User can toggle the status from the settings panel.
   * @returns enabled status.
   */
  public get enabled(): boolean {
    return this._enabled();
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
  protected abstract _process(call: ToolCallRequest): Promise<ToolCallOutput>;
}
