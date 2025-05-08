import { ToolCall, ToolCallOutput, ToolCallSpec } from '../types';

/**
 * Map of available tools for function calling.
 * Note that these tools are not necessarily enabled by the user.
 */
export const AVAILABLE_TOOLS = new Map<string, AgentTool>();

export abstract class AgentTool {
  id: string;
  isEnabled: () => boolean;

  constructor(id: string, enabled: () => boolean) {
    this.id = id;
    this.isEnabled = enabled;
    AVAILABLE_TOOLS.set(id, this);
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
  public abstract specs(): ToolCallSpec;

  /**
   * The actual tool call processing logic.
   * @param call: The tool call object from the API response.
   */
  protected abstract _process(call: ToolCall): ToolCallOutput;
}
