import { isDev } from '../../Config';
import { AgentTool } from './agent_tool';
import { JSReplAgentTool } from './js_repl_tool';

/**
 * Map of available tools for function calling.
 * Note that these tools are not necessarily enabled by the user.
 */
export const AVAILABLE_TOOLS = new Map<string, AgentTool>();

function registerTool(tool: AgentTool): AgentTool {
  AVAILABLE_TOOLS.set(tool.id, tool);
  if (isDev)
    console.log(
      `Successfully registered tool: ${tool.id}, enabled: ${tool.isEnabled()}`
    );
  return tool;
}

// Available agent tools
export const jsReplTool = registerTool(new JSReplAgentTool());
