import { AvailableToolId } from '../types';
import { AgentTool } from './agent_tool';
import { JSReplAgentTool } from './js_repl_tool';

/**
 * Map of available tools for function calling.
 * Note that these tools are not necessarily enabled by the user.
 */
export const AVAILABLE_TOOLS = new Map<AvailableToolId, AgentTool>();

function registerTool<T extends AgentTool>(tool: T): T {
  AVAILABLE_TOOLS.set(tool.id, tool);
  return tool;
}

// Available agent tools
export const jsReplTool = registerTool(new JSReplAgentTool());
