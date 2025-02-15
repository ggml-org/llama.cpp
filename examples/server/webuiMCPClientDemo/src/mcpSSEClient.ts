// mcpSSEClient.ts
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { ListToolsResultSchema, CallToolRequest, CallToolResultSchema, ListPromptsResultSchema, ListResourcesResultSchema, GetPromptRequest } from "@modelcontextprotocol/sdk/types.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";

interface McpServerConfig {
  name: string;
  type: string;
  serverUrl: string;
}

interface McpClientDict {
  [serverName: string]: {
    client: Client;
    tools: any[];
    prompts: any[];
    resources: any[];
  };
}

export class McpSSEClient {
  private _clients: McpClientDict = {};
  private _toolServerMap: { [toolName: string]: string } = {};
  private _promptServerMap: { [promptName: string]: string } = {};
  private _resourceServerMap: { [resourceName: string]: string } = {};

  constructor(private _serverConfigs: McpServerConfig[]) {}

  async initializeClients(): Promise<{ success: boolean, error?: string }> {
    for (const config of this._serverConfigs) {
      if (config.type !== "sse") {
        console.warn(`Unsupported transport type: ${config.type}. Skipping server ${config.name}.`);
        continue;
      }

      const transport = new SSEClientTransport(new URL(config.serverUrl));
      const client = new Client({
          name: "MCP React Client",
          version: "1.0.0",
        },
        {
          capabilities: {
            tools: {},
            prompts: {},
            resources: {},
          },
        }
      );
      transport.onerror = (error) => {
        console.error('SSE transport error:', error);
        return { success: false, error: error.message };
      };

      transport.onclose = () => {
        console.log('SSE transport closed');
      };

      transport.onmessage = (message) => {
        console.log('Message received:', message);
      };

      try {
        await client.connect(transport);
        const toolsResult = await client.listTools();
        const promptsResult = await client.listPrompts();
        const resourcesResult = await client.listResources();

        this._clients[config.name] = {
          client,
          tools: toolsResult?.tools?.map(tool => ({ ...tool, serverName: config.name })) || [],
          prompts: promptsResult?.prompts?.map(prompt => ({ ...prompt, serverName: config.name })) || [],
          resources: resourcesResult?.resources?.map(resource => ({ ...resource, serverName: config.name })) || [],
        };

        // Create a mapping of tool names to their respective server names
        toolsResult?.tools?.forEach((tool) => {
          this._toolServerMap[tool.name] = config.name;
        });
        // Create a mapping of prompt names to their respective server names
        promptsResult?.prompts?.forEach((prompt) => {
          this._promptServerMap[prompt.name] = config.name;
        });
        // Create a mapping of resource names to their respective server names
        resourcesResult?.resources?.forEach((resource) => {
          this._resourceServerMap[resource.uri] = config.name;
        });

        console.log(`Initialized client for server ${config.name} successfully.`);
      } catch (error) {
        console.error(`Failed to initialize client for server ${config.name}:`, error);
        return { success: false, error: error.message };
      }
    }
    return { success: true };
  }

  getAvailableTools(): any[] {
    const tools = Object.values(this._clients).flatMap(clientInfo => clientInfo.tools);
    console.log(tools);
    return tools;
  }

  getAvailablePrompts(): any[] {
    const prompts = Object.values(this._clients).flatMap(clientInfo => clientInfo.prompts);
    console.log(prompts);
    return prompts;
  }

  getAvailableResources(): any[] {
    const resources = Object.values(this._clients).flatMap(clientInfo => clientInfo.resources);
    console.log(resources);
    return resources;
  }

  async callTool(params: CallToolRequest): Promise<CallToolResultSchema | undefined> {
    console.log(params);
    const serverName = this._toolServerMap[params.name];
    if (!serverName) {
      console.error(`Tool ${params.name} not found.`);
      return undefined;
    }
    const clientInfo = this._clients[serverName];
    if (!clientInfo) {
      console.error(`No client found for server ${serverName}.`);
      return undefined;
    }

    const tool = clientInfo.tools.find(tool => tool.name === params.name);
    if (!tool) {
      console.error(`Tool ${params.name} not found on server ${serverName}.`);
      return undefined;
    }
    try { 
      return await clientInfo.client.callTool(params, CallToolResultSchema);
    } catch (error) {
      console.error(`Error calling tool ${params.name} on server ${serverName}:`, error);
      return undefined;
    }
  }

  async getPrompt(params: GetPromptRequest["params"]): Promise<any> {
    const serverName = this._promptServerMap[params.name];
    if (!serverName) {
      console.error(`Prompt ${params.name} not found.`);
      return undefined;
    }
    const clientInfo = this._clients[serverName];
    if (!clientInfo) {
      console.error(`No client found for server ${serverName}.`);
      return undefined;
    }
    const prompt = clientInfo.prompts.find(prompt => prompt.name === params.name);
    if (!prompt) {
      console.error(`Prompt ${params.name} not found on server ${serverName}.`);
      return undefined;
    }
    try { 
      return await clientInfo.client.getPrompt(params);
    } catch (error) {
      console.error(`Error getting prompt ${params.name} from server ${serverName}:`, error);
      return undefined;
    }
  }

  async readResource(params: any): Promise<any> {
    const serverName = this._resourceServerMap[params.name];
    if (!serverName) {
      console.error(`Resource ${params.name} not found.`);
      return undefined;
    }
    const clientInfo = this._clients[serverName];
    if (!clientInfo) {
      console.error(`No client found for server ${serverName}.`);
      return undefined;
    }
    const resource = clientInfo.resources.find(resource => resource.name === params.name);
    if (!resource) {
      console.error(`Resource ${params.name} not found on server ${serverName}.`);
      return undefined;
    }
    try { 
      return await clientInfo.client.readResource(params);
    } catch (error) {
      console.error(`Error reading resource ${params.name} from server ${serverName}:`, error);
      return undefined;
    }
  }

  async closeAllConnections(): Promise<void> {
    for (const clientInfo of Object.values(this._clients)) {
      await clientInfo.client.close();
    }
  }
}
