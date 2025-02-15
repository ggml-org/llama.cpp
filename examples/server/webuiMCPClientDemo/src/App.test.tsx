// App.test.tsx
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from './App';
import { McpSSEClient, McpServerConfig } from './mcpSSEClient';
import ChatComponent from './ChatComponent';
import ConfigComponent from './ConfigComponent';

// Mocking McpSSEClient methods
jest.mock('./mcpSSEClient', () => {
  class McpSSEClientMock {
    private _serverConfigs: McpServerConfig[];
    private _clients: any;

    constructor(serverConfigs: McpServerConfig[]) {
      this._serverConfigs = serverConfigs;
      this._clients = {};
    }

    async initializeClients() {
      for (const config of this._serverConfigs) {
        if (config.type !== "sse") {
          console.warn(`Unsupported transport type: ${config.type}. Skipping server ${config.name}.`);
          continue;
        }

        const client = new McpSSEClientMock(this._serverConfigs);
        const tools = [{ name: "sample-tool", description: "A sample tool", inputSchema: {}, serverName: config.name }];
        const prompts = [{ name: "sample-prompt", description: "A sample prompt", arguments: {}, serverName: config.name }];
        const resources = [{ uri: "file:///about.txt", name: "about", description: "A sample text resource named about", mimeType: "text/plain", serverName: config.name }];

        this._clients[config.name] = {
          client,
          tools,
          prompts,
          resources,
        };
      }
      return { success: true };
    }

    getAvailableTools() {
      return Object.values(this._clients).flatMap(clientInfo => clientInfo.tools);
    }

    getAvailablePrompts() {
      return Object.values(this._clients).flatMap(clientInfo => clientInfo.prompts);
    }

    getAvailableResources() {
      return Object.values(this._clients).flatMap(clientInfo => clientInfo.resources);
    }

    async callTool(params: any) {
      return { content: [{ type: "text", text: "Tool executed successfully" }] };
    }

    async getPrompt(params: any) {
      return { description: "Prompt executed successfully" };
    }

    async readResource(params: any) {
      return { content: [{ type: "text", text: "Resource read successfully" }] };
    }

    async closeAllConnections() {
      for (const clientInfo of Object.values(this._clients)) {
        await clientInfo.client.close();
      }
    }
  }

  return { McpSSEClient: McpSSEClientMock };
});

// Mocking ChatComponent and ConfigComponent
jest.mock('./ChatComponent', () => {
  const ChatComponentMock = ({ messages, input, setInput, onSendMessage }: any) => (
    <div>
      <div>
        {messages.map((msg: any, index: number) => (
          <div key={index}>{msg.role}: {msg.content}</div>
        ))}
      </div>
      <input type="text" value={input} onChange={(e: any) => setInput(e.target.value)} />
      <button onClick={onSendMessage}>Send</button>
    </div>
  );
  return ChatComponentMock;
});

jest.mock('./ConfigComponent', () => {
  const ConfigComponentMock = ({ serverConfigs, onConfigSave, onDeleteServer, availableTools, availablePrompts, availableResources, onToolCall, onPromptRun, updateAdditionalContext }: any) => (
    <div>
      <h2>Configure MCP Servers</h2>
      {serverConfigs.map((config: any, index: number) => (
        <div key={index}>
          <h3>{config.name}</h3>
          <button onClick={() => onToolCall({ name: "sample-tool", arguments: {} })}>Call Tool</button>
          <button onClick={() => onPromptRun({ name: "sample-prompt", arguments: {} })}>Run Prompt</button>
          <button onClick={() => onDeleteServer(index)}>Delete</button>
        </div>
      ))}
      <button onClick={() => onConfigSave(serverConfigs)}>Save Config</button>
    </div>
  );
  return ConfigComponentMock;
});

describe('App', () => {
  it('renders the App component', () => {
    render(<App />);
    expect(screen.getByText(/MCP Chat Interface/i)).toBeInTheDocument();
  });

  it('displays the initial connection status', () => {
    render(<App />);
    expect(screen.getByText(/Connection Status: Disconnected/i)).toBeInTheDocument();
  });

  it('updates the connection status to Connected upon successful initialization', async () => {
    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Connection Status: Connected/i)).toBeInTheDocument();
    });
  });

  it('updates the connection status to Error upon failed initialization', async () => {
    const McpSSEClientMock = McpSSEClient as jest.MockedClass<typeof McpSSEClient>;
    McpSSEClientMock.prototype.initializeClients = jest.fn().mockResolvedValue({ success: false, error: "Initialization failed" });

    render(<App />);
    await waitFor(() => {
      expect(screen.getByText(/Connection Status: Error/i)).toBeInTheDocument();
      expect(screen.getByText(/Error: Initialization failed/i)).toBeInTheDocument();
    });
  });

  it('renders the ConfigComponent and ChatComponent', () => {
    render(<App />);
    expect(screen.getByText(/Configure MCP Servers/i)).toBeInTheDocument();
    expect(screen.getByText(/You:/i)).toBeInTheDocument();
    expect(screen.getByText(/Assistant:/i)).toBeInTheDocument();
  });

  it('handles user input in the chat component', () => {
    render(<App />);
    const inputElement = screen.getByRole('textbox');
    fireEvent.change(inputElement, { target: { value: 'Hello Claude!' } });
    expect(inputElement).toHaveValue('Hello Claude!');
  });

  it('calls handleSendMessage when the Send button is clicked', async () => {
    const handleSendMessageMock = jest.fn();
    ChatComponent.mockImplementation(({ onSendMessage }: any) => (
      <div>
        <button onClick={onSendMessage}>Send</button>
      </div>
    ));

    render(<App />);
    const sendButton = screen.getByText(/Send/i);
    fireEvent.click(sendButton);

    await waitFor(() => {
      expect(handleSendMessageMock).toHaveBeenCalled();
    });
  });

  it('calls handleSendMessage when Enter is pressed in the chat input', async () => {
    const handleSendMessageMock = jest.fn();
    ChatComponent.mockImplementation(({ onSendMessage }: any) => (
      <div>
        <input type="text" onKeyPress={(e: any) => e.key === 'Enter' && onSendMessage()} />
      </div>
    ));

    render(<App />);
    const inputElement = screen.getByRole('textbox');
    fireEvent.change(inputElement, { target: { value: 'Hello Claude!' } });
    fireEvent.keyPress(inputElement, { key: 'Enter', code: 'Enter' });

    await waitFor(() => {
      expect(handleSendMessageMock).toHaveBeenCalled();
    });
  });

  it('calls handleToolCall when the Call Tool button is clicked', async () => {
    const handleToolCallMock = jest.fn();
    ConfigComponent.mockImplementation(({ onToolCall }: any) => (
      <div>
        <button onClick={() => onToolCall({ name: "sample-tool", arguments: {} })}>Call Tool</button>
      </div>
    ));

    render(<App />);
    const callToolButton = screen.getByText(/Call Tool/i);
    fireEvent.click(callToolButton);

    await waitFor(() => {
      expect(handleToolCallMock).toHaveBeenCalledWith({ name: "sample-tool", arguments: {} });
    });
  });

  it('calls handlePromptRun when the Run Prompt button is clicked', async () => {
    const handlePromptRunMock = jest.fn();
    ConfigComponent.mockImplementation(({ onPromptRun }: any) => (
      <div>
        <button onClick={() => onPromptRun({ name: "sample-prompt", arguments: {} })}>Run Prompt</button>
      </div>
    ));

    render(<App />);
    const runPromptButton = screen.getByText(/Run Prompt/i);
    fireEvent.click(runPromptButton);

    await waitFor(() => {
      expect(handlePromptRunMock).toHaveBeenCalledWith({ name: "sample-prompt", arguments: {} });
    });
  });

  it('calls handleConfigSave when the Save Config button is clicked', async () => {
    const handleConfigSaveMock = jest.fn();
    ConfigComponent.mockImplementation(({ onConfigSave }: any) => (
      <div>
        <button onClick={() => onConfigSave([{ name: "MCPServer", type: "sse", serverUrl: "http://localhost:8000/sse" }])}>Save Config</button>
      </div>
    ));

    render(<App />);
    const saveConfigButton = screen.getByText(/Save Config/i);
    fireEvent.click(saveConfigButton);

    await waitFor(() => {
      expect(handleConfigSaveMock).toHaveBeenCalledWith([{ name: "MCPServer", type: "sse", serverUrl: "http://localhost:8000/sse" }]);
    });
  });

  it('calls deleteServerConfig when the Delete button is clicked', async () => {
    const deleteServerConfigMock = jest.fn();
    ConfigComponent.mockImplementation(({ onDeleteServer }: any) => (
      <div>
        <button onClick={() => onDeleteServer(0)}>Delete</button>
      </div>
    ));

    render(<App />);
    const deleteButton = screen.getByText(/Delete/i);
    fireEvent.click(deleteButton);

    await waitFor(() => {
      expect(deleteServerConfigMock).toHaveBeenCalledWith(0);
    });
  });

  it('handles tool calls correctly', async () => {
    const handleToolCallMock = jest.fn().mockResolvedValue({ content: [{ type: "text", text: "Tool executed successfully" }] });
    ChatComponent.mockImplementation(({ onSendMessage }: any) => (
      <div>
        <button onClick={onSendMessage}>Send</button>
      </div>
    ));

    const McpSSEClientMock = McpSSEClient as jest.MockedClass<typeof McpSSEClient>;
    McpSSEClientMock.prototype.callTool = handleToolCallMock;

    render(<App />);
    const inputElement = screen.getByRole('textbox');
    fireEvent.change(inputElement, { target: { value: 'Call the sample tool.' } });
    const sendButton = screen.getByText(/Send/i);
    fireEvent.click(sendButton);

    await waitFor(() => {
      expect(handleToolCallMock).toHaveBeenCalledWith({ name: "sample-tool", arguments: {} });
    });
  });

  it('handles prompt runs correctly', async () => {
    const handlePromptRunMock = jest.fn().mockResolvedValue({ description: "Prompt executed successfully" });
    ChatComponent.mockImplementation(({ onSendMessage }: any) => (
      <div>
        <button onClick={onSendMessage}>Send</button>
      </div>
    ));

    const McpSSEClientMock = McpSSEClient as jest.MockedClass<typeof McpSSEClient>;
    McpSSEClientMock.prototype.getPrompt = handlePromptRunMock;

    render(<App />);
    const inputElement = screen.getByRole('textbox');
    fireEvent.change(inputElement, { target: { value: 'Run the sample prompt.' } });
    const sendButton = screen.getByText(/Send/i);
    fireEvent.click(sendButton);

    await waitFor(() => {
      expect(handlePromptRunMock).toHaveBeenCalledWith({ name: "sample-prompt", arguments: {} });
    });
  });

  it('updates additional context when resources are selected', async () => {
    const updateAdditionalContextMock = jest.fn();
    ConfigComponent.mockImplementation(({ updateAdditionalContext }: any) => (
      <div>
        <button onClick={() => updateAdditionalContext([{ uri: "file:///about.txt", description: "A sample text resource named about" }])}>Update Context</button>
      </div>
    ));

    render(<App />);
    const updateContextButton = screen.getByText(/Update Context/i);
    fireEvent.click(updateContextButton);

    await waitFor(() => {
      expect(updateAdditionalContextMock).toHaveBeenCalledWith([{ uri: "file:///about.txt", description: "A sample text resource named about" }]);
    });
  });
});

// Mock axios post to simulate server responses
jest.mock('axios', () => ({
  post: jest.fn().mockResolvedValue({
    data: {
      choices: [
        {
          message: {
            role: 'assistant',
            content: 'Assistant response here',
            finish_reason: 'stop',
          },
        },
      ],
    },
  }),
}));
