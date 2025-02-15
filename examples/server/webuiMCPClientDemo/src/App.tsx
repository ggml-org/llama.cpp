// App.tsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { McpSSEClient, McpServerConfig } from './mcpSSEClient.ts';
import ChatComponent from './ChatComponent.tsx';
import ConfigComponent from './ConfigComponent.tsx';

const App: React.FC = () => {
  const [messages, setMessages] = useState<{ role: string, content: string }[]>([]);
  const [input, setInput] = useState<string>('');
  const [tools, setTools] = useState<any[]>([]);
  const [prompts, setPrompts] = useState<any[]>([]);
  const [resources, setResources] = useState<any[]>([]);
  const [serverConfigs, setServerConfigs] = useState<McpServerConfig[]>([
    {"name": "MCPServer", "type": "sse", "serverUrl": "http://localhost:8000/sse"},
    {"name": "MCPServer2", "type": "sse", "serverUrl": "http://localhost:8001/sse"},
  ]);
  const [mcpClient, setMcpClient] = useState<McpSSEClient | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<string>('Disconnected');
  const [error, setError] = useState<string | undefined>(undefined);
  const [additionalContext, setAdditionalContext] = useState<{ uri: string, description?: string }[]>([]);
  const maxToolCalls = 5; // Maximum number of tool calls to prevent infinite loops

  useEffect(() => {
    const client = new McpSSEClient(serverConfigs);
    client.initializeClients().then((result) => {
      if (result.success) {
        setTools(client.getAvailableTools());
        setPrompts(client.getAvailablePrompts());
        setResources(client.getAvailableResources());
        setMcpClient(client);
        setConnectionStatus('Connected');
      } else {
        setConnectionStatus('Error');
        setError(result.error);
      }
    });

    return () => {
      if (mcpClient) {
        mcpClient.closeAllConnections();
      }
    };
  }, [serverConfigs]);

  const handleSendMessage = async () => {
    let newMessages = [...messages];
    let toolCallCount = 0;

    if (input !== '') {
      newMessages = [...newMessages, { role: 'user', content: input }];
      setMessages(newMessages);
      setInput('');
    }

    // Add additional context as separate messages
    additionalContext.forEach((resource) => {
      newMessages = [...newMessages, { role: 'assistant', content: `Additional Context: ${resource.description} (URI: ${resource.uri})` }];
    });

    try {
      while (true) {
        console.log("NewMessages:")
        console.log(newMessages)
        var response = await axios.post('http://localhost:8080/v1/chat/completions', {
          messages: newMessages,
          model: '',
          tools: tools.map(tool => ({
            type: 'function',
            function: {
              name: tool.name,
              description: tool.description,
              parameters: tool.inputSchema,
            },
          })),
        });

        console.log(response.data.choices[0]);
        var finishReason = response.data.choices[0].finish_reason;

        if (finishReason === 'tool_calls') {
          console.log("Processing tool call");

          const toolCalls = response.data.choices[0].message.tool_calls;
          
          const calltoolMessageContent = {
              role: "assistant",
              "tool_calls": toolCalls,
              content: '',
            };
          newMessages = [...newMessages, calltoolMessageContent];
          for (const toolCall of toolCalls) {
            if (toolCallCount >= maxToolCalls) {
              console.error("Max tool calls reached. Stopping further tool calls.");
              setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: 'Max tool calls reached. Stopping further tool calls.' }]);
              return;
            }

            const toolParams: CallToolRequest = {
              name: toolCall.function.name,
              arguments: JSON.parse(toolCall.function.arguments || '{}'),
            };

            const toolResponse = await mcpClient?.callTool(toolParams);
            
            const toolMessageContent = {
              tool_call_id: toolCall.id,
              role: "tool",
              name: toolCall.function.name,
              content: toolResponse?.content[0]?.text || 'Tool failed to execute.',
            };

            newMessages = [...newMessages, toolMessageContent];
            finishReason = "";
            response = "";
            toolCallCount++;
          }
        } else if (finishReason === 'stop' || finishReason === 'length') {
          // Append the assistant's message to the message history
          const content = response.data.choices[0].message.content;
          newMessages = [...newMessages, { role: 'assistant', content }];
          break;
        }
      }

      setMessages(newMessages);
    } catch (error) {
      console.error('Error sending message:', error);
      setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: 'Failed to send message.' }]);
      setConnectionStatus('Error');
      setError(error.message);
    }
  };

  const handleToolCall = async (toolParams: CallToolRequest) => {
    if (window.confirm(`Do you want to call the tool "${toolParams.name}" with arguments ${JSON.stringify(toolParams.arguments)}?`)) {
      const toolResponse = await mcpClient?.callTool(toolParams);
      if (toolResponse) {
        setMessages(prevMessages => [...prevMessages, { role: 'tool', content: toolResponse.content[0].text }]);
      } else {
        setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: 'Tool failed to execute.' }]);
      }
    }
  };

  const handlePromptRun = async (promptParams: GetPromptRequest["params"]) => {
    if (window.confirm(`Do you want to run the prompt "${promptParams.name}" with arguments ${JSON.stringify(promptParams.arguments)}?`)) {
      const promptResponse = await mcpClient?.getPrompt(promptParams);
      if (promptResponse) {
        setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: promptResponse.description }]);
      } else {
        setMessages(prevMessages => [...prevMessages, { role: 'assistant', content: 'Prompt failed to execute.' }]);
      }
    }
  };

  const handleConfigSave = (newConfigs: McpServerConfig[]) => {
    setServerConfigs(newConfigs);
    if (mcpClient) {
      mcpClient.closeAllConnections().then(() => {
        const newClient = new McpSSEClient(newConfigs);
        newClient.initializeClients().then((result) => {
          if (result.success) {
            setTools(newClient.getAvailableTools());
            setPrompts(newClient.getAvailablePrompts());
            setResources(newClient.getAvailableResources());
            setMcpClient(newClient);
            setConnectionStatus('Connected');
          } else {
            setConnectionStatus('Error');
            setError(result.error);
          }
        });
      });
    }
  };

  const deleteServerConfig = (index: number) => {
    const newConfigs = serverConfigs.filter((_, i) => i !== index);
    setServerConfigs(newConfigs);
    if (mcpClient) {
      mcpClient.closeAllConnections().then(() => {
        const newClient = new McpSSEClient(newConfigs);
        newClient.initializeClients().then((result) => {
          if (result.success) {
            setTools(newClient.getAvailableTools());
            setPrompts(newClient.getAvailablePrompts());
            setResources(newClient.getAvailableResources());
            setMcpClient(newClient);
            setConnectionStatus('Connected');
          } else {
            setConnectionStatus('Error');
            setError(result.error);
          }
        });
      });
    }
  };

  const updateAdditionalContext = (selectedResources: { uri: string, description?: string }[]) => {
    setAdditionalContext(selectedResources);
  };

  return (
    <div>
      <h1>MCP Chat Interface</h1>
      <p>Connection Status: {connectionStatus}</p>
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      <ConfigComponent 
        serverConfigs={serverConfigs} 
        onConfigSave={handleConfigSave} 
        onDeleteServer={deleteServerConfig} 
        availableTools={tools}
        availablePrompts={prompts}
        availableResources={resources}
        onToolCall={handleToolCall}
        onPromptRun={handlePromptRun}
        updateAdditionalContext={updateAdditionalContext}
      />
      <ChatComponent 
        messages={messages} 
        input={input} 
        setInput={setInput} 
        onSendMessage={handleSendMessage} 
      />
    </div>
  );
};

export default App;
