// ConfigComponent.tsx
import React, { useState, useEffect } from 'react';

interface ConfigComponentProps {
  serverConfigs: McpServerConfig[];
  onConfigSave: (newConfigs: McpServerConfig[]) => void;
  onDeleteServer: (index: number) => void;
  availableTools: any[];
  availablePrompts: any[];
  availableResources: any[];
  onToolCall: (toolParams: CallToolRequest) => void;
  onPromptRun: (promptParams: GetPromptRequest["params"]) => void;
  updateAdditionalContext: (selectedResources: { uri: string, description?: string }[]) => void;
}

const ConfigComponent: React.FC<ConfigComponentProps> = ({ serverConfigs, onConfigSave, onDeleteServer, availableTools, availablePrompts, availableResources, onToolCall, onPromptRun, updateAdditionalContext }) => {
  const [configs, setConfigs] = useState<McpServerConfig[]>(serverConfigs);
  const [selectedResources, setSelectedResources] = useState<string[]>([]);
  const [toolArguments, setToolArguments] = useState<{ [toolName: string]: any }>({});
  const [promptArguments, setPromptArguments] = useState<{ [promptName: string]: any }>({});

  useEffect(() => {
    setConfigs(serverConfigs);
  }, [serverConfigs]);

  const handleInputChange = (index: number, field: string, value: string) => {
    const newConfigs = [...configs];
    newConfigs[index][field] = value;
    setConfigs(newConfigs);
  };

  const addServerConfig = () => {
    setConfigs([...configs, { name: '', type: 'sse', serverUrl: '' }]);
  };

  const saveConfig = () => {
    onConfigSave(configs);
  };

  const deleteConfig = (index: number) => {
    onDeleteServer(index);
  };

  const handleResourceSelect = (resourceUri: string, resourceDescription?: string) => {
  setSelectedResources(prevSelected => {
    const resource = { uri: resourceUri, description: resourceDescription };
    const index = prevSelected.findIndex((res: { uri: string }) => res.uri === resourceUri);
    if (index > -1) {
      prevSelected.splice(index, 1);
    } else {
      prevSelected.push(resource);
    }
    return [...prevSelected]; // Ensure state update triggers re-render
  });
};


  const handleToolArgumentChange = (toolName: string, argument: string, value: string) => {
    setToolArguments(prevArgs => ({
      ...prevArgs,
      [toolName]: {
        ...(prevArgs[toolName] || {}),
        [argument]: value,
      },
    }));
  };

  const handlePromptArgumentChange = (promptName: string, argument: string, value: string) => {
    setPromptArguments(prevArgs => ({
      ...prevArgs,
      [promptName]: {
        ...(prevArgs[promptName] || {}),
        [argument]: value,
      },
    }));
  };

  useEffect(() => {
    updateAdditionalContext(selectedResources);
  }, [selectedResources, updateAdditionalContext]);

  const renderServerDetails = (serverName: string) => {
    const serverTools = availableTools.filter(tool => tool.serverName === serverName);
    const serverPrompts = availablePrompts.filter(prompt => prompt.serverName === serverName);
    const serverResources = availableResources.filter(resource => resource.serverName === serverName);

    if (!serverTools.length && !serverPrompts.length && !serverResources.length) return null;

    return (
      <div style={{ marginBottom: '10px', textAlign: 'left' }}>
        <h3>{serverName} Details</h3>
        <div>
          <h4>Tools</h4>
          <ul>
            {serverTools.map((tool: any, index: number) => (
              <li key={index}>
                {tool.name}: {tool.description}
                {Object.keys(tool.inputSchema.properties).map((argKey) => (
                  <div key={argKey}>
                    <input
                      type="text"
                      placeholder={`Enter ${argKey}`}
                      value={toolArguments[tool.name]?.[argKey] || ''}
                      onChange={(e) => handleToolArgumentChange(tool.name, argKey, e.target.value)}
                      style={{ marginLeft: '10px', padding: '5px', marginRight: '5px' }}
                    />
                  </div>
                ))}
                <button onClick={() => onToolCall({ name: tool.name, arguments: toolArguments[tool.name] || {} })} style={{ marginLeft: '10px', padding: '5px 10px', cursor: 'pointer' }}>Call Tool</button>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4>Prompts</h4>
          <ul>
            {serverPrompts.map((prompt: any, index: number) => (
              <li key={index}>
                {prompt.name}: {prompt.description}
                {Object.keys(prompt.arguments || {}).map((argKey) => (
                  <div key={argKey}>
                    <input
                      type="text"
                      placeholder={`Enter ${argKey}`}
                      value={promptArguments[prompt.name]?.[argKey] || ''}
                      onChange={(e) => handlePromptArgumentChange(prompt.name, argKey, e.target.value)}
                      style={{ marginLeft: '10px', padding: '5px', marginRight: '5px' }}
                    />
                  </div>
                ))}
                <button onClick={() => onPromptRun({ name: prompt.name, arguments: promptArguments[prompt.name] || {} })} style={{ marginLeft: '10px', padding: '5px 10px', cursor: 'pointer' }}>Run Prompt</button>
              </li>
            ))}
          </ul>
        </div>
        <div>
          <h4>Resources</h4>
          <ul>
            {serverResources.map((resource: any, index: number) => (
              <li key={index}>
                <input 
                    type="checkbox" 
                    checked={selectedResources.some((res: { uri: string }) => res.uri === resource.uri)} 
                    onChange={() => handleResourceSelect(resource.uri, resource.description)} 
                  />
                {resource.name} ({resource.uri}): {resource.description}
              </li>
            ))}
          </ul>
        </div>
      </div>
    );
  };

  return (
    <div style={{ marginBottom: '20px' }}>
      <h2>Configure MCP Servers</h2>
      {configs.map((config, index) => (
        <div key={index} style={{ marginBottom: '10px', display: 'flex', alignItems: 'center' }}>
          <input
            type="text"
            placeholder="Server Name"
            value={config.name}
            onChange={(e) => handleInputChange(index, 'name', e.target.value)}
            style={{ width: '200px', padding: '5px', marginRight: '5px' }}
          />
          <input
            type="text"
            placeholder="Server URL"
            value={config.serverUrl}
            onChange={(e) => handleInputChange(index, 'serverUrl', e.target.value)}
            style={{ width: '300px', padding: '5px', marginRight: '5px' }}
          />
          <button onClick={() => deleteConfig(index)} style={{ padding: '5px 10px', backgroundColor: 'red', color: 'white', border: 'none', cursor: 'pointer' }}>Delete</button>
        </div>
      ))}
      <button onClick={addServerConfig} style={{ padding: '5px 10px', marginRight: '5px', cursor: 'pointer' }}>Add Server</button>
      <button onClick={saveConfig} style={{ padding: '5px 10px', cursor: 'pointer' }}>Save Config</button>
      {serverConfigs.map((config, index) => (
        <div key={index}>
          <h3>{config.name}</h3>
          {renderServerDetails(config.name)}
        </div>
      ))}
    </div>
  );
};

export default ConfigComponent;
