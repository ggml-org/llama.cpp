```mermaid
flowchart TB
    subgraph Routes["Routes"]
        R1["/ (Welcome)"]
        R2["/chat/[id]"]
        R3["/skills"]
        RL["+layout.svelte"]
    end

    subgraph Components["Components"]
        C_Sidebar["ChatSidebar"]
        C_Screen["ChatScreen"]
        C_Form["ChatForm"]
        C_Messages["ChatMessages"]
        C_Message["ChatMessage"]
        C_ChatMessageAgenticContent["ChatMessageAgenticContent"]
        C_SkillToolResult["ChatMessageToolCallBlockSkill"]
        C_MessageEditForm["ChatMessageEditForm"]
        C_ModelsSelector["ModelsSelector"]
        C_Settings["ChatSettings"]
        C_SkillCatalog["SkillCatalog"]
        C_SkillCatalogList["SkillCatalogList"]
        C_SkillDetail["SkillDetail"]
        C_SkillPicker["ChatFormSkillPicker"]
        C_McpSettings["McpServersSettings"]
        C_McpResourceBrowser["McpResourceBrowser"]
        C_McpServersSelector["McpServersSelector"]
    end

    subgraph Hooks["Hooks"]
        H1["useModelsSelector"]
        H2["useProcessingState"]
    end

    subgraph Stores["Stores"]
        S1["chatStore<br/><i>Chat interactions and streaming</i>"]
        SA["agenticStore<br/><i>Multi-turn agentic loop orchestration</i>"]
        S2["conversationsStore<br/><i>Conversation data, messages and MCP overrides</i>"]
        S3["modelsStore<br/><i>Model selection and loading</i>"]
        S4["serverStore<br/><i>Server props and role detection</i>"]
        S5["settingsStore<br/><i>User configuration including Skills budget</i>"]
        S6["mcpStore<br/><i>MCP servers, tools, prompts</i>"]
        S7["mcpResourceStore<br/><i>MCP resources and attachments</i>"]
        S8["toolsStore<br/><i>Tool registry and Skills settings</i>"]
        S9["skillsStore<br/><i>CWD catalog and run snapshots</i>"]
        S10["skillActivationStore<br/><i>Durable conversation activations</i>"]
    end

    subgraph Services["Services"]
        SV1["ChatService"]
        SV2["ModelsService"]
        SV3["PropsService"]
        SV4["DatabaseService"]
        SV5["ParameterSyncService"]
        SV6["MCPService<br/><i>protocol operations</i>"]
        SV7["SkillsService<br/><i>catalog and reads</i>"]
        SV8["SkillsPackingService<br/><i>budget and tokenizer policy</i>"]
        SV9["SkillCommandService<br/><i>explicit activation</i>"]
    end

    subgraph Storage["Storage"]
        ST1["IndexedDB<br/><i>conversations, messages</i>"]
        ST2["LocalStorage<br/><i>config, userOverrides, mcpServers</i>"]
    end

    subgraph APIs["llama-server API"]
        API1["/v1/chat/completions"]
        API2["/props"]
        API3["/models/*"]
        API4["/v1/models"]
        API5["/skills"]
        API6["/skills/read"]
        API7["/tokenize"]
    end

    subgraph ExternalMCP["External MCP Servers"]
        EXT1["MCP Server 1<br/><i>WebSocket/HTTP/SSE</i>"]
        EXT2["MCP Server N"]
    end

    %% Routes -> Components
    R1 & R2 --> C_Screen
    R3 --> C_SkillCatalog
    RL --> C_Sidebar

    %% Layout runs MCP health checks; sidebar probes Skills availability
    RL --> S6

    %% Component hierarchy
    C_Screen --> C_Form & C_Messages & C_Settings
    C_Messages --> C_Message
    C_Message --> C_ChatMessageAgenticContent & C_SkillToolResult
    C_Message --> C_MessageEditForm
    C_Form & C_MessageEditForm --> C_ModelsSelector
    C_Form --> C_McpServersSelector & C_SkillPicker
    C_Settings --> C_McpSettings
    C_McpSettings --> C_McpResourceBrowser
    C_SkillCatalog --> C_SkillCatalogList & C_SkillDetail

    %% Components -> Hooks -> Stores
    C_Form & C_Messages --> H1 & H2
    H1 --> S3 & S4
    H2 --> S1 & S5

    %% Components -> Stores and Skills services
    C_Screen --> S1 & S2
    C_Sidebar --> S2 & S9
    C_ModelsSelector --> S3 & S4
    C_Settings --> S5 & S8 & S9
    C_McpSettings --> S6
    C_McpResourceBrowser --> S6 & S7
    C_McpServersSelector --> S6
    C_Form --> S6 & S8 & S9
    C_SkillCatalog --> S9 & SV8
    C_SkillDetail --> SV7
    C_Form --> SV9

    %% chatStore -> agenticStore -> MCP/Skills
    S1 --> SA
    SA --> SV1
    SA --> S6 & S8 & S9 & S10 & SV8

    %% Stores -> Services
    S1 --> SV1 & SV4
    S2 --> SV4
    S3 --> SV2 & SV3
    S4 --> SV3
    S5 --> SV5
    S6 --> SV6
    S7 --> SV6
    S9 --> SV7 & SV8
    S10 --> SV4

    %% Services -> Storage
    SV4 --> ST1
    SV5 --> ST2

    %% Services -> APIs
    SV1 --> API1
    SV2 --> API3 & API4
    SV3 --> API2
    SV7 --> API5 & API6
    SV8 --> API7
    SV9 --> SV7 & S10

    %% MCP -> External Servers
    SV6 --> EXT1 & EXT2

    %% Styling
    classDef routeStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef componentStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef hookStyle fill:#fff8e1,stroke:#ff8f00,stroke-width:2px
    classDef storeStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef serviceStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef storageStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef apiStyle fill:#e3f2fd,stroke:#1565c0,stroke-width:2px
    classDef externalStyle fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,stroke-dasharray: 5 5

    class R1,R2,R3,RL routeStyle
    class C_Sidebar,C_Screen,C_Form,C_Messages,C_Message,C_ChatMessageAgenticContent,C_SkillToolResult,C_MessageEditForm,C_ModelsSelector,C_Settings,C_SkillCatalog,C_SkillCatalogList,C_SkillDetail,C_SkillPicker,C_McpSettings,C_McpResourceBrowser,C_McpServersSelector componentStyle
    class H1,H2 hookStyle
    class S1,S2,S3,S4,S5,SA,S6,S7,S8,S9,S10 storeStyle
    class SV1,SV2,SV3,SV4,SV5,SV6,SV7,SV8,SV9 serviceStyle
    class ST1,ST2 storageStyle
    class API1,API2,API3,API4,API5,API6,API7 apiStyle
    class EXT1,EXT2 externalStyle
```
