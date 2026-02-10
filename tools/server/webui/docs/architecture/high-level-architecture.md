```mermaid
flowchart TB
subgraph Routes["📍 Routes"]
R1["/ (+page.svelte)"]
R2["/chat/[id]"]
RL["+layout.svelte"]
end

    subgraph Components["🧩 Components"]
        direction TB
        subgraph LayoutComponents["Layout"]
            C_Sidebar["ChatSidebar"]
            C_Screen["ChatScreen"]
        end
        subgraph ChatUIComponents["Chat UI"]
            C_Form["ChatForm"]
            C_Messages["ChatMessages"]
            C_Message["ChatMessage"]
            C_MessageUser["ChatMessageUser"]
            C_MessageEditForm["ChatMessageEditForm"]
            C_Attach["ChatAttachments"]
            C_ModelsSelector["ModelsSelector"]
            C_Settings["ChatSettings"]
        end
        subgraph MCPComponents["MCP UI"]
            C_McpSettings["McpServersSettings"]
            C_McpServerCard["McpServerCard"]
            C_McpResourceBrowser["McpResourceBrowser"]
            C_McpResourcePreview["McpResourcePreview"]
            C_McpServersSelector["McpServersSelector"]
        end
    end

    subgraph Hooks["🪝 Hooks"]
        H1["useModelChangeValidation"]
        H2["useProcessingState"]
        H3["isMobile"]
    end

    subgraph Stores["🗄️ Stores"]
        direction TB
        subgraph S1["chatStore"]
            S1State["<b>State:</b><br/>isLoading, currentResponse<br/>errorDialogState<br/>activeProcessingState<br/>chatLoadingStates<br/>chatStreamingStates<br/>abortControllers<br/>processingStates<br/>activeConversationId<br/>isStreamingActive"]
            S1LoadState["<b>Loading State:</b><br/>setChatLoading()<br/>isChatLoading()<br/>syncLoadingStateForChat()<br/>clearUIState()<br/>isChatLoadingPublic()<br/>getAllLoadingChats()<br/>getAllStreamingChats()"]
            S1ProcState["<b>Processing State:</b><br/>setActiveProcessingConversation()<br/>getProcessingState()<br/>clearProcessingState()<br/>getActiveProcessingState()<br/>updateProcessingStateFromTimings()<br/>getCurrentProcessingStateSync()<br/>restoreProcessingStateFromMessages()"]
            S1Stream["<b>Streaming:</b><br/>streamChatCompletion()<br/>startStreaming()<br/>stopStreaming()<br/>stopGeneration()<br/>isStreaming()"]
            S1Error["<b>Error Handling:</b><br/>showErrorDialog()<br/>dismissErrorDialog()<br/>isAbortError()"]
            S1Msg["<b>Message Operations:</b><br/>addMessage()<br/>sendMessage()<br/>updateMessage()<br/>deleteMessage()<br/>getDeletionInfo()"]
            S1Regen["<b>Regeneration:</b><br/>regenerateMessage()<br/>regenerateMessageWithBranching()<br/>continueAssistantMessage()"]
            S1Edit["<b>Editing:</b><br/>editAssistantMessage()<br/>editUserMessagePreserveResponses()<br/>editMessageWithBranching()<br/>clearEditMode()<br/>isEditModeActive()<br/>getAddFilesHandler()<br/>setEditModeActive()"]
            S1Utils["<b>Utilities:</b><br/>getApiOptions()<br/>parseTimingData()<br/>getOrCreateAbortController()<br/>getConversationModel()"]
        end
        subgraph SA["agenticStore"]
            SAState["<b>State:</b><br/>sessions (Map)<br/>isAnyRunning"]
            SASession["<b>Session Management:</b><br/>getSession()<br/>updateSession()<br/>clearSession()<br/>getActiveSessions()<br/>isRunning()<br/>currentTurn()<br/>totalToolCalls()<br/>lastError()<br/>streamingToolCall()"]
            SAConfig["<b>Configuration:</b><br/>getConfig()<br/>maxTurns, maxToolPreviewLines"]
            SAFlow["<b>Agentic Loop:</b><br/>runAgenticFlow()<br/>executeAgenticLoop()<br/>normalizeToolCalls()<br/>emitToolCallResult()<br/>extractBase64Attachments()"]
        end
        subgraph S2["conversationsStore"]
            S2State["<b>State:</b><br/>conversations<br/>activeConversation<br/>activeMessages<br/>isInitialized<br/>pendingMcpServerOverrides<br/>titleUpdateConfirmationCallback"]
            S2Lifecycle["<b>Lifecycle:</b><br/>initialize()<br/>loadConversations()<br/>clearActiveConversation()"]
            S2ConvCRUD["<b>Conversation CRUD:</b><br/>createConversation()<br/>loadConversation()<br/>deleteConversation()<br/>deleteAll()<br/>updateConversationName()<br/>updateConversationTitleWithConfirmation()"]
            S2MsgMgmt["<b>Message Management:</b><br/>refreshActiveMessages()<br/>addMessageToActive()<br/>updateMessageAtIndex()<br/>findMessageIndex()<br/>sliceActiveMessages()<br/>removeMessageAtIndex()<br/>getConversationMessages()"]
            S2Nav["<b>Navigation:</b><br/>navigateToSibling()<br/>updateCurrentNode()<br/>updateConversationTimestamp()"]
            S2McpOverrides["<b>MCP Per-Chat Overrides:</b><br/>getMcpServerOverride()<br/>getAllMcpServerOverrides()<br/>setMcpServerOverride()<br/>toggleMcpServerForChat()<br/>removeMcpServerOverride()<br/>isMcpServerEnabledForChat()<br/>clearPendingMcpServerOverrides()"]
            S2Export["<b>Import/Export:</b><br/>downloadConversation()<br/>exportAllConversations()<br/>importConversations()<br/>importConversationsData()<br/>triggerDownload()"]
            S2Utils["<b>Utilities:</b><br/>setTitleUpdateConfirmationCallback()"]
        end
        subgraph S3["modelsStore"]
            S3State["<b>State:</b><br/>models, routerModels<br/>selectedModelId<br/>selectedModelName<br/>loading, updating, error<br/>modelLoadingStates<br/>modelPropsCache<br/>modelPropsFetching<br/>propsCacheVersion"]
            S3Getters["<b>Computed Getters:</b><br/>selectedModel<br/>loadedModelIds<br/>loadingModelIds<br/>singleModelName"]
            S3Modal["<b>Modalities:</b><br/>getModelModalities()<br/>modelSupportsVision()<br/>modelSupportsAudio()<br/>getModelModalitiesArray()<br/>getModelProps()<br/>updateModelModalities()"]
            S3Status["<b>Status Queries:</b><br/>isModelLoaded()<br/>isModelOperationInProgress()<br/>getModelStatus()<br/>isModelPropsFetching()"]
            S3Fetch["<b>Data Fetching:</b><br/>fetch()<br/>fetchRouterModels()<br/>fetchModelProps()<br/>fetchModalitiesForLoadedModels()"]
            S3Select["<b>Model Selection:</b><br/>selectModelById()<br/>selectModelByName()<br/>clearSelection()<br/>findModelByName()<br/>findModelById()<br/>hasModel()"]
            S3LoadUnload["<b>Loading/Unloading Models:</b><br/>loadModel()<br/>unloadModel()<br/>ensureModelLoaded()<br/>waitForModelStatus()<br/>pollForModelStatus()"]
            S3Utils["<b>Utilities:</b><br/>toDisplayName()<br/>clear()"]
        end
        subgraph S4["serverStore"]
            S4State["<b>State:</b><br/>props<br/>loading, error<br/>role<br/>fetchPromise"]
            S4Getters["<b>Getters:</b><br/>defaultParams<br/>contextSize<br/>isRouterMode<br/>isModelMode"]
            S4Data["<b>Data Handling:</b><br/>fetch()<br/>getErrorMessage()<br/>clear()"]
            S4Utils["<b>Utilities:</b><br/>detectRole()"]
        end
        subgraph S5["settingsStore"]
            S5State["<b>State:</b><br/>config<br/>theme<br/>isInitialized<br/>userOverrides"]
            S5Lifecycle["<b>Lifecycle:</b><br/>initialize()<br/>loadConfig()<br/>saveConfig()<br/>loadTheme()<br/>saveTheme()"]
            S5Update["<b>Config Updates:</b><br/>updateConfig()<br/>updateMultipleConfig()<br/>updateTheme()"]
            S5Reset["<b>Reset:</b><br/>resetConfig()<br/>resetTheme()<br/>resetAll()<br/>resetParameterToServerDefault()"]
            S5Sync["<b>Server Sync:</b><br/>syncWithServerDefaults()<br/>forceSyncWithServerDefaults()"]
            S5Utils["<b>Utilities:</b><br/>getConfig()<br/>getAllConfig()<br/>getParameterInfo()<br/>getParameterDiff()<br/>getServerDefaults()<br/>clearAllUserOverrides()"]
        end
        subgraph S6["mcpStore"]
            S6State["<b>State:</b><br/>isInitializing, error<br/>toolCount, connectedServers<br/>healthChecks (Map)<br/>connections (Map)<br/>toolsIndex (Map)"]
            S6Lifecycle["<b>Lifecycle:</b><br/>ensureInitialized()<br/>initialize()<br/>shutdown()<br/>acquireConnection()<br/>releaseConnection()"]
            S6Health["<b>Health Checks:</b><br/>runHealthCheck()<br/>runHealthChecksForServers()<br/>updateHealthCheck()<br/>getHealthCheckState()<br/>clearHealthCheck()"]
            S6Servers["<b>Server Management:</b><br/>getServers()<br/>addServer()<br/>updateServer()<br/>removeServer()<br/>getServerById()<br/>getServerDisplayName()"]
            S6Tools["<b>Tool Operations:</b><br/>getToolDefinitionsForLLM()<br/>getToolNames()<br/>hasTool()<br/>getToolServer()<br/>executeTool()<br/>executeToolByName()"]
            S6Prompts["<b>Prompt Operations:</b><br/>getAllPrompts()<br/>getPrompt()<br/>hasPromptsCapability()<br/>getPromptCompletions()"]
        end
        subgraph S7["mcpResourceStore"]
            S7State["<b>State:</b><br/>serverResources (Map)<br/>cachedResources (Map)<br/>subscriptions (Map)<br/>attachments[]<br/>isLoading"]
            S7Resources["<b>Resource Discovery:</b><br/>setServerResources()<br/>getServerResources()<br/>getAllResourceInfos()<br/>getAllTemplateInfos()<br/>clearServerResources()"]
            S7Cache["<b>Caching:</b><br/>cacheResourceContent()<br/>getCachedContent()<br/>invalidateCache()<br/>clearCache()"]
            S7Subs["<b>Subscriptions:</b><br/>addSubscription()<br/>removeSubscription()<br/>isSubscribed()<br/>handleResourceUpdate()"]
            S7Attach["<b>Attachments:</b><br/>addAttachment()<br/>updateAttachmentContent()<br/>removeAttachment()<br/>clearAttachments()<br/>toMessageExtras()"]
        end

        subgraph ReactiveExports["⚡ Reactive Exports"]
            direction LR
            subgraph ChatExports["chatStore"]
                RE1["isLoading()"]
                RE2["currentResponse()"]
                RE3["errorDialog()"]
                RE4["activeProcessingState()"]
                RE5["isChatStreaming()"]
                RE6["isChatLoading()"]
                RE7["getChatStreaming()"]
                RE8["getAllLoadingChats()"]
                RE9["getAllStreamingChats()"]
                RE9a["isEditModeActive()"]
                RE9b["getAddFilesHandler()"]
                RE9c["setEditModeActive()"]
                RE9d["clearEditMode()"]
            end
            subgraph AgenticExports["agenticStore"]
                REA1["agenticIsRunning()"]
                REA2["agenticCurrentTurn()"]
                REA3["agenticTotalToolCalls()"]
                REA4["agenticLastError()"]
                REA5["agenticStreamingToolCall()"]
                REA6["agenticIsAnyRunning()"]
            end
            subgraph ConvExports["conversationsStore"]
                RE10["conversations()"]
                RE11["activeConversation()"]
                RE12["activeMessages()"]
                RE13["isConversationsInitialized()"]
            end
            subgraph ModelsExports["modelsStore"]
                RE15["modelOptions()"]
                RE16["routerModels()"]
                RE17["modelsLoading()"]
                RE18["modelsUpdating()"]
                RE19["modelsError()"]
                RE20["selectedModelId()"]
                RE21["selectedModelName()"]
                RE22["selectedModelOption()"]
                RE23["loadedModelIds()"]
                RE24["loadingModelIds()"]
                RE25["propsCacheVersion()"]
                RE26["singleModelName()"]
            end
            subgraph ServerExports["serverStore"]
                RE27["serverProps()"]
                RE28["serverLoading()"]
                RE29["serverError()"]
                RE30["serverRole()"]
                RE31["defaultParams()"]
                RE32["contextSize()"]
                RE33["isRouterMode()"]
                RE34["isModelMode()"]
            end
            subgraph SettingsExports["settingsStore"]
                RE35["config()"]
                RE36["theme()"]
                RE37["isInitialized()"]
            end
            subgraph MCPExports["mcpStore / mcpResourceStore"]
                RE38["mcpResources()"]
                RE39["mcpResourceAttachments()"]
                RE40["mcpHasResourceAttachments()"]
                RE41["mcpTotalResourceCount()"]
                RE42["mcpResourcesLoading()"]
            end
        end
    end

    subgraph Services["⚙️ Services"]
        direction TB
        subgraph SV1["ChatService"]
            SV1Msg["<b>Messaging:</b><br/>sendMessage()"]
            SV1Stream["<b>Streaming:</b><br/>handleStreamResponse()<br/>handleNonStreamResponse()"]
            SV1Convert["<b>Conversion:</b><br/>convertDbMessageToApiChatMessageData()<br/>mergeToolCallDeltas()"]
            SV1Utils["<b>Utilities:</b><br/>stripReasoningContent()<br/>extractModelName()<br/>parseErrorResponse()"]
        end
        subgraph SV2["ModelsService"]
            SV2List["<b>Listing:</b><br/>list()<br/>listRouter()"]
            SV2LoadUnload["<b>Load/Unload:</b><br/>load()<br/>unload()"]
            SV2Status["<b>Status:</b><br/>isModelLoaded()<br/>isModelLoading()"]
        end
        subgraph SV3["PropsService"]
            SV3Fetch["<b>Fetching:</b><br/>fetch()<br/>fetchForModel()"]
        end
        subgraph SV4["DatabaseService"]
            SV4Conv["<b>Conversations:</b><br/>createConversation()<br/>getConversation()<br/>getAllConversations()<br/>updateConversation()<br/>deleteConversation()"]
            SV4Msg["<b>Messages:</b><br/>createMessageBranch()<br/>createRootMessage()<br/>createSystemMessage()<br/>getConversationMessages()<br/>updateMessage()<br/>deleteMessage()<br/>deleteMessageCascading()"]
            SV4Node["<b>Navigation:</b><br/>updateCurrentNode()"]
            SV4Import["<b>Import:</b><br/>importConversations()"]
        end
        subgraph SV5["ParameterSyncService"]
            SV5Extract["<b>Extraction:</b><br/>extractServerDefaults()"]
            SV5Merge["<b>Merging:</b><br/>mergeWithServerDefaults()"]
            SV5Info["<b>Info:</b><br/>getParameterInfo()<br/>canSyncParameter()<br/>getSyncableParameterKeys()<br/>validateServerParameter()"]
            SV5Diff["<b>Diff:</b><br/>createParameterDiff()"]
        end
        subgraph SV6["MCPService"]
            SV6Transport["<b>Transport:</b><br/>createTransport()<br/>WebSocket / StreamableHTTP / SSE"]
            SV6Conn["<b>Connection:</b><br/>connect()<br/>disconnect()"]
            SV6Tools["<b>Tools:</b><br/>listTools()<br/>callTool()"]
            SV6Prompts["<b>Prompts:</b><br/>listPrompts()<br/>getPrompt()"]
            SV6Resources["<b>Resources:</b><br/>listResources()<br/>listResourceTemplates()<br/>readResource()<br/>subscribeResource()<br/>unsubscribeResource()"]
            SV6Complete["<b>Completions:</b><br/>complete()"]
        end
    end

    subgraph ExternalMCP["🔌 External MCP Servers"]
        EXT1["MCP Server 1<br/>(WebSocket/StreamableHTTP/SSE)"]
        EXT2["MCP Server N"]
    end

    subgraph Storage["💾 Storage"]
        ST1["IndexedDB"]
        ST2["conversations"]
        ST3["messages"]
        ST5["LocalStorage"]
        ST6["config"]
        ST7["userOverrides"]
        ST8["mcpServers"]
    end

    subgraph APIs["🌐 llama-server API"]
        API1["/v1/chat/completions"]
        API2["/props<br/>/props?model="]
        API3["/models<br/>/models/load<br/>/models/unload"]
        API4["/v1/models"]
    end

    %% Routes render Components
    R1 --> C_Screen
    R2 --> C_Screen
    RL --> C_Sidebar

    %% Layout runs MCP health checks on startup
    RL --> S6

    %% Component hierarchy
    C_Screen --> C_Form & C_Messages & C_Settings
    C_Messages --> C_Message
    C_Message --> C_MessageUser
    C_MessageUser --> C_MessageEditForm
    C_MessageEditForm --> C_ModelsSelector
    C_MessageEditForm --> C_Attach
    C_Form --> C_ModelsSelector
    C_Form --> C_Attach
    C_Form --> C_McpServersSelector
    C_Message --> C_Attach

    %% MCP Components hierarchy
    C_Settings --> C_McpSettings
    C_McpSettings --> C_McpServerCard
    C_McpServerCard --> C_McpResourceBrowser
    C_McpResourceBrowser --> C_McpResourcePreview

    %% Components use Hooks
    C_Form --> H1
    C_Message --> H1 & H2
    C_MessageEditForm --> H1
    C_Screen --> H2

    %% Hooks use Stores
    H1 --> S3 & S4
    H2 --> S1 & S5

    %% Components use Stores
    C_Screen --> S1 & S2
    C_Messages --> S2
    C_Message --> S1 & S2 & S3
    C_Form --> S1 & S3 & S6
    C_Sidebar --> S2
    C_ModelsSelector --> S3 & S4
    C_Settings --> S5
    C_McpSettings --> S6
    C_McpServerCard --> S6
    C_McpResourceBrowser --> S6 & S7
    C_McpServersSelector --> S6

    %% Stores export Reactive State
    S1 -. exports .-> ChatExports
    SA -. exports .-> AgenticExports
    S2 -. exports .-> ConvExports
    S3 -. exports .-> ModelsExports
    S4 -. exports .-> ServerExports
    S5 -. exports .-> SettingsExports
    S6 -. exports .-> MCPExports
    S7 -. exports .-> MCPExports

    %% chatStore → agenticStore (agentic loop orchestration)
    S1 --> SA
    SA --> SV1
    SA --> S6

    %% Stores use Services
    S1 --> SV1 & SV4
    S2 --> SV4
    S3 --> SV2 & SV3
    S4 --> SV3
    S5 --> SV5
    S6 --> SV6
    S7 --> SV6

    %% Services to Storage
    SV4 --> ST1
    ST1 --> ST2 & ST3
    SV5 --> ST5
    ST5 --> ST6 & ST7 & ST8

    %% Services to APIs
    SV1 --> API1
    SV2 --> API3 & API4
    SV3 --> API2

    %% MCP → External Servers
    SV6 --> EXT1 & EXT2

    %% Styling
    classDef routeStyle fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef componentStyle fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef componentGroupStyle fill:#e1bee7,stroke:#7b1fa2,stroke-width:1px
    classDef hookStyle fill:#fff8e1,stroke:#ff8f00,stroke-width:2px
    classDef storeStyle fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef stateStyle fill:#ffe0b2,stroke:#e65100,stroke-width:1px
    classDef methodStyle fill:#ffecb3,stroke:#e65100,stroke-width:1px
    classDef reactiveStyle fill:#fffde7,stroke:#f9a825,stroke-width:1px
    classDef serviceStyle fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px
    classDef serviceMStyle fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px
    classDef externalStyle fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px,stroke-dasharray: 5 5
    classDef storageStyle fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef apiStyle fill:#e3f2fd,stroke:#1565c0,stroke-width:2px

    class R1,R2,RL routeStyle
    class C_Sidebar,C_Screen,C_Form,C_Messages,C_Message,C_MessageUser,C_MessageEditForm componentStyle
    class C_ModelsSelector,C_Settings componentStyle
    class C_Attach componentStyle
    class C_McpSettings,C_McpServerCard,C_McpResourceBrowser,C_McpResourcePreview,C_McpServersSelector componentStyle
    class H1,H2,H3 hookStyle
    class LayoutComponents,ChatUIComponents,MCPComponents componentGroupStyle
    class Hooks hookStyle
    classDef agenticStyle fill:#e8eaf6,stroke:#283593,stroke-width:2px
    classDef agenticMethodStyle fill:#c5cae9,stroke:#283593,stroke-width:1px

    class S1,S2,S3,S4,S5,SA,S6,S7 storeStyle
    class S1State,S2State,S3State,S4State,S5State,SAState,S6State,S7State stateStyle
    class S1Msg,S1Regen,S1Edit,S1Stream,S1LoadState,S1ProcState,S1Error,S1Utils methodStyle
    class SASession,SAConfig,SAFlow methodStyle
    class S2Lifecycle,S2ConvCRUD,S2MsgMgmt,S2Nav,S2McpOverrides,S2Export,S2Utils methodStyle
    class S3Getters,S3Modal,S3Status,S3Fetch,S3Select,S3LoadUnload,S3Utils methodStyle
    class S4Getters,S4Data,S4Utils methodStyle
    class S5Lifecycle,S5Update,S5Reset,S5Sync,S5Utils methodStyle
    class S6Lifecycle,S6Health,S6Servers,S6Tools,S6Prompts methodStyle
    class S7Resources,S7Cache,S7Subs,S7Attach methodStyle
    class ChatExports,AgenticExports,ConvExports,ModelsExports,ServerExports,SettingsExports,MCPExports reactiveStyle
    class SV1,SV2,SV3,SV4,SV5,SV6 serviceStyle
    class SV6Transport,SV6Conn,SV6Tools,SV6Prompts,SV6Resources,SV6Complete serviceMStyle
    class EXT1,EXT2 externalStyle
    class SV1Msg,SV1Stream,SV1Convert,SV1Utils serviceMStyle
    class SV2List,SV2LoadUnload,SV2Status serviceMStyle
    class SV3Fetch serviceMStyle
    class SV4Conv,SV4Msg,SV4Node,SV4Import serviceMStyle
    class SV5Extract,SV5Merge,SV5Info,SV5Diff serviceMStyle
    class ST1,ST2,ST3,ST5,ST6,ST7,ST8 storageStyle
    class API1,API2,API3,API4 apiStyle
```
