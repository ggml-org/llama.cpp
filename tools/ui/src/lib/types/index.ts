/**
 * Unified exports for all type definitions
 * Import types from '$lib/types' for cleaner imports
 */

// API types
export type {
	ApiChatMessageContentPart,
	ApiContextSizeError,
	ApiErrorResponse,
	ApiChatMessageData,
	ApiModelStatus,
	ApiModelDataEntry,
	ApiModelLoadStage,
	ApiModelsSseProgress,
	ApiModelsSseData,
	ApiModelsSseEvent,
	ApiModelDetails,
	ApiModelListResponse,
	ApiLlamaCppServerProps,
	ApiChatCompletionRequest,
	ApiChatCompletionToolCallFunctionDelta,
	ApiChatCompletionToolCallDelta,
	ApiChatCompletionToolCall,
	ApiChatCompletionStreamChunk,
	ApiChatCompletionResponse,
	ApiSlotData,
	ApiProcessingState,
	ApiRouterModelMeta,
	ApiRouterModelsLoadRequest,
	ApiRouterModelsLoadResponse,
	ApiRouterModelsStatusRequest,
	ApiRouterModelsStatusResponse,
	ApiRouterModelsListResponse,
	ApiRouterModelsUnloadRequest,
	ApiRouterModelsUnloadResponse,
	AudioInputFormat,
	ApiStreamSession
} from './api';

// Chat types
export type {
	AttachmentMenuItem,
	ChatUploadedFile,
	ChatAttachmentDisplayItem,
	ChatMessageSiblingInfo,
	ChatMessageActions,
	ChatMessageActionsContext,
	ChatMessageDeletionInfo,
	ChatMessageEditContext,
	ChatMessageEditState,
	ChatMessageEditActions,
	ChatMessageAssistantEditActions,
	ChatFormActionsContext,
	ChatMessagePromptProgress,
	ChatMessageTimings,
	ChatMessageAgenticTimings,
	ChatMessageAgenticTurnStats,
	ChatMessageToolCallTiming,
	ChatStreamCallbacks,
	ErrorDialogState,
	LiveProcessingStats,
	LiveGenerationStats,
	AttachmentDisplayItemsOptions,
	FileProcessingResult,
	FileMentionEntry,
	ChatFormCommand,
	ChatCommandsOptions,
	ControlAction
} from './chat.d';

// Database types
export type {
	McpServerOverride,
	DatabaseConversation,
	DatabaseMessageExtraAudioFile,
	DatabaseMessageExtraVideoFile,
	DatabaseMessageExtraImageFile,
	DatabaseMessageExtraLegacyContext,
	DatabaseMessageExtraMcpPrompt,
	DatabaseMessageExtraMcpResource,
	DatabaseMessageExtraPdfFile,
	DatabaseMessageExtraTextFile,
	DatabaseMessageExtra,
	DatabaseMessage,
	ExportedConversation,
	ExportedConversations
} from './database';

// Model types
export type {
	ModelCapabilities,
	ModelModalities,
	ModelOption,
	ModelLoadProgress,
	ModalityCapabilities
} from './models';

// Settings types
export type {
	SettingsConfigValue,
	SettingsFieldConfig,
	SettingsChatServiceOptions,
	SettingsConfigType,
	SettingsExportType,
	ParameterValue,
	ParameterRecord,
	ParameterInfo,
	SyncableParameter,
	SettingsEntry,
	SettingsSectionTitle,
	SettingsSectionEntry,
	SettingsSection
} from './settings';

// Common types
export type {
	KeyValuePair,
	BinaryDetectionOptions,
	ClipboardTextAttachment,
	ClipboardMcpPromptAttachment,
	ClipboardAttachment,
	ParsedClipboardContent
} from './common';

// MCP types
export type {
	ClientCapabilities,
	ServerCapabilities,
	Implementation,
	MCPConnectionLog,
	MCPServerInfo,
	MCPCapabilitiesInfo,
	MCPToolInfo,
	MCPPromptInfo,
	MCPConnectionDetails,
	MCPPhaseCallback,
	MCPConnection,
	HealthCheckState,
	HealthCheckParams,
	MCPServerConfig,
	MCPClientConfig,
	MCPServerSettingsEntry,
	MCPServerDisplayInfo,
	RecommendedMCPServer,
	MCPToolCall,
	OpenAIToolDefinition,
	ServerStatus,
	ToolCallParams,
	ToolExecutionResult,
	ServerToolInfo,
	Tool,
	Prompt,
	GetPromptResult,
	PromptMessage,
	MCPProgressState,
	MCPResourceAnnotations,
	MCPResourceIcon,
	MCPResource,
	MCPResourceTemplate,
	MCPTextResourceContent,
	MCPBlobResourceContent,
	MCPResourceContent,
	MCPReadResourceResult,
	MCPResourceInfo,
	MCPResourceTemplateInfo,
	MCPCachedResource,
	MCPResourceAttachment,
	MCPResourceSubscription,
	MCPServerResources
} from './mcp';

// Search result types
export type { SearchResult } from './search';

// Glob search types (working-directory / mention pickers)
export type {
	GlobEntry,
	GlobSearchArgs,
	GlobSearchResult,
	GlobEntryResult,
	GlobSearchChildOptions,
	GlobSearchChildResult
} from './glob';

// ChatFormInputRich token types (chat form)
export type { ChatFormInputRichToken } from './chat-form-input-rich';

// Agentic types
export type {
	AgenticConfig,
	AgenticToolCallPayload,
	AgenticMessage,
	AgenticAssistantMessage,
	AgenticToolCallList,
	AgenticChatCompletionRequest,
	AgenticSession,
	AgenticFlowCallbacks,
	AgenticFlowOptions,
	AgenticFlowParams,
	AgenticFlowResult,
	SteeringMessage,
	AgenticSection,
	ToolResultLine,
	ContinueIntent
} from './agentic';

// Navigation types
export type { DesktopIconStripItem } from './navigation';

// Tools types
export type { ToolEntry, ToolGroup, ToolUiEntry } from './tools';

// Reasoning
export type { ReasoningEffortLevel } from './reasoning';

// Splash
export type { SplashDimensions } from './splash';
