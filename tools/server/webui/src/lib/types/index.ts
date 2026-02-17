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
	ApiRouterModelsUnloadResponse
} from './api';

// Chat types
export type {
	ChatUploadedFile,
	ChatAttachmentDisplayItem,
	ChatAttachmentPreviewItem,
	ChatMessageSiblingInfo,
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
	FileProcessingResult
} from './chat.d';

// Database types
export type {
	McpServerOverride,
	DatabaseConversation,
	DatabaseMessageExtraAudioFile,
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
export type { ModelModalities, ModelOption, ModalityCapabilities } from './models';

// Settings types
export type {
	SettingsConfigValue,
	SettingsFieldConfig,
	SettingsChatServiceOptions,
	SettingsConfigType,
	ParameterValue,
	ParameterRecord,
	ParameterInfo,
	SyncableParameter
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
	MCPToolCall,
	OpenAIToolDefinition,
	ServerStatus,
	ToolCallParams,
	ToolExecutionResult,
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
	AgenticFlowResult
} from './agentic';
