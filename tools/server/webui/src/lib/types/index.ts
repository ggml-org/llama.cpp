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
	ApiChatCompletionRequestMessage,
	ApiToolDefinition,
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
	ChatMessageType,
	ChatRole,
	ChatUploadedFile,
	ChatAttachmentDisplayItem,
	ChatAttachmentPreviewItem,
	ChatMessageSiblingInfo,
	ChatMessagePromptProgress,
	ChatMessageTimings
} from './chat';

// Database types
export type {
	DatabaseConversation,
	DatabaseMessageExtraAudioFile,
	DatabaseMessageExtraImageFile,
	DatabaseMessageExtraLegacyContext,
	DatabaseMessageExtraPdfFile,
	DatabaseMessageExtraTextFile,
	DatabaseMessageExtra,
	DatabaseMessage,
	ExportedConversation,
	ExportedConversations
} from './database';

// Model types
export type { ModelModalities, ModelOption } from './models';

// Settings types
export type {
	SettingsConfigValue,
	SettingsFieldConfig,
	SettingsChatServiceOptions,
	SettingsConfigType
} from './settings';
