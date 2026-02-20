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
	ChatStreamCallbacks,
	ErrorDialogState,
	LiveProcessingStats,
	LiveGenerationStats,
	AttachmentDisplayItemsOptions,
	FileProcessingResult
} from './chat.d';

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
	ParsedClipboardContent
} from './common';
