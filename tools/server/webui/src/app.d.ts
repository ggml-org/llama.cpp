// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces

// Import chat types from dedicated module

import type {
	// API types
	ApiChatCompletionRequest,
	ApiChatCompletionResponse,
	ApiChatCompletionStreamChunk,
	ApiChatCompletionToolCall,
	ApiChatCompletionToolCallDelta,
	ApiChatMessageData,
	ApiChatMessageContentPart,
	ApiContextSizeError,
	ApiErrorResponse,
	ApiLlamaCppServerProps,
	ApiModelDataEntry,
	ApiModelListResponse,
	ApiProcessingState,
	ApiRouterModelMeta,
	ApiRouterModelsLoadRequest,
	ApiRouterModelsLoadResponse,
	ApiRouterModelsStatusRequest,
	ApiRouterModelsStatusResponse,
	ApiRouterModelsListResponse,
	ApiRouterModelsUnloadRequest,
	ApiRouterModelsUnloadResponse,
	// Chat types
	ChatAttachmentDisplayItem,
	ChatAttachmentPreviewItem,
	ChatMessageType,
	ChatRole,
	ChatUploadedFile,
	ChatMessageSiblingInfo,
	ChatMessagePromptProgress,
	ChatMessageTimings,
	// Database types
	DatabaseConversation,
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraAudioFile,
	DatabaseMessageExtraImageFile,
	DatabaseMessageExtraTextFile,
	DatabaseMessageExtraPdfFile,
	DatabaseMessageExtraLegacyContext,
	ExportedConversation,
	ExportedConversations,
	// Model types
	ModelModalities,
	ModelOption,
	// Settings types
	SettingsChatServiceOptions,
	SettingsConfigValue,
	SettingsFieldConfig,
	SettingsConfigType
} from '$lib/types';

import { ServerRole, ServerModelStatus, ModelModality } from '$lib/enums';

declare global {
	// namespace App {
	// interface Error {}
	// interface Locals {}
	// interface PageData {}
	// interface PageState {}
	// interface Platform {}
	// }

	export {
		// API types
		ApiChatCompletionRequest,
		ApiChatCompletionResponse,
		ApiChatCompletionStreamChunk,
		ApiChatCompletionToolCall,
		ApiChatCompletionToolCallDelta,
		ApiChatMessageData,
		ApiChatMessageContentPart,
		ApiContextSizeError,
		ApiErrorResponse,
		ApiLlamaCppServerProps,
		ApiModelDataEntry,
		ApiModelListResponse,
		ApiProcessingState,
		ApiRouterModelMeta,
		ApiRouterModelsLoadRequest,
		ApiRouterModelsLoadResponse,
		ApiRouterModelsStatusRequest,
		ApiRouterModelsStatusResponse,
		ApiRouterModelsListResponse,
		ApiRouterModelsUnloadRequest,
		ApiRouterModelsUnloadResponse,
		// Chat types
		ChatAttachmentDisplayItem,
		ChatAttachmentPreviewItem,
		ChatMessagePromptProgress,
		ChatMessageSiblingInfo,
		ChatMessageTimings,
		ChatMessageType,
		ChatRole,
		ChatUploadedFile,
		// Database types
		DatabaseConversation,
		DatabaseMessage,
		DatabaseMessageExtra,
		DatabaseMessageExtraAudioFile,
		DatabaseMessageExtraImageFile,
		DatabaseMessageExtraTextFile,
		DatabaseMessageExtraPdfFile,
		DatabaseMessageExtraLegacyContext,
		ExportedConversation,
		ExportedConversations,
		// Enum types
		ModelModality,
		ServerRole,
		ServerModelStatus,
		// Model types
		ModelModalities,
		ModelOption,
		// Settings types
		SettingsChatServiceOptions,
		SettingsConfigValue,
		SettingsFieldConfig,
		SettingsConfigType
	};
}

declare global {
	interface Window {
		idxThemeStyle?: number;
		idxCodeBlock?: number;
	}
}
