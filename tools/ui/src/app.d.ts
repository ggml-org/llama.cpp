// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces

import 'vite-plugin-pwa/pwa-assets';
import 'vite-plugin-pwa/svelte';
import { ModelModality, ServerModelStatus, ServerRole } from '$lib/enums';
// Import chat types from dedicated module
import type {
	// API types
	ApiChatCompletionRequest,
	ApiChatCompletionResponse,
	ApiChatCompletionStreamChunk,
	ApiChatCompletionToolCall,
	ApiChatCompletionToolCallDelta,
	ApiChatMessageContentPart,
	ApiChatMessageData,
	ApiContextSizeError,
	ApiErrorResponse,
	ApiLlamaCppServerProps,
	ApiModelDataEntry,
	ApiModelListResponse,
	ApiModelLoadStage,
	ApiModelsSseData,
	ApiModelsSseEvent,
	ApiModelsSseProgress,
	ApiProcessingState,
	ApiRouterModelMeta,
	ApiRouterModelsListResponse,
	ApiRouterModelsLoadRequest,
	ApiRouterModelsLoadResponse,
	ApiRouterModelsStatusRequest,
	ApiRouterModelsStatusResponse,
	ApiRouterModelsUnloadRequest,
	ApiRouterModelsUnloadResponse,
	ChatAttachmentDisplayItem,
	// Chat types
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
	DatabaseMessageExtraLegacyContext,
	DatabaseMessageExtraPdfFile,
	DatabaseMessageExtraTextFile,
	DatabaseMessageExtraVideoFile,
	ExportedConversation,
	ExportedConversations,
	ModelLoadProgress,
	// Model types
	ModelModalities,
	ModelOption,
	// Settings types
	SettingsChatServiceOptions,
	SettingsConfigType,
	SettingsConfigValue,
	SettingsFieldConfig
} from '$lib/types';

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
		ApiModelLoadStage,
		ApiModelsSseProgress,
		ApiModelsSseData,
		ApiModelsSseEvent,
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
		DatabaseMessageExtraVideoFile,
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
		ModelLoadProgress,
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

		// File System Access API - not in the DOM lib and unavailable in some browsers
		showDirectoryPicker?: (options?: {
			id?: string;
			mode?: 'read' | 'readwrite';
			startIn?: FileSystemHandle | string;
		}) => Promise<FileSystemDirectoryHandle>;
	}
}
