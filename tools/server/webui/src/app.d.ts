// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces

// Import chat types from dedicated module

import type {
	ApiChatCompletionRequest,
	ApiChatCompletionResponse,
	ApiChatCompletionStreamChunk,
	ApiChatMessageData,
	ApiChatMessageContentPart,
	ApiContextSizeError,
	ApiErrorResponse,
	ApiLlamaCppServerProps,
	ApiProcessingState
} from '$lib/types/api';

import type {
	ChatMessageType,
	ChatRole,
	ChatUploadedFile,
	ChatMessageSiblingInfo,
	ChatMessagePromptProgress,
	ChatMessageTimings
} from '$lib/types/chat';

import type {
	DatabaseConversation,
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraAudioFile,
	DatabaseMessageExtraImageFile,
	DatabaseMessageExtraTextFile,
	DatabaseMessageExtraPdfFile,
	DatabaseMessageExtraLegacyContext
} from '$lib/types/database';

import type {
	SettingsConfigValue,
	SettingsFieldConfig,
	SettingsConfigType
} from '$lib/types/settings';

declare global {
	// namespace App {
	// interface Error {}
	// interface Locals {}
	// interface PageData {}
	// interface PageState {}
	// interface Platform {}
	// }

	export {
		ApiChatCompletionRequest,
		ApiChatCompletionResponse,
		ApiChatCompletionStreamChunk,
		ApiChatMessageData,
		ApiChatMessageContentPart,
		ApiContextSizeError,
		ApiErrorResponse,
		ApiLlamaCppServerProps,
		ApiProcessingState,
		ChatMessageData,
		ChatMessagePromptProgress,
		ChatMessageSiblingInfo,
		ChatMessageTimings,
		ChatMessageType,
		ChatRole,
		ChatUploadedFile,
		DatabaseConversation,
		DatabaseMessage,
		DatabaseMessageExtra,
		DatabaseMessageExtraAudioFile,
		DatabaseMessageExtraImageFile,
		DatabaseMessageExtraTextFile,
		DatabaseMessageExtraPdfFile,
		DatabaseMessageExtraLegacyContext,
		SettingsConfigValue,
		SettingsFieldConfig,
		SettingsConfigType,
		SettingsChatServiceOptions
	};
}
