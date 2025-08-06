// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces

// Import chat types from dedicated module

import type {
	ApiChatCompletionRequest,
	ApiChatCompletionResponse,
	ApiChatCompletionStreamChunk,
	ApiChatMessageData,
	ApiChatMessageContentPart,
	ApiLlamaCppServerProps,
} from '$lib/types/api';

import type {
	ChatMessageType,
	ChatRole,
	ChatUploadedFile,
} from '$lib/types/chat';

import type {
	DatabaseAppSettings,
	DatabaseConversation,
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraAudioFile,
	DatabaseMessageExtraImageFile,
	DatabaseMessageExtraTextFile,
	DatabaseMessageExtraPdfFile,
} from '$lib/types/database';

import type {
	SettingsConfigValue,
	SettingsFieldConfig,
	SettingsConfigType,
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
		ApiLlamaCppServerProps,
		ChatMessageData,
		ChatMessageType,
		ChatRole,
		ChatUploadedFile,
		DatabaseAppSettings,
		DatabaseConversation,
		DatabaseMessage,
		DatabaseMessageExtra,
		DatabaseMessageExtraAudioFile,
		DatabaseMessageExtraImageFile,
		DatabaseMessageExtraTextFile,
		DatabaseMessageExtraPdfFile,
		SettingsConfigValue,
		SettingsFieldConfig,
		SettingsConfigType,
		SettingsChatServiceOptions,
	}
}
