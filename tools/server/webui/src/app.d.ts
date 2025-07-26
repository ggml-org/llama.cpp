// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces

// Import chat types from dedicated module
import type {
	ChatMessageData,
	ChatCompletionRequest,
	ChatCompletionResponse,
	ChatCompletionStreamChunk
} from '$lib/types/chat';

declare global {
	// namespace App {
	// interface Error {}
	// interface Locals {}
	// interface PageData {}
	// interface PageState {}
	// interface Platform {}
	// }

	export type {
		ChatMessageData,
		ChatCompletionRequest,
		ChatCompletionResponse,
		ChatCompletionStreamChunk
	};
}
