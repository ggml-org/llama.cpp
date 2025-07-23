// See https://svelte.dev/docs/kit/types#app.d.ts
// for information about these interfaces
declare global {
	namespace App {
		// interface Error {}
		// interface Locals {}
		// interface PageData {}
		// interface PageState {}
		// interface Platform {}
	}

	export interface ChatMessageData {
		role: 'user' | 'assistant';
		content: string;
		timestamp?: Date;
	}

	export interface ChatCompletionRequest {
		messages: Array<{ role: string; content: string }>;
		stream?: boolean;
		temperature?: number;
		max_tokens?: number;
		top_p?: number;
		frequency_penalty?: number;
		presence_penalty?: number;
	}

	export interface ChatCompletionResponse {
		id: string;
		object: string;
		created: number;
		model: string;
		choices: Array<{
			index: number;
			message: {
				role: string;
				content: string;
			};
			finish_reason: string;
		}>;
		usage?: {
			prompt_tokens: number;
			completion_tokens: number;
			total_tokens: number;
		};
	}

	export interface ChatCompletionStreamChunk {
		id: string;
		object: string;
		created: number;
		model: string;
		choices: Array<{
			index: number;
			delta: {
				role?: string;
				content?: string;
			};
			finish_reason?: string;
		}>;
	}
}

export {};
