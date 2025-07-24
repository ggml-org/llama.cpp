import type { ChatMessage } from '$lib/stores/database';

export interface LlamaCppServerProps {
	build_info: string;
	model_path: string;
	n_ctx: number;
	modalities?: {
		vision: boolean;
		audio: boolean;
	};
}

export interface ChatCompletionMessage {
	role: 'user' | 'assistant' | 'system';
	content: string;
}

export interface ChatCompletionRequest {
	model?: string;
	messages: ChatCompletionMessage[];
	stream?: boolean;
	temperature?: number;
	max_tokens?: number;
	top_p?: number;
	frequency_penalty?: number;
	presence_penalty?: number;
}

export interface ChatCompletionChunk {
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
		finish_reason: string | null;
	}>;
}

export class ApiService {
	private static baseUrl = import.meta.env.VITE_BASE_URL;

	/**
	 * Send a chat completion request with streaming support
	 */
	static async sendChatCompletion(
		messages: ChatMessage[],
		onChunk?: (content: string) => void,
		onComplete?: () => void,
		onError?: (error: Error) => void
	): Promise<string> {
		try {
			// Convert database messages to API format
			const apiMessages: ChatCompletionMessage[] = messages.map((msg) => ({
				role: msg.role as 'user' | 'assistant' | 'system',
				content: msg.content
			}));

			const request: ChatCompletionRequest = {
				messages: apiMessages,
				stream: true,
				temperature: 0.7,
				max_tokens: 2048
			};

			console.log('Sending chat completion request:', request);

			const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					Accept: 'text/event-stream'
				},
				body: JSON.stringify(request)
			});

			if (!response.ok) {
				throw new Error(`HTTP ${response.status}: ${response.statusText}`);
			}

			if (!response.body) {
				throw new Error('No response body received');
			}

			// Handle streaming response
			const reader = response.body.getReader();
			const decoder = new TextDecoder();
			let fullResponse = '';

			try {
				while (true) {
					const { done, value } = await reader.read();

					if (done) break;

					const chunk = decoder.decode(value, { stream: true });
					const lines = chunk.split('\n').filter((line) => line.trim());

					for (const line of lines) {
						if (line.startsWith('data: ')) {
							const data = line.slice(6).trim();

							if (data === '[DONE]') {
								onComplete?.();
								return fullResponse;
							}

							try {
								const parsed: ChatCompletionChunk = JSON.parse(data);
								const content = parsed.choices[0]?.delta?.content;

								if (content) {
									fullResponse += content;
									onChunk?.(content);
								}
							} catch (parseError) {
								console.warn('Failed to parse SSE data:', data, parseError);
							}
						}
					}
				}
			} finally {
				reader.releaseLock();
			}

			onComplete?.();
			return fullResponse;
		} catch (error) {
			console.error('Chat completion error:', error);
			onError?.(error as Error);
			throw error;
		}
	}

	/**
	 * Get server properties including model info, build info, and supported modalities
	 */
	static async getServerProps(): Promise<LlamaCppServerProps> {
		try {
			const response = await fetch(`${this.baseUrl}/props`, {
				headers: {
					'Content-Type': 'application/json'
				}
			});

			if (!response.ok) {
				throw new Error(`Failed to fetch server props: ${response.status}`);
			}

			const data = await response.json();
			return data as LlamaCppServerProps;
		} catch (error) {
			console.error('Error fetching server props:', error);
			throw error;
		}
	}
}
