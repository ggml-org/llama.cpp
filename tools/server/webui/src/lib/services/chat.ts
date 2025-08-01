import type {
	ApiChatMessageContentPart,
	ApiChatMessageData,
	ApiChatCompletionRequest,
	ApiChatCompletionStreamChunk,
	ApiChatCompletionResponse,
	ApiLlamaCppServerProps
} from '$lib/types/api';

import type {
	DatabaseMessage,
	DatabaseMessageExtra,
	DatabaseMessageExtraImageFile,
	DatabaseMessageExtraTextFile
} from '$lib/types/database'

export class ChatService {
	private baseUrl: string;
	private abortController: AbortController | null = null;

	constructor(baseUrl = import.meta.env.VITE_BASE_URL) {
		this.baseUrl = baseUrl;
	}

	async sendMessage(
		messages: ApiChatMessageData[],
		options: {
			stream?: boolean;
			temperature?: number;
			max_tokens?: number;
			onChunk?: (chunk: string) => void;
			onComplete?: (response: string) => void;
			onError?: (error: Error) => void;
		} = {}
	): Promise<string | void> {
		const { stream, temperature, max_tokens, onChunk, onComplete, onError } = options;

		// Cancel any ongoing request
		this.abort();
		this.abortController = new AbortController();

		const requestBody: ApiChatCompletionRequest = {
			messages: messages.map((msg) => ({
				role: msg.role,
				content: msg.content
			})),
			stream,
			temperature,
			max_tokens
		};

		try {
			const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(requestBody),
				signal: this.abortController.signal
			});

			if (!response.ok) {
				throw new Error(`HTTP error! status: ${response.status}`);
			}

			if (stream) {
				return this.handleStreamResponse(response, onChunk, onComplete, onError);
			} else {
				return this.handleNonStreamResponse(response, onComplete, onError);
			}
		} catch (error) {
			if (error instanceof Error && error.name === 'AbortError') {
				return;
			}

			const err = error instanceof Error ? error : new Error('Unknown error');

			onError?.(err);

			throw err;
		}
	}

	private async handleStreamResponse(
		response: Response,
		onChunk?: (chunk: string) => void,
		onComplete?: (response: string) => void,
		onError?: (error: Error) => void
	): Promise<void> {
		const reader = response.body?.getReader();

		if (!reader) {
			throw new Error('No response body');
		}

		const decoder = new TextDecoder();
		let fullResponse = '';
		let thinkContent = '';
		let regularContent = '';
		let insideThinkTag = false;

		try {
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				const chunk = decoder.decode(value, { stream: true });
				const lines = chunk.split('\n');

				for (const line of lines) {
					if (line.startsWith('data: ')) {
						const data = line.slice(6);
						if (data === '[DONE]') {
							// Log final separated content
							if (thinkContent.trim()) {
								console.log('🤔 Think content:', thinkContent.trim());
							}
							if (regularContent.trim()) {
								console.log('💬 Regular response:', regularContent.trim());
							}
							onComplete?.(fullResponse);
							return;
						}

						try {
							const parsed: ApiChatCompletionStreamChunk = JSON.parse(data);
							const content = parsed.choices[0]?.delta?.content;
							if (content) {
								fullResponse += content;

								// Process content character by character to handle think tags
								insideThinkTag = this.processContentForThinkTags(
									content,
									insideThinkTag,
									(thinkChunk) => {
										thinkContent += thinkChunk;
									},
									(regularChunk) => {
										regularContent += regularChunk;
									}
								);

								onChunk?.(content);
							}
						} catch (e) {
							console.error('Error parsing JSON chunk:', e);
						}
					}
				}
			}
		} catch (error) {
			const err = error instanceof Error ? error : new Error('Stream error');

			onError?.(err);

			throw err;
		} finally {
			reader.releaseLock();
		}
	}

	private async handleNonStreamResponse(
		response: Response,
		onComplete?: (response: string) => void,
		onError?: (error: Error) => void
	): Promise<string> {
		try {
			const data: ApiChatCompletionResponse = await response.json();
			const content = data.choices[0]?.message?.content || '';

			onComplete?.(content);

			return content;
		} catch (error) {
			const err = error instanceof Error ? error : new Error('Parse error');

			onError?.(err);

			throw err;
		}
	}

	/**
	 * Convert Message with extras to ApiChatMessageData format for API requests
	 */
	private static convertMessageToChatData(message: DatabaseMessage & { extra?: DatabaseMessageExtra[] }): ApiChatMessageData {
		// If no extras, return simple text message
		if (!message.extra || message.extra.length === 0) {
			return {
				role: message.role as 'user' | 'assistant' | 'system',
				content: message.content
			};
		}

		// Build multimodal content array
		const contentParts: ApiChatMessageContentPart[] = [];

		// Add text content first
		if (message.content) {
			contentParts.push({
				type: 'text',
				text: message.content
			});
		}

		// Add image files
		const imageFiles = message.extra.filter((extra): extra is DatabaseMessageExtraImageFile => extra.type === 'imageFile');
		imageFiles.forEach((image) => {
			contentParts.push({
				type: 'image_url',
				image_url: { url: image.base64Url }
			});
		});

		// Add text files as additional text content
		const textFiles = message.extra.filter((extra): extra is DatabaseMessageExtraTextFile => extra.type === 'textFile');
		textFiles.forEach((textFile) => {
			contentParts.push({
				type: 'text',
				text: `\n\n--- File: ${textFile.name} ---\n${textFile.content}`
			});
		});

		return {
			role: message.role as 'user' | 'assistant' | 'system',
			content: contentParts
		};
	}

	/**
	 * Unified method to send chat completions supporting both ApiChatMessageData and Message types
	 */
	async sendChatCompletion(
		messages: (ApiChatMessageData[] | DatabaseMessage[]) | (DatabaseMessage & { extra?: DatabaseMessageExtra[] })[],
		options: {
			stream?: boolean;
			temperature?: number;
			max_tokens?: number;
			onChunk?: (chunk: string) => void;
			onComplete?: (response?: string) => void;
			onError?: (error: Error) => void;
		} = {}
	): Promise<string | void> {
		// Handle both array formats and convert messages with extras
		const normalizedMessages: ApiChatMessageData[] = messages.map((msg) => {
			// Check if this is already a ApiChatMessageData object
			if ('content' in msg && (typeof msg.content === 'string' || Array.isArray(msg.content))) {
				return msg as ApiChatMessageData;
			}
			
			// Convert DatabaseMessage with extras to ApiChatMessageData
			return ChatService.convertMessageToChatData(msg as DatabaseMessage & { extra?: DatabaseMessageExtra[] });
		});

		// Set default options for API compatibility
		const finalOptions = {
			stream: true,
			temperature: 0.7,
			max_tokens: 2048,
			...options
		};

		return this.sendMessage(normalizedMessages, finalOptions);
	}

	/**
	 * Static method for backward compatibility with ApiService
	 */
	static async sendChatCompletion(
		messages: DatabaseMessage[],
		onChunk?: (content: string) => void,
		onComplete?: () => void,
		onError?: (error: Error) => void
	): Promise<string> {
		const service = new ChatService();
		const result = await service.sendChatCompletion(messages, {
			stream: true,
			temperature: 0.7,
			max_tokens: 2048,
			onChunk,
			onComplete: () => onComplete?.(),
			onError
		});
		return result as string;
	}

	/**
	 * Get server properties - static method for API compatibility
	 */
	static async getServerProps(): Promise<ApiLlamaCppServerProps> {
		try {
			const response = await fetch(`${import.meta.env.VITE_BASE_URL}/props`, {
				headers: {
					'Content-Type': 'application/json'
				}
			});

			if (!response.ok) {
				throw new Error(`Failed to fetch server props: ${response.status}`);
			}

			const data = await response.json();
			return data;
		} catch (error) {
			console.error('Error fetching server props:', error);
			throw error;
		}
	}

	private processContentForThinkTags(
		content: string,
		currentInsideThinkTag: boolean,
		addThinkContent: (chunk: string) => void,
		addRegularContent: (chunk: string) => void
	): boolean {
		let i = 0;
		let insideThinkTag = currentInsideThinkTag;

		while (i < content.length) {
			// Check for opening <think> tag
			if (!insideThinkTag && content.substring(i, i + 7) === '<think>') {
				insideThinkTag = true;
				i += 7; // Skip the <think> tag
				continue;
			}

			// Check for closing </think> tag
			if (insideThinkTag && content.substring(i, i + 8) === '</think>') {
				insideThinkTag = false;
				i += 8; // Skip the </think> tag
				continue;
			}

			// Add character to appropriate content bucket
			if (insideThinkTag) {
				addThinkContent(content[i]);
			} else {
				addRegularContent(content[i]);
			}

			i++;
		}

		return insideThinkTag;
	}

	abort(): void {
		if (this.abortController) {
			this.abortController.abort();
			this.abortController = null;
		}
	}
}
