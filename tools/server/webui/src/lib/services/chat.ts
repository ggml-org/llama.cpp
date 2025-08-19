import { config } from '$lib/stores/settings.svelte';

/**
 * Service class for handling chat completions with the llama.cpp server.
 * Provides methods for sending messages, handling streaming responses, and managing chat sessions.
 */
export class ChatService {
	private abortController: AbortController | null = null;

	/**
	 * Sends a chat completion request to the llama.cpp server.
	 * Supports both streaming and non-streaming responses with comprehensive parameter configuration.
	 * 
	 * @param messages - Array of chat messages to send to the API
	 * @param options - Configuration options for the chat completion request. See `SettingsChatServiceOptions` type for details.
	 * @returns {Promise<string | void>} that resolves to the complete response string (non-streaming) or void (streaming)
	 * @throws {Error} if the request fails or is aborted
	 */
	async sendMessage(
		messages: ApiChatMessageData[],
		options: SettingsChatServiceOptions = {}
		): Promise<string | void> {
		const { 
			stream, onChunk, onComplete, onError,
			// Generation parameters
			temperature, max_tokens,
			// Sampling parameters
			dynatemp_range, dynatemp_exponent, top_k, top_p, min_p,
			xtc_probability, xtc_threshold, typical_p,
			// Penalty parameters
			repeat_last_n, repeat_penalty, presence_penalty, frequency_penalty,
			dry_multiplier, dry_base, dry_allowed_length, dry_penalty_last_n,
			// Other parameters
			samplers, custom
		} = options;

		// Cancel any ongoing request and create a new abort controller
		this.abort();
		this.abortController = new AbortController();

		// Build base request body with system message injection
		const processedMessages = this.injectSystemMessage(messages);
		const requestBody: ApiChatCompletionRequest = {
			messages: processedMessages.map((msg: ApiChatMessageData) => ({
				role: msg.role,
				content: msg.content
			})),
			reasoning_format: 'auto',
			stream
		};

		// Add generation parameters if provided
		if (temperature !== undefined) requestBody.temperature = temperature;
		if (max_tokens !== undefined) requestBody.max_tokens = max_tokens;

		// Add sampling parameters if provided
		if (dynatemp_range !== undefined) requestBody.dynatemp_range = dynatemp_range;
		if (dynatemp_exponent !== undefined) requestBody.dynatemp_exponent = dynatemp_exponent;
		if (top_k !== undefined) requestBody.top_k = top_k;
		if (top_p !== undefined) requestBody.top_p = top_p;
		if (min_p !== undefined) requestBody.min_p = min_p;
		if (xtc_probability !== undefined) requestBody.xtc_probability = xtc_probability;
		if (xtc_threshold !== undefined) requestBody.xtc_threshold = xtc_threshold;
		if (typical_p !== undefined) requestBody.typical_p = typical_p;

		// Add penalty parameters if provided
		if (repeat_last_n !== undefined) requestBody.repeat_last_n = repeat_last_n;
		if (repeat_penalty !== undefined) requestBody.repeat_penalty = repeat_penalty;
		if (presence_penalty !== undefined) requestBody.presence_penalty = presence_penalty;
		if (frequency_penalty !== undefined) requestBody.frequency_penalty = frequency_penalty;
		if (dry_multiplier !== undefined) requestBody.dry_multiplier = dry_multiplier;
		if (dry_base !== undefined) requestBody.dry_base = dry_base;
		if (dry_allowed_length !== undefined) requestBody.dry_allowed_length = dry_allowed_length;
		if (dry_penalty_last_n !== undefined) requestBody.dry_penalty_last_n = dry_penalty_last_n;

		// Add sampler configuration if provided
		if (samplers !== undefined) {
			requestBody.samplers = typeof samplers === 'string' ? samplers.split(';').filter((s: string) => s.trim()) : samplers;
		}

		// Add custom parameters if provided
		if (custom) {
			try {
				const customParams = typeof custom === 'string' ? JSON.parse(custom) : custom;
				Object.assign(requestBody, customParams);
			} catch (error) {
				console.warn('Failed to parse custom parameters:', error);
			}
		}

		try {
			const response = await fetch(`/v1/chat/completions`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json'
				},
				body: JSON.stringify(requestBody),
				signal: this.abortController.signal
			});

			if (!response.ok) {
				let errorMessage = `Server error (${response.status})`;
				
				switch (response.status) {
					case 400:
						errorMessage = 'Invalid request - check your message format';
						break;
					case 401:
						errorMessage = 'Unauthorized - check server authentication';
						break;
					case 404:
						errorMessage = 'Chat endpoint not found - server may not support chat completions';
						break;
					case 500:
						errorMessage = 'Server internal error - check server logs';
						break;
					case 503:
						errorMessage = 'Server unavailable - try again later';
						break;
					default:
						errorMessage = `Server error (${response.status}): ${response.statusText}`;
				}
				
				throw new Error(errorMessage);
			}

			if (stream) {
				return this.handleStreamResponse(response, onChunk, onComplete, onError, options.onReasoningChunk);
			} else {
				return this.handleNonStreamResponse(response, onComplete, onError);
			}
		} catch (error) {
			if (error instanceof Error && error.name === 'AbortError') {
				console.log('Chat completion request was aborted');
				return;
			}

			// Handle network errors with user-friendly messages
			let friendlyError: Error;
			if (error instanceof Error) {
				if (error.name === 'TypeError' && error.message.includes('fetch')) {
					friendlyError = new Error('Unable to connect to server - please check if the server is running');
				} else if (error.message.includes('ECONNREFUSED')) {
					friendlyError = new Error('Connection refused - server may be offline');
				} else if (error.message.includes('ETIMEDOUT')) {
					friendlyError = new Error('Request timeout - server may be overloaded');
				} else {
					friendlyError = error;
				}
			} else {
				friendlyError = new Error('Unknown error occurred while sending message');
			}

			console.error('Error in sendMessage:', error);
			if (onError) {
				onError(friendlyError);
			}
			throw friendlyError;
		}
	}

	/**
	 * Handles streaming response from the chat completion API.
	 * Processes server-sent events and extracts content chunks from the stream.
	 * 
	 * @param response - The fetch Response object containing the streaming data
	 * @param onChunk - Optional callback invoked for each content chunk received
	 * @param onComplete - Optional callback invoked when the stream is complete with full response
	 * @param onError - Optional callback invoked if an error occurs during streaming
	 * @param onReasoningChunk - Optional callback invoked for each reasoning content chunk
	 * @returns {Promise<void>} Promise that resolves when streaming is complete
	 * @throws {Error} if the stream cannot be read or parsed
	 */
	private async handleStreamResponse(
		response: Response,
		onChunk?: (chunk: string) => void,
		onComplete?: (response: string, reasoningContent?: string) => void,
		onError?: (error: Error) => void,
		onReasoningChunk?: (chunk: string) => void
	): Promise<void> {
		const reader = response.body?.getReader();

		if (!reader) {
			throw new Error('No response body');
		}

		const decoder = new TextDecoder();
		let fullResponse = '';
		let fullReasoningContent = '';
		let thinkContent = '';
		let regularContent = '';
		let insideThinkTag = false;
		let hasReceivedData = false;

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
							// Check if we received any actual content
							if (!hasReceivedData && fullResponse.length === 0) {
								// Empty response - likely a context error
								const contextError = new Error('The request exceeds the available context size. Try increasing the context size or enable context shift.');
								contextError.name = 'ContextError';
								onError?.(contextError);
								return;
							}
							onComplete?.(regularContent, fullReasoningContent || undefined);
							return;
						}

						try {
							const parsed: ApiChatCompletionStreamChunk = JSON.parse(data);
							const content = parsed.choices[0]?.delta?.content;
							const reasoningContent = parsed.choices[0]?.delta?.reasoning_content;

							if (content) {
								hasReceivedData = true;
								fullResponse += content;

								// Track the regular content before processing this chunk
								const regularContentBefore = regularContent;

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

								// Only send the new regular content that was added in this chunk
								const newRegularContent = regularContent.slice(regularContentBefore.length);
								if (newRegularContent) {
									onChunk?.(newRegularContent);
								}
							}

							if (reasoningContent) {
								hasReceivedData = true;
								fullReasoningContent += reasoningContent;
								onReasoningChunk?.(reasoningContent);
							}
						} catch (e) {
							console.error('Error parsing JSON chunk:', e);
						}
					}
				}
			}

			// If we reach here without receiving [DONE] and no data, it's likely a context error
			if (!hasReceivedData && fullResponse.length === 0) {
				const contextError = new Error('The request exceeds the available context size. Try increasing the context size or enable context shift.');
				contextError.name = 'ContextError';
				onError?.(contextError);
				return;
			}
		} catch (error) {
			const err = error instanceof Error ? error : new Error('Stream error');

			onError?.(err);

			throw err;
		} finally {
			reader.releaseLock();
		}
	}

	/**
	 * Handles non-streaming response from the chat completion API.
	 * Parses the JSON response and extracts the generated content.
	 * 
	 * @param response - The fetch Response object containing the JSON data
	 * @param onComplete - Optional callback invoked when response is successfully parsed
	 * @param onError - Optional callback invoked if an error occurs during parsing
		 * @returns {Promise<string>} Promise that resolves to the generated content string
	 * @throws {Error} if the response cannot be parsed or is malformed
	 */
	private async handleNonStreamResponse(
		response: Response,
		onComplete?: (response: string, reasoningContent?: string) => void,
		onError?: (error: Error) => void
		): Promise<string> {
		try {
			// Check if response body is empty
			const responseText = await response.text();
			if (!responseText.trim()) {
				// Empty response - likely a context error
				const contextError = new Error('The request exceeds the available context size. Try increasing the context size or enable context shift.');
				contextError.name = 'ContextError';
				onError?.(contextError);
				throw contextError;
			}

			const data: ApiChatCompletionResponse = JSON.parse(responseText);
			const content = data.choices[0]?.message?.content || '';
			const reasoningContent = data.choices[0]?.message?.reasoning_content;

			if (reasoningContent) {
				console.log('Full reasoning content:', reasoningContent);
			}

			// Check if content is empty even with valid JSON structure
			if (!content.trim()) {
				const contextError = new Error('The request exceeds the available context size. Try increasing the context size or enable context shift.');
				contextError.name = 'ContextError';
				onError?.(contextError);
				throw contextError;
			}

			onComplete?.(content, reasoningContent);

			return content;
		} catch (error) {
			// If it's already a ContextError, re-throw it
			if (error instanceof Error && error.name === 'ContextError') {
				throw error;
			}

			const err = error instanceof Error ? error : new Error('Parse error');

			onError?.(err);

			throw err;
		}
	}

	/**
	 * Converts a database message with attachments to API chat message format.
	 * Processes various attachment types (images, text files, PDFs) and formats them
	 * as content parts suitable for the chat completion API.
	 * 
	 * @param message - Database message object with optional extra attachments
	 * @param message.content - The text content of the message
	 * @param message.role - The role of the message sender (user, assistant, system)
	 * @param message.extra - Optional array of message attachments (images, files, etc.)
	 * @returns {ApiChatMessageData} object formatted for the chat completion API
	 * @static
	 */
	static convertMessageToChatServiceData(message: DatabaseMessage & { extra?: DatabaseMessageExtra[] }): ApiChatMessageData {
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
		const imageFiles = message.extra.filter((extra: DatabaseMessageExtra): extra is DatabaseMessageExtraImageFile => extra.type === 'imageFile');

		for (const image of imageFiles) {
			contentParts.push({
				type: 'image_url',
				image_url: { url: image.base64Url }
			});
		}

		// Add text files as additional text content
		const textFiles = message.extra.filter((extra: DatabaseMessageExtra): extra is DatabaseMessageExtraTextFile => extra.type === 'textFile');

		for (const textFile of textFiles) {
			contentParts.push({
				type: 'text',
				text: `\n\n--- File: ${textFile.name} ---\n${textFile.content}`
			});
		}

		// Add audio files
		const audioFiles = message.extra.filter((extra: DatabaseMessageExtra): extra is DatabaseMessageExtraAudioFile => extra.type === 'audioFile');

		for (const audio of audioFiles) {
			contentParts.push({
				type: 'input_audio',
				input_audio: {
					data: audio.base64Data,
					format: audio.mimeType.includes('wav') ? 'wav' : 'mp3'
				}
			});
		}

		// Add PDF files as text content
		const pdfFiles = message.extra.filter((extra: DatabaseMessageExtra): extra is DatabaseMessageExtraPdfFile => extra.type === 'pdfFile');

		for (const pdfFile of pdfFiles) {
			if (pdfFile.processedAsImages && pdfFile.images) {
				// If PDF was processed as images, add each page as an image
				for (let i = 0; i < pdfFile.images.length; i++) {
					contentParts.push({
						type: 'image_url',
						image_url: { url: pdfFile.images[i] }
					});
				}
			} else {
				// If PDF was processed as text, add as text content
				contentParts.push({
					type: 'text',
					text: `\n\n--- PDF File: ${pdfFile.name} ---\n${pdfFile.content}`
				});
			}
		}

		return {
			role: message.role as 'user' | 'assistant' | 'system',
			content: contentParts
		};
	}

	/**
	 * Unified method to send chat completions supporting both ApiChatMessageData and DatabaseMessage types.
	 * Automatically converts database messages with attachments to the appropriate API format.
	 * 
	 * @param messages - Array of messages in either API format or database format with attachments
	 * @param options - Configuration options for the chat completion
	 * @param options.stream - Whether to use streaming response (default: true)
	 * @param options.temperature - Controls randomness in generation (default: 0.7)
	 * @param options.max_tokens - Maximum number of tokens to generate (default: 2048)
	 * @param options.onChunk - Callback for streaming response chunks
	 * @param options.onComplete - Callback when response is complete
	 * @param options.onError - Callback for error handling
	 * @returns Promise that resolves to the complete response string or void for streaming
	 */
	async sendChatCompletion(
		messages: (ApiChatMessageData[] | DatabaseMessage[]) | (DatabaseMessage & { extra?: DatabaseMessageExtra[] })[],
		options: {
			stream?: boolean;
			temperature?: number;
			max_tokens?: number;
			onChunk?: (chunk: string) => void;
			onReasoningChunk?: (chunk: string) => void;
			onComplete?: (response?: string, reasoningContent?: string) => void;
			onError?: (error: Error) => void;
		} = {}
	): Promise<string | void> {
		// Handle both array formats and convert messages with extras
		const normalizedMessages: ApiChatMessageData[] = messages.map((msg) => {
			// Check if this is already a ApiChatMessageData object by checking for DatabaseMessage-specific fields
			if ('id' in msg && 'convId' in msg && 'timestamp' in msg) {
				// This is a DatabaseMessage, convert it
				const dbMsg = msg as DatabaseMessage & { extra?: DatabaseMessageExtra[] };
				const converted = ChatService.convertMessageToChatServiceData(dbMsg);
				return converted;
			} else {
				// This is already an ApiChatMessageData object
				return msg as ApiChatMessageData;
			}
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
	 * Static method for backward compatibility with the legacy ApiService.
	 * Creates a temporary ChatService instance and sends a chat completion request.
	 * 
	 * @param messages - Array of database messages to send
	 * @param onChunk - Optional callback for streaming response chunks
	 * @param onComplete - Optional callback when response is complete
	 * @param onError - Optional callback for error handling
	 * @returns Promise that resolves to the complete response string
	 * @static
	 * @deprecated Use ChatService instance methods instead
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
			const response = await fetch(`/props`, {
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

	/**
	 * Processes content to separate thinking tags from regular content.
	 * Parses <think> and </think> tags to route content to appropriate handlers.
	 * 
	 * @param content - The content string to process
	 * @param currentInsideThinkTag - Current state of whether we're inside a think tag
	 * @param addThinkContent - Callback to handle content inside think tags
	 * @param addRegularContent - Callback to handle regular content outside think tags
	 * @returns Boolean indicating if we're still inside a think tag after processing
	 * @private
	 */
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

	/**
	 * Aborts any ongoing chat completion request.
	 * Cancels the current request and cleans up the abort controller.
	 * 
	 * @public
	 */
	public abort(): void {
		if (this.abortController) {
			this.abortController.abort();
			this.abortController = null;
		}
	}

	/**
	 * Injects a system message at the beginning of the conversation if configured in settings.
	 * Checks for existing system messages to avoid duplication and retrieves the system message
	 * from the current configuration settings.
	 * 
	 * @param messages - Array of chat messages to process
	 * @returns Array of messages with system message injected at the beginning if configured
	 * @private
	 */
	private injectSystemMessage(messages: ApiChatMessageData[]): ApiChatMessageData[] {
		const currentConfig = config();
		const systemMessage = currentConfig.systemMessage?.toString().trim();
		
		// If no system message is configured, return messages as-is
		if (!systemMessage) {
			return messages;
		}
		
		// Check if first message is already a system message
		if (messages.length > 0 && messages[0].role === 'system') {
			return messages;
		}
		
		// Inject system message at the beginning
		const systemMsg: ApiChatMessageData = {
			role: 'system',
			content: systemMessage
		};
		
		return [systemMsg, ...messages];
	}
}
