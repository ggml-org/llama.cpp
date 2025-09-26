import { config } from '$lib/stores/settings.svelte';
import { slotsService } from './slots';
/**
 * ChatService - Low-level API communication layer for llama.cpp server interactions
 *
 * This service handles direct communication with the llama.cpp server's chat completion API.
 * It provides the network layer abstraction for AI model interactions while remaining
 * stateless and focused purely on API communication.
 *
 * **Architecture & Relationship with ChatStore:**
 * - **ChatService** (this class): Stateless API communication layer
 *   - Handles HTTP requests/responses with llama.cpp server
 *   - Manages streaming and non-streaming response parsing
 *   - Provides request abortion capabilities
 *   - Converts database messages to API format
 *   - Handles error translation and context detection
 *
 * - **ChatStore**: Stateful orchestration and UI state management
 *   - Uses ChatService for all AI model communication
 *   - Manages conversation state, message history, and UI reactivity
 *   - Coordinates with DatabaseStore for persistence
 *   - Handles complex workflows like branching and regeneration
 *
 * **Key Responsibilities:**
 * - Message format conversion (DatabaseMessage → API format)
 * - Streaming response handling with real-time callbacks
 * - Reasoning content extraction and processing
 * - File attachment processing (images, PDFs, audio, text)
 * - Context error detection and reporting
 * - Request lifecycle management (abort, cleanup)
 */
export class ChatService {
	private abortController: AbortController | null = null;

	/**
	 * Sends a chat completion request to the llama.cpp server.
	 * Supports both streaming and non-streaming responses with comprehensive parameter configuration.
	 * Automatically converts database messages with attachments to the appropriate API format.
	 *
	 * @param messages - Array of chat messages to send to the API (supports both ApiChatMessageData and DatabaseMessage with attachments)
	 * @param options - Configuration options for the chat completion request. See `SettingsChatServiceOptions` type for details.
	 * @returns {Promise<string | void>} that resolves to the complete response string (non-streaming) or void (streaming)
	 * @throws {Error} if the request fails or is aborted
	 */
	async sendMessage(
		messages: ApiChatMessageData[] | (DatabaseMessage & { extra?: DatabaseMessageExtra[] })[],
		options: SettingsChatServiceOptions = {}
	): Promise<string | void> {
		const {
			stream,
			onChunk,
			onComplete,
			onError,
			// Generation parameters
			temperature,
			max_tokens,
			// Sampling parameters
			dynatemp_range,
			dynatemp_exponent,
			top_k,
			top_p,
			min_p,
			xtc_probability,
			xtc_threshold,
			typ_p,
			// Penalty parameters
			repeat_last_n,
			repeat_penalty,
			presence_penalty,
			frequency_penalty,
			dry_multiplier,
			dry_base,
			dry_allowed_length,
			dry_penalty_last_n,
			// Other parameters
			samplers,
			custom,
			timings_per_token
		} = options;

		// Cancel any ongoing request and create a new abort controller
		this.abort();
		this.abortController = new AbortController();

		// Convert database messages with attachments to API format if needed
		const normalizedMessages: ApiChatMessageData[] = messages
			.map((msg) => {
				// Check if this is a DatabaseMessage by checking for DatabaseMessage-specific fields
				if ('id' in msg && 'convId' in msg && 'timestamp' in msg) {
					// This is a DatabaseMessage, convert it
					const dbMsg = msg as DatabaseMessage & { extra?: DatabaseMessageExtra[] };
					return ChatService.convertMessageToChatServiceData(dbMsg);
				} else {
					// This is already an ApiChatMessageData object
					return msg as ApiChatMessageData;
				}
			})
			.filter((msg) => {
				// Filter out empty system messages
				if (msg.role === 'system') {
					const content = typeof msg.content === 'string' ? msg.content : '';

					return content.trim().length > 0;
				}

				return true;
			});

		// Build base request body with system message injection
		const processedMessages = this.injectSystemMessage(normalizedMessages);

		const requestBody: ApiChatCompletionRequest = {
			messages: processedMessages.map((msg: ApiChatMessageData) => ({
				role: msg.role,
				content: msg.content
			})),
			stream
		};

		requestBody.reasoning_format = 'auto';

		if (temperature !== undefined) requestBody.temperature = temperature;
		// Set max_tokens to -1 (infinite) if not provided or empty
		requestBody.max_tokens =
			max_tokens !== undefined && max_tokens !== null && max_tokens !== 0 ? max_tokens : -1;

		if (dynatemp_range !== undefined) requestBody.dynatemp_range = dynatemp_range;
		if (dynatemp_exponent !== undefined) requestBody.dynatemp_exponent = dynatemp_exponent;
		if (top_k !== undefined) requestBody.top_k = top_k;
		if (top_p !== undefined) requestBody.top_p = top_p;
		if (min_p !== undefined) requestBody.min_p = min_p;
		if (xtc_probability !== undefined) requestBody.xtc_probability = xtc_probability;
		if (xtc_threshold !== undefined) requestBody.xtc_threshold = xtc_threshold;
		if (typ_p !== undefined) requestBody.typ_p = typ_p;

		if (repeat_last_n !== undefined) requestBody.repeat_last_n = repeat_last_n;
		if (repeat_penalty !== undefined) requestBody.repeat_penalty = repeat_penalty;
		if (presence_penalty !== undefined) requestBody.presence_penalty = presence_penalty;
		if (frequency_penalty !== undefined) requestBody.frequency_penalty = frequency_penalty;
		if (dry_multiplier !== undefined) requestBody.dry_multiplier = dry_multiplier;
		if (dry_base !== undefined) requestBody.dry_base = dry_base;
		if (dry_allowed_length !== undefined) requestBody.dry_allowed_length = dry_allowed_length;
		if (dry_penalty_last_n !== undefined) requestBody.dry_penalty_last_n = dry_penalty_last_n;

		if (samplers !== undefined) {
			requestBody.samplers =
				typeof samplers === 'string'
					? samplers.split(';').filter((s: string) => s.trim())
					: samplers;
		}

		if (timings_per_token !== undefined) requestBody.timings_per_token = timings_per_token;

		if (custom) {
			try {
				const customParams = typeof custom === 'string' ? JSON.parse(custom) : custom;
				Object.assign(requestBody, customParams);
			} catch (error) {
				console.warn('Failed to parse custom parameters:', error);
			}
		}

		try {
			const currentConfig = config();
			const apiKey = currentConfig.apiKey?.toString().trim();

			const response = await fetch(`./v1/chat/completions`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {})
				},
				body: JSON.stringify(requestBody),
				signal: this.abortController.signal
			});

			if (!response.ok) {
				// Use the new parseErrorResponse method to handle structured errors
				const error = await this.parseErrorResponse(response);
				if (onError) {
					onError(error);
				}
				throw error;
			}

			if (stream) {
				return this.handleStreamResponse(
					response,
					onChunk,
					onComplete,
					onError,
					options.onReasoningChunk
				);
			} else {
				return this.handleNonStreamResponse(response, onComplete, onError);
			}
		} catch (error) {
			if (error instanceof Error && error.name === 'AbortError') {
				console.log('Chat completion request was aborted');
				return;
			}

			let userFriendlyError: Error;

			if (error instanceof Error) {
				if (error.name === 'TypeError' && error.message.includes('fetch')) {
					userFriendlyError = new Error(
						'Unable to connect to server - please check if the server is running'
					);
				} else if (error.message.includes('ECONNREFUSED')) {
					userFriendlyError = new Error('Connection refused - server may be offline');
				} else if (error.message.includes('ETIMEDOUT')) {
					userFriendlyError = new Error('Request timeout - server may be overloaded');
				} else {
					userFriendlyError = error;
				}
			} else {
				userFriendlyError = new Error('Unknown error occurred while sending message');
			}

			console.error('Error in sendMessage:', error);
			if (onError) {
				onError(userFriendlyError);
			}
			throw userFriendlyError;
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
		onComplete?: (
			response: string,
			reasoningContent?: string,
			timings?: ChatMessageTimings
		) => void,
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
		let regularContent = '';
		let insideThinkTag = false;
		let hasReceivedData = false;
		let lastTimings: ChatMessageTimings | undefined;

		try {
			let chunk = '';
			while (true) {
				const { done, value } = await reader.read();
				if (done) break;

				chunk += decoder.decode(value, { stream: true });
				const lines = chunk.split('\n');
				chunk = lines.pop() || ''; // Save incomplete line for next read

				for (const line of lines) {
					if (line.startsWith('data: ')) {
						const data = line.slice(6);
						if (data === '[DONE]') {
							if (!hasReceivedData && fullResponse.length === 0) {
								const contextError = new Error(
									'The request exceeds the available context size. Try increasing the context size or enable context shift.'
								);
								contextError.name = 'ContextError';
								onError?.(contextError);
								return;
							}

							onComplete?.(regularContent, fullReasoningContent || undefined, lastTimings);

							return;
						}

						try {
							const parsed: ApiChatCompletionStreamChunk = JSON.parse(data);

							const content = parsed.choices[0]?.delta?.content;
							const reasoningContent = parsed.choices[0]?.delta?.reasoning_content;
							const timings = parsed.timings;
							const promptProgress = parsed.prompt_progress;

							if (timings || promptProgress) {
								this.updateProcessingState(timings, promptProgress);

								// Store the latest timing data
								if (timings) {
									lastTimings = timings;
								}
							}

							if (content) {
								hasReceivedData = true;
								fullResponse += content;

								// Track the regular content before processing this chunk
								const regularContentBefore = regularContent;

								// Process content character by character to handle think tags
								insideThinkTag = this.processContentForThinkTags(
									content,
									insideThinkTag,
									() => {
										// Think content is ignored - we don't include it in API requests
									},
									(regularChunk) => {
										regularContent += regularChunk;
									}
								);

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

			if (!hasReceivedData && fullResponse.length === 0) {
				const contextError = new Error(
					'The request exceeds the available context size. Try increasing the context size or enable context shift.'
				);
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
		onComplete?: (
			response: string,
			reasoningContent?: string,
			timings?: ChatMessageTimings
		) => void,
		onError?: (error: Error) => void
	): Promise<string> {
		try {
			const responseText = await response.text();

			if (!responseText.trim()) {
				const contextError = new Error(
					'The request exceeds the available context size. Try increasing the context size or enable context shift.'
				);
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

			if (!content.trim()) {
				const contextError = new Error(
					'The request exceeds the available context size. Try increasing the context size or enable context shift.'
				);
				contextError.name = 'ContextError';
				onError?.(contextError);
				throw contextError;
			}

			onComplete?.(content, reasoningContent);

			return content;
		} catch (error) {
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
	static convertMessageToChatServiceData(
		message: DatabaseMessage & { extra?: DatabaseMessageExtra[] }
	): ApiChatMessageData {
		if (!message.extra || message.extra.length === 0) {
			return {
				role: message.role as 'user' | 'assistant' | 'system',
				content: message.content
			};
		}

		const contentParts: ApiChatMessageContentPart[] = [];

		if (message.content) {
			contentParts.push({
				type: 'text',
				text: message.content
			});
		}

		const imageFiles = message.extra.filter(
			(extra: DatabaseMessageExtra): extra is DatabaseMessageExtraImageFile =>
				extra.type === 'imageFile'
		);

		for (const image of imageFiles) {
			contentParts.push({
				type: 'image_url',
				image_url: { url: image.base64Url }
			});
		}

		const textFiles = message.extra.filter(
			(extra: DatabaseMessageExtra): extra is DatabaseMessageExtraTextFile =>
				extra.type === 'textFile'
		);

		for (const textFile of textFiles) {
			contentParts.push({
				type: 'text',
				text: `\n\n--- File: ${textFile.name} ---\n${textFile.content}`
			});
		}

		const audioFiles = message.extra.filter(
			(extra: DatabaseMessageExtra): extra is DatabaseMessageExtraAudioFile =>
				extra.type === 'audioFile'
		);

		for (const audio of audioFiles) {
			contentParts.push({
				type: 'input_audio',
				input_audio: {
					data: audio.base64Data,
					format: audio.mimeType.includes('wav') ? 'wav' : 'mp3'
				}
			});
		}

		const pdfFiles = message.extra.filter(
			(extra: DatabaseMessageExtra): extra is DatabaseMessageExtraPdfFile =>
				extra.type === 'pdfFile'
		);

		for (const pdfFile of pdfFiles) {
			if (pdfFile.processedAsImages && pdfFile.images) {
				for (let i = 0; i < pdfFile.images.length; i++) {
					contentParts.push({
						type: 'image_url',
						image_url: { url: pdfFile.images[i] }
					});
				}
			} else {
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
	 * Get server properties - static method for API compatibility
	 */
	static async getServerProps(): Promise<ApiLlamaCppServerProps> {
		try {
			const currentConfig = config();
			const apiKey = currentConfig.apiKey?.toString().trim();

			const response = await fetch(`./props`, {
				headers: {
					'Content-Type': 'application/json',
					...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {})
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
			if (!insideThinkTag && content.substring(i, i + 7) === '<think>') {
				insideThinkTag = true;
				i += 7; // Skip the <think> tag
				continue;
			}

			if (insideThinkTag && content.substring(i, i + 8) === '</think>') {
				insideThinkTag = false;
				i += 8; // Skip the </think> tag
				continue;
			}

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

		if (!systemMessage) {
			return messages;
		}

		if (messages.length > 0 && messages[0].role === 'system') {
			if (messages[0].content !== systemMessage) {
				const updatedMessages = [...messages];
				updatedMessages[0] = {
					role: 'system',
					content: systemMessage
				};
				return updatedMessages;
			}

			return messages;
		}

		const systemMsg: ApiChatMessageData = {
			role: 'system',
			content: systemMessage
		};

		return [systemMsg, ...messages];
	}

	/**
	 * Parses error response and creates appropriate error with context information
	 * @param response - HTTP response object
	 * @returns Promise<Error> - Parsed error with context info if available
	 */
	private async parseErrorResponse(response: Response): Promise<Error> {
		try {
			const errorText = await response.text();
			const errorData: ApiErrorResponse = JSON.parse(errorText);

			if (errorData.error?.type === 'exceed_context_size_error') {
				const contextError = errorData.error as ApiContextSizeError;
				const error = new Error(contextError.message);
				error.name = 'ContextError';
				// Attach structured context information
				(
					error as Error & {
						contextInfo?: { promptTokens: number; maxContext: number; estimatedTokens: number };
					}
				).contextInfo = {
					promptTokens: contextError.n_prompt_tokens,
					maxContext: contextError.n_ctx,
					estimatedTokens: contextError.n_prompt_tokens
				};
				return error;
			}

			// Fallback for other error types
			const message = errorData.error?.message || 'Unknown server error';
			return new Error(message);
		} catch {
			// If we can't parse the error response, return a generic error
			return new Error(`Server error (${response.status}): ${response.statusText}`);
		}
	}

	/**
	 * Updates the processing state with timing information from the server response
	 * @param timings - Timing data from the API response
	 * @param promptProgress - Progress data from the API response
	 */
	private updateProcessingState(
		timings?: ChatMessageTimings,
		promptProgress?: ChatMessagePromptProgress
	): void {
		// Calculate tokens per second from timing data
		const tokensPerSecond =
			timings?.predicted_ms && timings?.predicted_n
				? (timings.predicted_n / timings.predicted_ms) * 1000
				: 0;

		// Update slots service with timing data (async but don't wait)
		slotsService
			.updateFromTimingData({
				prompt_n: timings?.prompt_n || 0,
				predicted_n: timings?.predicted_n || 0,
				predicted_per_second: tokensPerSecond,
				cache_n: timings?.cache_n || 0,
				prompt_progress: promptProgress
			})
			.catch((error) => {
				console.warn('Failed to update processing state:', error);
			});
	}
}

export const chatService = new ChatService();
