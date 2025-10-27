import { config } from '$lib/stores/settings.svelte';
import { selectedModelName } from '$lib/stores/models.svelte';
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
 *   - Handles error translation for server responses
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
 * - Request lifecycle management (abort, cleanup)
 */
export class ChatService {
	private abortControllers: Map<string, AbortController> = new Map();

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
		options: SettingsChatServiceOptions = {},
		conversationId?: string
	): Promise<string | void> {
		const {
			stream,
			onChunk,
			onComplete,
			onError,
			onReasoningChunk,
			onModel,
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

		const currentConfig = config();

		const requestId = conversationId || 'default';

		if (this.abortControllers.has(requestId)) {
			this.abortControllers.get(requestId)?.abort();
		}

		const abortController = new AbortController();
		this.abortControllers.set(requestId, abortController);

		const normalizedMessages: ApiChatMessageData[] = messages
			.map((msg) => {
				if ('id' in msg && 'convId' in msg && 'timestamp' in msg) {
					const dbMsg = msg as DatabaseMessage & { extra?: DatabaseMessageExtra[] };
					return ChatService.convertMessageToChatServiceData(dbMsg);
				} else {
					return msg as ApiChatMessageData;
				}
			})
			.filter((msg) => {
				if (msg.role === 'system') {
					const content = typeof msg.content === 'string' ? msg.content : '';

					return content.trim().length > 0;
				}

				return true;
			});

		const processedMessages = this.injectSystemMessage(normalizedMessages);

		const requestBody: ApiChatCompletionRequest = {
			messages: processedMessages.map((msg: ApiChatMessageData) => ({
				role: msg.role,
				content: msg.content
			})),
			stream
		};

		const modelSelectorEnabled = Boolean(currentConfig.modelSelectorEnabled);
		const activeModel = modelSelectorEnabled ? selectedModelName() : null;

		if (modelSelectorEnabled && activeModel) {
			requestBody.model = activeModel;
		}

		requestBody.reasoning_format = currentConfig.disableReasoningFormat ? 'none' : 'auto';

		if (temperature !== undefined) requestBody.temperature = temperature;
		if (max_tokens !== undefined) {
			// Set max_tokens to -1 (infinite) when explicitly configured as 0 or null
			requestBody.max_tokens = max_tokens !== null && max_tokens !== 0 ? max_tokens : -1;
		}

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
			const apiKey = currentConfig.apiKey?.toString().trim();

			const response = await fetch(`./v1/chat/completions`, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {})
				},
				body: JSON.stringify(requestBody),
				signal: abortController.signal
			});

			if (!response.ok) {
				const error = await this.parseErrorResponse(response);
				if (onError) {
					onError(error);
				}
				throw error;
			}

			if (stream) {
				await this.handleStreamResponse(
					response,
					onChunk,
					onComplete,
					onError,
					onReasoningChunk,
					onModel,
					conversationId,
					abortController.signal
				);
				return;
			} else {
				return this.handleNonStreamResponse(response, onComplete, onError, onModel);
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
					userFriendlyError.name = 'NetworkError';
				} else if (error.message.includes('ECONNREFUSED')) {
					userFriendlyError = new Error('Connection refused - server may be offline');
					userFriendlyError.name = 'NetworkError';
				} else if (error.message.includes('ETIMEDOUT')) {
					userFriendlyError = new Error('Request timed out - the server took too long to respond');
					userFriendlyError.name = 'TimeoutError';
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
		} finally {
			this.abortControllers.delete(requestId);
		}
	}

	/**
	 * Handles streaming response from the chat completion API
	 * @param response - The Response object from the fetch request
	 * @param onChunk - Optional callback invoked for each content chunk received
	 * @param onComplete - Optional callback invoked when the stream is complete with full response
	 * @param onError - Optional callback invoked if an error occurs during streaming
	 * @param onReasoningChunk - Optional callback invoked for each reasoning content chunk
	 * @param conversationId - Optional conversation ID for per-conversation state tracking
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
		onReasoningChunk?: (chunk: string) => void,
		onModel?: (model: string) => void,
		conversationId?: string,
		abortSignal?: AbortSignal
	): Promise<void> {
		const reader = response.body?.getReader();

		if (!reader) {
			throw new Error('No response body');
		}

		const decoder = new TextDecoder();
		let aggregatedContent = '';
		let fullReasoningContent = '';
		let hasReceivedData = false;
		let lastTimings: ChatMessageTimings | undefined;
		let streamFinished = false;
		let modelEmitted = false;

		try {
			let chunk = '';
			while (true) {
				if (abortSignal?.aborted) break;

				const { done, value } = await reader.read();
				if (done) break;

				if (abortSignal?.aborted) break;

				chunk += decoder.decode(value, { stream: true });
				const lines = chunk.split('\n');
				chunk = lines.pop() || '';

				for (const line of lines) {
					if (abortSignal?.aborted) break;

					if (line.startsWith('data: ')) {
						const data = line.slice(6);
						if (data === '[DONE]') {
							streamFinished = true;
							continue;
						}

						try {
							const parsed: ApiChatCompletionStreamChunk = JSON.parse(data);

							const chunkModel = this.extractModelName(parsed);
							if (chunkModel && !modelEmitted) {
								modelEmitted = true;
								onModel?.(chunkModel);
							}

							const content = parsed.choices[0]?.delta?.content;
							const reasoningContent = parsed.choices[0]?.delta?.reasoning_content;
							const timings = parsed.timings;
							const promptProgress = parsed.prompt_progress;

							if (timings || promptProgress) {
								this.updateProcessingState(timings, promptProgress, conversationId);
								if (timings) {
									lastTimings = timings;
								}
							}

							if (content) {
								hasReceivedData = true;
								aggregatedContent += content;
								if (!abortSignal?.aborted) {
									onChunk?.(content);
								}
							}

							if (reasoningContent) {
								hasReceivedData = true;
								fullReasoningContent += reasoningContent;
								if (!abortSignal?.aborted) {
									onReasoningChunk?.(reasoningContent);
								}
							}
						} catch (e) {
							console.error('Error parsing JSON chunk:', e);
						}
					}
				}

				if (abortSignal?.aborted) break;
			}

			if (abortSignal?.aborted) return;

			if (streamFinished) {
				if (!hasReceivedData && aggregatedContent.length === 0) {
					const noResponseError = new Error('No response received from server. Please try again.');
					throw noResponseError;
				}

				onComplete?.(aggregatedContent, fullReasoningContent || undefined, lastTimings);
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
		onError?: (error: Error) => void,
		onModel?: (model: string) => void
	): Promise<string> {
		try {
			const responseText = await response.text();

			if (!responseText.trim()) {
				const noResponseError = new Error('No response received from server. Please try again.');
				throw noResponseError;
			}

			const data: ApiChatCompletionResponse = JSON.parse(responseText);

			const responseModel = this.extractModelName(data);
			if (responseModel) {
				onModel?.(responseModel);
			}

			const content = data.choices[0]?.message?.content || '';
			const reasoningContent = data.choices[0]?.message?.reasoning_content;

			if (reasoningContent) {
				console.log('Full reasoning content:', reasoningContent);
			}

			if (!content.trim()) {
				const noResponseError = new Error('No response received from server. Please try again.');
				throw noResponseError;
			}

			onComplete?.(content, reasoningContent);

			return content;
		} catch (error) {
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

		// Handle legacy 'context' type from old webui (pasted content)
		const legacyContextFiles = message.extra.filter(
			(extra: DatabaseMessageExtra): extra is DatabaseMessageExtraLegacyContext =>
				extra.type === 'context'
		);

		for (const legacyContextFile of legacyContextFiles) {
			contentParts.push({
				type: 'text',
				text: `\n\n--- File: ${legacyContextFile.name} ---\n${legacyContextFile.content}`
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
	 * Aborts any ongoing chat completion request.
	 * Cancels the current request and cleans up the abort controller.
	 *
	 * @public
	 */
	public abort(conversationId?: string): void {
		if (conversationId) {
			const abortController = this.abortControllers.get(conversationId);
			if (abortController) {
				abortController.abort();
				this.abortControllers.delete(conversationId);
			}
		} else {
			for (const controller of this.abortControllers.values()) {
				controller.abort();
			}
			this.abortControllers.clear();
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

			const message = errorData.error?.message || 'Unknown server error';
			const error = new Error(message);
			error.name = response.status === 400 ? 'ServerError' : 'HttpError';

			return error;
		} catch {
			const fallback = new Error(`Server error (${response.status}): ${response.statusText}`);
			fallback.name = 'HttpError';
			return fallback;
		}
	}

	private extractModelName(data: unknown): string | undefined {
		const asRecord = (value: unknown): Record<string, unknown> | undefined => {
			return typeof value === 'object' && value !== null
				? (value as Record<string, unknown>)
				: undefined;
		};

		const getTrimmedString = (value: unknown): string | undefined => {
			return typeof value === 'string' && value.trim() ? value.trim() : undefined;
		};

		const root = asRecord(data);
		if (!root) return undefined;

		// 1) root (some implementations provide `model` at the top level)
		const rootModel = getTrimmedString(root.model);
		if (rootModel) return rootModel;

		// 2) streaming choice (delta) or final response (message)
		const firstChoice = Array.isArray(root.choices) ? asRecord(root.choices[0]) : undefined;
		if (!firstChoice) return undefined;

		// priority: delta.model (first chunk) else message.model (final response)
		const deltaModel = getTrimmedString(asRecord(firstChoice.delta)?.model);
		if (deltaModel) return deltaModel;

		const messageModel = getTrimmedString(asRecord(firstChoice.message)?.model);
		if (messageModel) return messageModel;

		// avoid guessing from non-standard locations (metadata, etc.)
		return undefined;
	}

	private updateProcessingState(
		timings?: ChatMessageTimings,
		promptProgress?: ChatMessagePromptProgress,
		conversationId?: string
	): void {
		const tokensPerSecond =
			timings?.predicted_ms && timings?.predicted_n
				? (timings.predicted_n / timings.predicted_ms) * 1000
				: 0;

		slotsService
			.updateFromTimingData(
				{
					prompt_n: timings?.prompt_n || 0,
					predicted_n: timings?.predicted_n || 0,
					predicted_per_second: tokensPerSecond,
					cache_n: timings?.cache_n || 0,
					prompt_progress: promptProgress
				},
				conversationId
			)
			.catch((error) => {
				console.warn('Failed to update processing state:', error);
			});
	}
}

export const chatService = new ChatService();
