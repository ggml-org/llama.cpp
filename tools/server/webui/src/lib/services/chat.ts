import { getJsonHeaders } from '$lib/utils';
import { AttachmentType } from '$lib/enums';

/**
 * ChatService - Low-level API communication layer for Chat Completions
 *
 * **Terminology - Chat vs Conversation:**
 * - **Chat**: The active interaction space with the Chat Completions API. This service
 *   handles the real-time communication with the AI backend - sending messages, receiving
 *   streaming responses, and managing request lifecycles. "Chat" is ephemeral and runtime-focused.
 * - **Conversation**: The persistent database entity storing all messages and metadata.
 *   Managed by ConversationsService/Store, conversations persist across sessions.
 *
 * This service handles direct communication with the llama-server's Chat Completions API.
 * It provides the network layer abstraction for AI model interactions while remaining
 * stateless and focused purely on API communication.
 *
 * **Architecture & Relationships:**
 * - **ChatService** (this class): Stateless API communication layer
 *   - Handles HTTP requests/responses with the llama-server
 *   - Manages streaming and non-streaming response parsing
 *   - Provides per-conversation request abortion capabilities
 *   - Converts database messages to API format
 *   - Handles error translation for server responses
 *
 * - **chatStore**: Uses ChatService for all AI model communication
 * - **conversationsStore**: Provides message context for API requests
 *
 * **Key Responsibilities:**
 * - Message format conversion (DatabaseMessage → API format)
 * - Streaming response handling with real-time callbacks
 * - Reasoning content extraction and processing
 * - File attachment processing (images, PDFs, audio, text)
 * - Request lifecycle management (abort via AbortSignal)
 */
export class ChatService {
	// ─────────────────────────────────────────────────────────────────────────────
	// Messaging
	// ─────────────────────────────────────────────────────────────────────────────

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
	static async sendMessage(
		messages: ApiChatMessageData[] | (DatabaseMessage & { extra?: DatabaseMessageExtra[] })[],
		options: SettingsChatServiceOptions = {},
		conversationId?: string,
		signal?: AbortSignal
	): Promise<string | void> {
		const {
			stream,
			onChunk,
			onComplete,
			onError,
			onReasoningChunk,
			onToolCallChunk,
			onModel,
			onTimings,
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
			backend_sampling,
			custom,
			timings_per_token,
			// Config options
			disableReasoningFormat
		} = options;

		const normalizedMessages: ApiChatMessageData[] = messages
			.map((msg) => {
				if ('id' in msg && 'convId' in msg && 'timestamp' in msg) {
					const dbMsg = msg as DatabaseMessage & { extra?: DatabaseMessageExtra[] };
					return ChatService.convertDbMessageToApiChatMessageData(dbMsg);
				} else {
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

		const requestBody: ApiChatCompletionRequest = {
			messages: normalizedMessages.map((msg: ApiChatMessageData) => ({
				role: msg.role,
				content: msg.content
			})),
			stream,
			return_progress: stream ? true : undefined
		};

		// Include model in request if provided (required in ROUTER mode)
		if (options.model) {
			requestBody.model = options.model;
		}

		requestBody.reasoning_format = disableReasoningFormat ? 'none' : 'auto';

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

		if (backend_sampling !== undefined) requestBody.backend_sampling = backend_sampling;

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
			const response = await fetch(`./v1/chat/completions`, {
				method: 'POST',
				headers: getJsonHeaders(),
				body: JSON.stringify(requestBody),
				signal
			});

			if (!response.ok) {
				const error = await ChatService.parseErrorResponse(response);
				if (onError) {
					onError(error);
				}
				throw error;
			}

			if (stream) {
				await ChatService.handleStreamResponse(
					response,
					onChunk,
					onComplete,
					onError,
					onReasoningChunk,
					onToolCallChunk,
					onModel,
					onTimings,
					conversationId,
					signal
				);
				return;
			} else {
				return ChatService.handleNonStreamResponse(
					response,
					onComplete,
					onError,
					onToolCallChunk,
					onModel
				);
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
		}
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Streaming
	// ─────────────────────────────────────────────────────────────────────────────

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
	private static async handleStreamResponse(
		response: Response,
		onChunk?: (chunk: string) => void,
		onComplete?: (
			response: string,
			reasoningContent?: string,
			timings?: ChatMessageTimings,
			toolCalls?: string
		) => void,
		onError?: (error: Error) => void,
		onReasoningChunk?: (chunk: string) => void,
		onToolCallChunk?: (chunk: string) => void,
		onModel?: (model: string) => void,
		onTimings?: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => void,
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
		let aggregatedToolCalls: ApiChatCompletionToolCall[] = [];
		let lastTimings: ChatMessageTimings | undefined;
		let streamFinished = false;
		let modelEmitted = false;
		let toolCallIndexOffset = 0;
		let hasOpenToolCallBatch = false;

		const finalizeOpenToolCallBatch = () => {
			if (!hasOpenToolCallBatch) {
				return;
			}

			toolCallIndexOffset = aggregatedToolCalls.length;
			hasOpenToolCallBatch = false;
		};

		const processToolCallDelta = (toolCalls?: ApiChatCompletionToolCallDelta[]) => {
			if (!toolCalls || toolCalls.length === 0) {
				return;
			}

			aggregatedToolCalls = ChatService.mergeToolCallDeltas(
				aggregatedToolCalls,
				toolCalls,
				toolCallIndexOffset
			);

			if (aggregatedToolCalls.length === 0) {
				return;
			}

			hasOpenToolCallBatch = true;

			const serializedToolCalls = JSON.stringify(aggregatedToolCalls);

			if (!serializedToolCalls) {
				return;
			}

			if (!abortSignal?.aborted) {
				onToolCallChunk?.(serializedToolCalls);
			}
		};

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
							const content = parsed.choices[0]?.delta?.content;
							const reasoningContent = parsed.choices[0]?.delta?.reasoning_content;
							const toolCalls = parsed.choices[0]?.delta?.tool_calls;
							const timings = parsed.timings;
							const promptProgress = parsed.prompt_progress;

							const chunkModel = ChatService.extractModelName(parsed);
							if (chunkModel && !modelEmitted) {
								modelEmitted = true;
								onModel?.(chunkModel);
							}

							if (promptProgress) {
								ChatService.notifyTimings(undefined, promptProgress, onTimings);
							}

							if (timings) {
								ChatService.notifyTimings(timings, promptProgress, onTimings);
								lastTimings = timings;
							}

							if (content) {
								finalizeOpenToolCallBatch();
								aggregatedContent += content;
								if (!abortSignal?.aborted) {
									onChunk?.(content);
								}
							}

							if (reasoningContent) {
								finalizeOpenToolCallBatch();
								fullReasoningContent += reasoningContent;
								if (!abortSignal?.aborted) {
									onReasoningChunk?.(reasoningContent);
								}
							}

							processToolCallDelta(toolCalls);
						} catch (e) {
							console.error('Error parsing JSON chunk:', e);
						}
					}
				}

				if (abortSignal?.aborted) break;
			}

			if (abortSignal?.aborted) return;

			if (streamFinished) {
				finalizeOpenToolCallBatch();

				const finalToolCalls =
					aggregatedToolCalls.length > 0 ? JSON.stringify(aggregatedToolCalls) : undefined;

				onComplete?.(
					aggregatedContent,
					fullReasoningContent || undefined,
					lastTimings,
					finalToolCalls
				);
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
	private static async handleNonStreamResponse(
		response: Response,
		onComplete?: (
			response: string,
			reasoningContent?: string,
			timings?: ChatMessageTimings,
			toolCalls?: string
		) => void,
		onError?: (error: Error) => void,
		onToolCallChunk?: (chunk: string) => void,
		onModel?: (model: string) => void
	): Promise<string> {
		try {
			const responseText = await response.text();

			if (!responseText.trim()) {
				const noResponseError = new Error('No response received from server. Please try again.');
				throw noResponseError;
			}

			const data: ApiChatCompletionResponse = JSON.parse(responseText);

			const responseModel = ChatService.extractModelName(data);
			if (responseModel) {
				onModel?.(responseModel);
			}

			const content = data.choices[0]?.message?.content || '';
			const reasoningContent = data.choices[0]?.message?.reasoning_content;
			const toolCalls = data.choices[0]?.message?.tool_calls;

			if (reasoningContent) {
				console.log('Full reasoning content:', reasoningContent);
			}

			let serializedToolCalls: string | undefined;

			if (toolCalls && toolCalls.length > 0) {
				const mergedToolCalls = ChatService.mergeToolCallDeltas([], toolCalls);

				if (mergedToolCalls.length > 0) {
					serializedToolCalls = JSON.stringify(mergedToolCalls);
					if (serializedToolCalls) {
						onToolCallChunk?.(serializedToolCalls);
					}
				}
			}

			if (!content.trim() && !serializedToolCalls) {
				const noResponseError = new Error('No response received from server. Please try again.');
				throw noResponseError;
			}

			onComplete?.(content, reasoningContent, undefined, serializedToolCalls);

			return content;
		} catch (error) {
			const err = error instanceof Error ? error : new Error('Parse error');

			onError?.(err);

			throw err;
		}
	}

	/**
	 * Merges tool call deltas into an existing array of tool calls.
	 * Handles both existing and new tool calls, updating existing ones and adding new ones.
	 *
	 * @param existing - The existing array of tool calls to merge into
	 * @param deltas - The array of tool call deltas to merge
	 * @param indexOffset - Optional offset to apply to the index of new tool calls
	 * @returns {ApiChatCompletionToolCall[]} The merged array of tool calls
	 */
	private static mergeToolCallDeltas(
		existing: ApiChatCompletionToolCall[],
		deltas: ApiChatCompletionToolCallDelta[],
		indexOffset = 0
	): ApiChatCompletionToolCall[] {
		const result = existing.map((call) => ({
			...call,
			function: call.function ? { ...call.function } : undefined
		}));

		for (const delta of deltas) {
			const index =
				typeof delta.index === 'number' && delta.index >= 0
					? delta.index + indexOffset
					: result.length;

			while (result.length <= index) {
				result.push({ function: undefined });
			}

			const target = result[index]!;

			if (delta.id) {
				target.id = delta.id;
			}

			if (delta.type) {
				target.type = delta.type;
			}

			if (delta.function) {
				const fn = target.function ? { ...target.function } : {};

				if (delta.function.name) {
					fn.name = delta.function.name;
				}

				if (delta.function.arguments) {
					fn.arguments = (fn.arguments ?? '') + delta.function.arguments;
				}

				target.function = fn;
			}
		}

		return result;
	}

	// ─────────────────────────────────────────────────────────────────────────────
	// Conversion
	// ─────────────────────────────────────────────────────────────────────────────

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
	static convertDbMessageToApiChatMessageData(
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
				extra.type === AttachmentType.IMAGE
		);

		for (const image of imageFiles) {
			contentParts.push({
				type: 'image_url',
				image_url: { url: image.base64Url }
			});
		}

		const textFiles = message.extra.filter(
			(extra: DatabaseMessageExtra): extra is DatabaseMessageExtraTextFile =>
				extra.type === AttachmentType.TEXT
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
				extra.type === AttachmentType.LEGACY_CONTEXT
		);

		for (const legacyContextFile of legacyContextFiles) {
			contentParts.push({
				type: 'text',
				text: `\n\n--- File: ${legacyContextFile.name} ---\n${legacyContextFile.content}`
			});
		}

		const audioFiles = message.extra.filter(
			(extra: DatabaseMessageExtra): extra is DatabaseMessageExtraAudioFile =>
				extra.type === AttachmentType.AUDIO
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
				extra.type === AttachmentType.PDF
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

	// ─────────────────────────────────────────────────────────────────────────────
	// Utilities
	// ─────────────────────────────────────────────────────────────────────────────

	/**
	 * Parses error response and creates appropriate error with context information
	 * @param response - HTTP response object
	 * @returns Promise<Error> - Parsed error with context info if available
	 */
	private static async parseErrorResponse(
		response: Response
	): Promise<Error & { contextInfo?: { n_prompt_tokens: number; n_ctx: number } }> {
		try {
			const errorText = await response.text();
			const errorData: ApiErrorResponse = JSON.parse(errorText);

			const message = errorData.error?.message || 'Unknown server error';
			const error = new Error(message) as Error & {
				contextInfo?: { n_prompt_tokens: number; n_ctx: number };
			};
			error.name = response.status === 400 ? 'ServerError' : 'HttpError';

			if (errorData.error && 'n_prompt_tokens' in errorData.error && 'n_ctx' in errorData.error) {
				error.contextInfo = {
					n_prompt_tokens: errorData.error.n_prompt_tokens,
					n_ctx: errorData.error.n_ctx
				};
			}

			return error;
		} catch {
			const fallback = new Error(
				`Server error (${response.status}): ${response.statusText}`
			) as Error & {
				contextInfo?: { n_prompt_tokens: number; n_ctx: number };
			};
			fallback.name = 'HttpError';
			return fallback;
		}
	}

	/**
	 * Extracts model name from Chat Completions API response data.
	 * Handles various response formats including streaming chunks and final responses.
	 *
	 * WORKAROUND: In single model mode, llama-server returns a default/incorrect model name
	 * in the response. We override it with the actual model name from serverStore.
	 *
	 * @param data - Raw response data from the Chat Completions API
	 * @returns Model name string if found, undefined otherwise
	 * @private
	 */
	private static extractModelName(data: unknown): string | undefined {
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

	/**
	 * Calls the onTimings callback with timing data from streaming response.
	 *
	 * @param timings - Timing information from the Chat Completions API response
	 * @param promptProgress - Prompt processing progress data
	 * @param onTimingsCallback - Callback function to invoke with timing data
	 * @private
	 */
	private static notifyTimings(
		timings: ChatMessageTimings | undefined,
		promptProgress: ChatMessagePromptProgress | undefined,
		onTimingsCallback:
			| ((timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => void)
			| undefined
	): void {
		if (!onTimingsCallback || (!timings && !promptProgress)) return;

		onTimingsCallback(timings, promptProgress);
	}
}
