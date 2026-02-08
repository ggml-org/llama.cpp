import { getJsonHeaders } from '$lib/utils';
import { ChatService } from '$lib/services/chat';

import type {
	ApiCompletionRequest,
	ApiCompletionResponse,
	ApiCompletionStreamChunk
} from '$lib/types/api';
import type { ChatMessageTimings, ChatMessagePromptProgress } from '$lib/types/chat';
import type { SettingsChatServiceOptions } from '$lib/types/settings';

/**
 * CompletionService - Low-level API communication layer for raw text completions.
 * Used in the notebook page.
 */
export class CompletionService {
	/**
	 * Sends a completion request to the llama.cpp server.
	 * Supports both streaming and non-streaming responses.
	 *
	 * @param prompt - The text prompt to complete
	 * @param options - Configuration options for the completion request
	 * @returns {Promise<string | void>} that resolves to the complete response string (non-streaming) or void (streaming)
	 * @throws {Error} if the request fails or is aborted
	 */
	static async sendCompletion(
		prompt: string,
		options: SettingsChatServiceOptions = {},
		signal?: AbortSignal
	): Promise<string | void> {
		const {
			stream,
			onChunk,
			onComplete,
			onError,
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
			timings_per_token
		} = options;

		const requestBody: ApiCompletionRequest = {
			prompt,
			stream
		};

		// Include model in request if provided
		if (options.model) {
			requestBody.model = options.model;
		}

		if (temperature !== undefined) requestBody.temperature = temperature;
		if (max_tokens !== undefined) {
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
			const response = await fetch(`./completion`, {
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
				await CompletionService.handleCompletionStreamResponse(
					response,
					onChunk,
					onComplete,
					onError,
					onModel,
					onTimings,
					signal
				);
				return;
			} else {
				return CompletionService.handleCompletionNonStreamResponse(
					response,
					onComplete,
					onError,
					onModel
				);
			}
		} catch (error) {
			if (error instanceof Error && error.name === 'AbortError') {
				console.log('Completion request was aborted');
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
				userFriendlyError = new Error('Unknown error occurred while sending completion');
			}

			console.error('Error in sendCompletion:', error);
			if (onError) {
				onError(userFriendlyError);
			}
			throw userFriendlyError;
		}
	}

	/**
	 * Handles streaming response from the completion API
	 */
	private static async handleCompletionStreamResponse(
		response: Response,
		onChunk?: (chunk: string) => void,
		onComplete?: (
			response: string,
			reasoningContent?: string,
			timings?: ChatMessageTimings,
			toolCalls?: string
		) => void,
		onError?: (error: Error) => void,
		onModel?: (model: string) => void,
		onTimings?: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => void,
		abortSignal?: AbortSignal
	): Promise<void> {
		const reader = response.body?.getReader();

		if (!reader) {
			throw new Error('No response body');
		}

		const decoder = new TextDecoder();
		let aggregatedContent = '';
		let lastTimings: ChatMessageTimings | undefined;
		let streamFinished = false;
		let modelEmitted = false;

		try {
			let chunk = '';
			while (true) {
				if (abortSignal?.aborted) {
					break;
				}

				const { done, value } = await reader.read();
				if (done) {
					break;
				}

				if (abortSignal?.aborted) {
					break;
				}

				chunk += decoder.decode(value, { stream: true });
				const lines = chunk.split('\n');
				chunk = lines.pop() || '';

				for (const line of lines) {
					if (abortSignal?.aborted) {
						break;
					}

					if (line.startsWith('data: ')) {
						const data = line.slice(6);

						try {
							const parsed: ApiCompletionStreamChunk = JSON.parse(data);
							const content = parsed.content;
							const timings = parsed.timings;
							const model = parsed.model;
							const promptProgress = parsed.prompt_progress;

							if (parsed.stop) {
								streamFinished = true;
							}

							if (model && !modelEmitted) {
								modelEmitted = true;
								onModel?.(model);
							}

							if (promptProgress) {
								ChatService.notifyTimings(undefined, promptProgress, onTimings);
							}

							if (timings) {
								ChatService.notifyTimings(timings, promptProgress, onTimings);
								lastTimings = timings;
							}

							if (content) {
								aggregatedContent += content;
								if (!abortSignal?.aborted) {
									onChunk?.(content);
								}
							}
						} catch (e) {
							console.error('Error parsing JSON chunk:', e);
						}
					}
				}

				if (streamFinished) {
					break;
				}
			}

			if (abortSignal?.aborted) {
				return;
			}

			if (streamFinished) {
				onComplete?.(aggregatedContent, undefined, lastTimings, undefined);
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
	 * Handles non-streaming response from the completion API
	 */
	private static async handleCompletionNonStreamResponse(
		response: Response,
		onComplete?: (
			response: string,
			reasoningContent?: string,
			timings?: ChatMessageTimings,
			toolCalls?: string
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

			const data: ApiCompletionResponse = JSON.parse(responseText);

			if (data.model) {
				onModel?.(data.model);
			}

			const content = data.content || '';

			if (!content.trim()) {
				const noResponseError = new Error('No response received from server. Please try again.');
				throw noResponseError;
			}

			onComplete?.(content, undefined, data.timings, undefined);

			return content;
		} catch (error) {
			const err = error instanceof Error ? error : new Error('Parse error');
			onError?.(err);
			throw err;
		}
	}
}
