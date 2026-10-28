import { ChatService } from '$lib/services/chat.service';
import type { ApiCompletionRequest, ApiCompletionStreamChunk } from '$lib/types/api';
import type { ChatMessageTimings } from '$lib/types/chat';
import type { CompletionServiceCallbacks, SettingsChatServiceOptions } from '$lib/types/settings';
import { getJsonHeaders } from '$lib/utils';

/**
 * CompletionService - Low-level API communication layer for raw text completions.
 * Used in the notebook page.
 */
export class CompletionService {
	/**
	 * Sends a completion request to the llama.cpp server.
	 * Supports only streaming responses.
	 *
	 * @param prompt - The text prompt to complete
	 * @param callbacks - Callbacks methods (onChunk, onComplete, ...)
	 * @param options - Configuration options for the completion request
	 * @returns {Promise<void>} that resolves to void
	 * @throws {Error} if the request fails or is aborted
	 */
	static async sendCompletion(
		prompt: string,
		callbacks: CompletionServiceCallbacks,
		options: SettingsChatServiceOptions = {},
		signal?: AbortSignal
	): Promise<string | void> {
		const {
			backend_sampling,
			customJson,
			dry_allowed_length,
			dry_base,
			dry_multiplier,
			dry_penalty_last_n,
			dynatemp_exponent,
			// Sampling parameters
			dynatemp_range,
			frequency_penalty,
			max_tokens,
			min_p,
			presence_penalty,
			// Penalty parameters
			repeat_last_n,
			repeat_penalty,
			// Other parameters
			samplers,
			// Generation parameters
			temperature,
			timings_per_token,
			top_k,
			top_p,
			typ_p,
			xtc_probability,
			xtc_threshold
		} = options;
		// We only support streaming responses
		const stream: boolean = true;
		const cache_prompt: boolean = true;
		const return_progress: boolean = true;
		const requestBody: ApiCompletionRequest = {
			cache_prompt,
			prompt,
			return_progress,
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

		if (typ_p !== undefined) {
			// On the completion endpoint, typ_p is called typical_p
			requestBody.typical_p = typ_p;
		}

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

		if (customJson) {
			try {
				const customParams = typeof customJson === 'string' ? JSON.parse(customJson) : customJson;

				Object.assign(requestBody, customParams);
			} catch (error) {
				console.warn('Failed to parse custom parameters:', error);
			}
		}

		try {
			const response = await fetch(`./completion`, {
				body: JSON.stringify(requestBody),
				headers: getJsonHeaders(),
				method: 'POST',
				signal
			});

			if (!response.ok) {
				const error = await ChatService.parseErrorResponse(response);

				callbacks.onError?.(error);

				throw error;
			}

			await CompletionService.handleCompletionStreamResponse(response, callbacks, signal);

			return;
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
			callbacks.onError?.(userFriendlyError);

			throw userFriendlyError;
		}
	}

	/**
	 * Handles streaming response from the completion API
	 */
	private static async handleCompletionStreamResponse(
		response: Response,
		callbacks: CompletionServiceCallbacks,
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
								callbacks.onModel?.(model);
							}

							if (promptProgress) {
								callbacks.onTimings?.(undefined, promptProgress);
							}

							if (timings) {
								callbacks.onTimings?.(timings, promptProgress);
								lastTimings = timings;
							}

							if (content) {
								aggregatedContent += content;

								if (!abortSignal?.aborted) {
									callbacks.onChunk?.(content);
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
				callbacks.onComplete?.(aggregatedContent, lastTimings);
			}
		} catch (error) {
			const err = error instanceof Error ? error : new Error('Stream error');

			callbacks.onError?.(err);

			throw err;
		} finally {
			reader.releaseLock();
		}
	}
}
