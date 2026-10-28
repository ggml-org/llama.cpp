/**
 * NotebookStore - Reactive State Store for Notebook Operations
 *
 */
import { STATS_UNITS } from '$lib/constants';
import { ErrorDialogType } from '$lib/enums';
import { getETASecs } from '$lib/hooks/use-processing-state.svelte';
import { CompletionService } from '$lib/services/completion.service';
import { tokenize } from '$lib/services/tokenize.service';
import { chatStore } from '$lib/stores/chat/index.svelte';
import { modelsStore } from '$lib/stores/models/index.svelte';
import { settingsStore } from '$lib/stores/settings/index.svelte';
import type { ApiProcessingState } from '$lib/types';
import type {
	ChatMessagePromptProgress,
	ChatMessageTimings,
	ErrorDialogState
} from '$lib/types/chat';

export class NotebookStore {
	abortController: AbortController | null = null;
	content = $state('');
	error = $state<ErrorDialogState | null>(null);

	generationEndTokens = $state(0);

	generationStartTokens = $state(0);
	isGenerating = $state(false);
	previousContent = $state<string | null>(null);

	processingState = $state<ApiProcessingState | null>(null);

	tokenizeTimeout: ReturnType<typeof setTimeout> | undefined;

	totalTokens = $state(0);
	undoneContent = $state<string | null>(null);

	dismissError() {
		this.error = null;
	}

	async generate(model?: string) {
		if (this.isGenerating) return;

		this.previousContent = this.content;
		this.undoneContent = null;
		this.isGenerating = true;
		this.processingState = null;
		this.abortController = new AbortController();
		this.error = null;

		// Save number of tokens before generation
		this.generationStartTokens = this.totalTokens;

		try {
			const currentConfig = settingsStore.config;
			const callbacks = {
				onChunk: (chunk: string) => {
					this.content += chunk;
				},
				onComplete: () => {
					this.resetState();
				},
				onError: (error: unknown) => {
					if (error instanceof Error && error.name === 'AbortError') {
						// aborted by user
					} else {
						console.error('Notebook generation error:', error);
						this.error = {
							message: error instanceof Error ? error.message : String(error),
							type: ErrorDialogType.SERVER
						};
					}

					this.resetState();
				},
				onTimings: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => {
					this.updateProcessingStateFromTimings({
						cache_n: timings?.cache_n || 0,
						predicted_ms: timings?.predicted_ms,
						predicted_n: timings?.predicted_n || 0,
						prompt_ms: promptProgress?.time_ms ?? timings?.prompt_ms,
						prompt_n: timings?.prompt_n || 0,
						prompt_progress: promptProgress
					});
				}
			};

			await CompletionService.sendCompletion(
				this.content,
				callbacks,
				{
					...currentConfig,
					model,
					timings_per_token: true
				},
				this.abortController.signal
			);
		} catch (error) {
			console.error('Notebook generation failed:', error);
			this.error = {
				message: error instanceof Error ? error.message : String(error),
				type: ErrorDialogType.SERVER
			};
			this.resetState();
		}
		// Save number of tokens after generation
		this.generationEndTokens = this.totalTokens;
	}

	getProcessingContextDetail(): string {
		const state = this.processingState;
		const contextUsed = state?.contextUsed;
		const contextTotal = modelsStore.contextSize;

		if (
			typeof contextUsed === 'number' &&
			typeof contextTotal === 'number' &&
			contextUsed >= 0 &&
			contextTotal > 0
		) {
			const contextPercent = Math.round((contextUsed / contextTotal) * 100);

			return `Context: ${contextUsed}/${contextTotal} (${contextPercent}%)`;
		}

		return '';
	}

	getProcessingDetails(): string[] {
		const details: string[] = [];
		const state = this.processingState;

		if (!state) {
			return [];
		}

		const contextDetail = this.getProcessingContextDetail();

		if (contextDetail) {
			details.push(contextDetail);
		}

		if (state.tokensDecoded > 0) {
			if (state.outputTokensMax <= 0) {
				details.push(`Output: ${state.tokensDecoded}/∞`);
			} else {
				const outputPercent = Math.round((state.tokensDecoded / state.outputTokensMax) * 100);

				details.push(`Output: ${state.tokensDecoded}/${state.outputTokensMax} (${outputPercent}%)`);
			}
		}

		if (state.tokensDecoded > 0 && state.tokensPerSecond && state.tokensPerSecond > 0) {
			details.push(`${state.tokensPerSecond.toFixed(1)} ${STATS_UNITS.TOKENS_PER_SECOND}`);
		}

		return details;
	}

	getPromptProcessingText(): string {
		const state = this.processingState;

		let processing_text = 'Processing...';

		if (state?.promptProgress) {
			const { cache, processed, time_ms, total } = state.promptProgress;
			const actualProcessed = processed - cache;
			const actualTotal = total - cache;

			if (actualProcessed < actualTotal && actualProcessed > 0) {
				const percent = Math.round((actualProcessed / actualTotal) * 100);
				const eta = getETASecs(actualProcessed, actualTotal, time_ms);

				if (eta !== undefined) {
					const etaSecs = Math.ceil(eta);

					processing_text = `Processing ${percent}% (ETA: ${etaSecs}s)`;
				} else {
					processing_text = `Processing ${percent}%`;
				}
			}
		}

		return processing_text;
	}

	redo() {
		if (this.undoneContent !== null) {
			this.previousContent = this.content;
			this.content = this.undoneContent;
			this.undoneContent = null;
			this.totalTokens = this.generationEndTokens;
		}
	}

	resetState() {
		this.isGenerating = false;

		if (this.processingState) {
			this.processingState.status = 'idle';
			chatStore.processing.setState('notebook', this.processingState);
		}
	}

	resetUndoRedo() {
		this.previousContent = null;
		this.undoneContent = null;
	}

	stop() {
		if (this.abortController) {
			this.abortController.abort();
			this.abortController = null;
		}

		this.resetState();
	}

	undo() {
		if (this.previousContent !== null) {
			this.undoneContent = this.content;
			this.content = this.previousContent;
			this.previousContent = null;
			this.totalTokens = this.generationStartTokens;
		}
	}

	updateProcessingStateFromTimings(timingData: {
		prompt_n: number;
		prompt_ms?: number;
		predicted_n: number;
		predicted_ms?: number;
		cache_n: number;
		prompt_progress?: ChatMessagePromptProgress;
	}): void {
		this.processingState = chatStore.processing.parseTimingData(timingData);
		this.totalTokens = this.processingState?.contextUsed ?? 0;

		if (this.processingState) {
			chatStore.processing.setState('notebook', this.processingState);
		}
	}

	updateTokenCount(model?: string) {
		if (this.tokenizeTimeout) {
			clearTimeout(this.tokenizeTimeout);
		}

		this.tokenizeTimeout = setTimeout(async () => {
			if (this.content.length === 0) {
				this.totalTokens = 0;
				this.processingState = null;
				chatStore.processing.setState('notebook', null);

				return;
			}

			const tokens = await tokenize(this.content, model);

			this.totalTokens = tokens.length;

			if (!this.isGenerating) {
				const contextTotal = modelsStore.contextSize;

				if (this.processingState) {
					this.processingState.contextUsed = this.totalTokens;
					this.processingState.promptTokens = this.totalTokens;
					this.processingState.contextTotal = contextTotal;
					this.processingState.tokensRemaining = (contextTotal ?? 0) - this.totalTokens;
				} else {
					const state: ApiProcessingState = {
						cacheTokens: 0,
						contextTotal,
						contextUsed: this.totalTokens,
						hasNextToken: false,
						outputTokensMax: settingsStore.config.max_tokens || -1,
						outputTokensUsed: 0,
						promptTokens: this.totalTokens,
						speculative: false,
						status: 'idle',
						temperature: settingsStore.config.temperature ?? 0.8,
						tokensDecoded: 0,
						tokensPerSecond: 0,
						tokensRemaining: (contextTotal ?? 0) - this.totalTokens,
						topP: settingsStore.config.top_p ?? 0.95
					};

					this.processingState = state;
				}

				chatStore.processing.setState('notebook', this.processingState);
			}
		}, 500);
	}
}

export const notebookStore = new NotebookStore();
