/**
 * NotebookStore - Reactive State Store for Notebook Operations
 *
 */
import { CompletionService } from '$lib/services/completion.service';
import { config } from '$lib/stores/settings.svelte';
import { tokenize } from '$lib/services/tokenize.service';
import { STATS_UNITS } from '$lib/constants/processing-info';
import { getContextSize } from '$lib/stores/models.svelte';
import { ErrorDialogType } from '$lib/enums';
import { getETASecs } from '$lib/hooks/use-processing-state.svelte';
import { chatStore } from '$lib/stores/chat.svelte';

import type {
	ChatMessageTimings,
	ChatMessagePromptProgress,
	ErrorDialogState
} from '$lib/types/chat';
import type { ApiProcessingState } from '$lib/types';

export class NotebookStore {
	content = $state('');
	isGenerating = $state(false);
	abortController: AbortController | null = null;

	processingState = $state<ApiProcessingState | null>(null);

	totalTokens = $state(0);
	generationStartTokens = $state(0);
	generationEndTokens = $state(0);

	tokenizeTimeout: ReturnType<typeof setTimeout> | undefined;

	error = $state<ErrorDialogState | null>(null);

	previousContent = $state<string | null>(null);
	undoneContent = $state<string | null>(null);

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
			const currentConfig = config();
			const callbacks = {
				onChunk: (chunk: string) => {
					this.content += chunk;
				},
				onTimings: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => {
					this.updateProcessingStateFromTimings({
						prompt_n: timings?.prompt_n || 0,
						prompt_ms: promptProgress?.time_ms ?? timings?.prompt_ms,
						predicted_n: timings?.predicted_n || 0,
						predicted_ms: timings?.predicted_ms,
						cache_n: timings?.cache_n || 0,
						prompt_progress: promptProgress
					});
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

	dismissError() {
		this.error = null;
	}

	updateProcessingStateFromTimings(timingData: {
		prompt_n: number;
		prompt_ms?: number;
		predicted_n: number;
		predicted_ms?: number;
		cache_n: number;
		prompt_progress?: ChatMessagePromptProgress;
	}): void {
		this.processingState = chatStore.parseTimingData(timingData, getContextSize());
		this.totalTokens = this.processingState?.contextUsed ?? 0;
	}

	undo() {
		if (this.previousContent !== null) {
			this.undoneContent = this.content;
			this.content = this.previousContent;
			this.previousContent = null;
			this.totalTokens = this.generationStartTokens;
		}
	}

	redo() {
		if (this.undoneContent !== null) {
			this.previousContent = this.content;
			this.content = this.undoneContent;
			this.undoneContent = null;
			this.totalTokens = this.generationEndTokens;
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

	resetState() {
		this.isGenerating = false;
		if (this.processingState) {
			this.processingState.status = 'idle';
		}
	}

	updateTokenCount(model?: string) {
		if (this.tokenizeTimeout) {
			clearTimeout(this.tokenizeTimeout);
		}

		this.tokenizeTimeout = setTimeout(async () => {
			if (this.content.length === 0) {
				this.totalTokens = 0;
				return;
			}
			const tokens = await tokenize(this.content, model);
			this.totalTokens = tokens.length;
		}, 500);
	}

	getPromptProcessingText(): string {
		const state = this.processingState;
		let processing_text = 'Processing...';

		if (state?.promptProgress) {
			const { processed, total, time_ms, cache } = state.promptProgress;
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

	getProcessingContextDetail(): string {
		const state = this.processingState;
		const contextUsed = state?.contextUsed;
		const contextTotal = getContextSize();

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
}

export const notebookStore = new NotebookStore();
