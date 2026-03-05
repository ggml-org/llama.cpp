import { CompletionService } from '$lib/services/completion.service';
import { config } from '$lib/stores/settings.svelte';
import { tokenize } from '$lib/services/tokenize.service';
import type { ChatMessageTimings, ChatMessagePromptProgress } from '$lib/types/chat';
import { STATS_UNITS } from '$lib/constants/processing-info';
import { contextSize, isRouterMode } from '$lib/stores/server.svelte';
import { selectedModelContextSize } from '$lib/stores/models.svelte';

export class NotebookStore {
	content = $state('');
	isGenerating = $state(false);
	abortController: AbortController | null = null;

	// Statistics
	cacheTokens = $state(0);
	promptTokens = $state(0);
	promptMs = $state(0);
	predictedTokens = $state(0);
	predictedMs = $state(0);
	totalTokens = $state(0);
	generationStartTokens = $state(0);
	generationEndTokens = $state(0);
	tokenizeTimeout: ReturnType<typeof setTimeout> | undefined;
	promptProgress = $state<ChatMessagePromptProgress | null>(null);

	error = $state<{
		message: string;
		type: 'timeout' | 'server';
		contextInfo?: { n_prompt_tokens: number; n_ctx: number };
	} | null>(null);

	previousContent = $state<string | null>(null);
	undoneContent = $state<string | null>(null);

	async generate(model?: string) {
		if (this.isGenerating) return;

		this.previousContent = this.content;
		this.undoneContent = null;
		this.isGenerating = true;
		this.abortController = new AbortController();
		this.error = null;

		// Reset stats
		this.cacheTokens = 0;
		this.promptTokens = 0;
		this.promptMs = 0;
		this.predictedTokens = 0;
		this.predictedMs = 0;
		this.promptProgress = null;

		// Save number of tokens before generation
		this.generationStartTokens = this.totalTokens;

		try {
			const currentConfig = config();
			const callbacks = {
				onChunk: (chunk: string) => {
					this.content += chunk;
				},
				onTimings: (timings?: ChatMessageTimings, promptProgress?: ChatMessagePromptProgress) => {
					if (timings) {
						if (timings.cache_n) this.cacheTokens = timings.cache_n;
						if (timings.prompt_n) this.promptTokens = timings.prompt_n;
						if (timings.prompt_ms) this.promptMs = timings.prompt_ms;
						if (timings.predicted_n) this.predictedTokens = timings.predicted_n;
						if (timings.predicted_ms) this.predictedMs = timings.predicted_ms;
					}

					if (promptProgress) {
						// Update prompt stats from progress
						const { processed, time_ms } = promptProgress;
						if (processed > 0) this.promptTokens = processed;
						if (time_ms > 0) this.promptMs = time_ms;
						this.promptProgress = promptProgress;
					}

					// Update totalTokens live
					this.totalTokens = this.cacheTokens + this.promptTokens + this.predictedTokens;
				},
				onComplete: () => {
					this.isGenerating = false;
				},
				onError: (error: unknown) => {
					if (error instanceof Error && error.name === 'AbortError') {
						// aborted by user
					} else {
						console.error('Notebook generation error:', error);
						this.error = {
							message: error instanceof Error ? error.message : String(error),
							type: 'server'
						};
					}
					this.isGenerating = false;
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
				type: 'server'
			};
			this.isGenerating = false;
		}
		// Save number of tokens after generation
		this.generationEndTokens = this.totalTokens;
	}

	dismissError() {
		this.error = null;
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
		this.isGenerating = false;
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

	getETASecs(done: number, total: number, elapsedMs: number): number | undefined {
		const elapsedSecs = elapsedMs / 1000;
		const progressETASecs =
			done === 0 || elapsedSecs < 0.5
				? undefined // can be the case for the 0% progress report
				: elapsedSecs * (total / done - 1);
		return progressETASecs;
	}

	getContextTotal(): number | null {
		if (isRouterMode()) {
			const modelContextSize = selectedModelContextSize();

			if (typeof modelContextSize === 'number' && modelContextSize > 0) {
				return modelContextSize;
			}
		} else {
			const propsContextSize = contextSize();

			if (typeof propsContextSize === 'number' && propsContextSize > 0) {
				return propsContextSize;
			}
		}

		return null;
	}

	getProcessingDetails(): string[] {
		const details: string[] = [];

		if (this.promptProgress) {
			const { processed, total, time_ms, cache } = this.promptProgress;
			const actualProcessed = processed - cache;
			const actualTotal = total - cache;

			if (actualProcessed < actualTotal && actualProcessed > 0) {
				const percent = Math.round((actualProcessed / actualTotal) * 100);
				const eta = this.getETASecs(actualProcessed, actualTotal, time_ms);

				if (eta !== undefined) {
					const etaSecs = Math.ceil(eta);
					details.push(`Processing ${percent}% (ETA: ${etaSecs}s)`);
				} else {
					details.push(`Processing ${percent}%`);
				}
			}
		}

		const contextTotal = this.getContextTotal();
		const contextUsed = this.promptTokens + this.cacheTokens + this.predictedTokens;

		if (typeof contextTotal === 'number' && contextUsed >= 0 && contextTotal > 0) {
			const contextPercent = Math.round((contextUsed / contextTotal) * 100);
			details.push(`Context: ${contextUsed}/${contextTotal} (${contextPercent}%)`);
		}

		if (this.predictedTokens > 0) {
			const currentConfig = config();
			const outputTokensMax = currentConfig.max_tokens || -1;
			if (outputTokensMax <= 0) {
				details.push(`Output: ${this.predictedTokens}/∞`);
			} else {
				const outputPercent = Math.round((this.predictedTokens / outputTokensMax) * 100);
				details.push(`Output: ${this.predictedTokens}/${outputTokensMax} (${outputPercent}%)`);
			}
		}

		if (this.predictedTokens > 0 && this.predictedMs > 0) {
			const tokensPerSecond = (this.predictedTokens / this.predictedMs) * 1000;
			details.push(`${tokensPerSecond.toFixed(1)} ${STATS_UNITS.TOKENS_PER_SECOND}`);
		}

		return details;
	}
}

export const notebookStore = new NotebookStore();
