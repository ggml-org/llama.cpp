import { CompletionService } from '$lib/services/completion.service';
import { config } from '$lib/stores/settings.svelte';
import { tokenize } from '$lib/services/tokenize.service';

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

		// Save number of tokens before generation
		this.generationStartTokens = this.totalTokens;

		try {
			const currentConfig = config();
			const callbacks = {
				onChunk: (chunk: string) => {
					this.content += chunk;
				},
				onTimings: (timings: ChatMessageTimings, promptProgress: ChatMessagePromptProgress) => {
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
}

export const notebookStore = new NotebookStore();
