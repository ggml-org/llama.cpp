import { ChatService } from '$lib/services/chat';
import { config } from '$lib/stores/settings.svelte';

export class NotebookStore {
	content = $state('');
	isGenerating = $state(false);
	abortController: AbortController | null = null;

	// Statistics
	promptTokens = $state(0);
	promptMs = $state(0);
	predictedTokens = $state(0);
	predictedMs = $state(0);

	error = $state<{
		message: string;
		type: 'timeout' | 'server';
		contextInfo?: { n_prompt_tokens: number; n_ctx: number };
	} | null>(null);

	async generate(model?: string) {
		if (this.isGenerating) return;

		this.isGenerating = true;
		this.abortController = new AbortController();
		this.error = null;

		// Reset stats
		this.promptTokens = 0;
		this.promptMs = 0;
		this.predictedTokens = 0;
		this.predictedMs = 0;

		try {
			const currentConfig = config();
			await ChatService.sendCompletion(
				this.content,
				{
					...currentConfig,
					model: model ?? currentConfig.model,
					stream: true,
					timings_per_token: true,
					onChunk: (chunk: string) => {
						this.content += chunk;
					},
					onTimings: (timings: ChatMessageTimings, promptProgress: ChatMessagePromptProgress) => {
						if (timings) {
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
	}

	dismissError() {
		this.error = null;
	}

	stop() {
		if (this.abortController) {
			this.abortController.abort();
			this.abortController = null;
		}
		this.isGenerating = false;
	}
}

export const notebookStore = new NotebookStore();
