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

	async generate(model?: string) {
		if (this.isGenerating) return;

		this.isGenerating = true;
		this.abortController = new AbortController();

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
					onChunk: (chunk) => {
						this.content += chunk;
					},
					onTimings: (timings) => {
						if (timings) {
							if (timings.prompt_n) this.promptTokens = timings.prompt_n;
							if (timings.prompt_ms) this.promptMs = timings.prompt_ms;
							if (timings.predicted_n) this.predictedTokens = timings.predicted_n;
							if (timings.predicted_ms) this.predictedMs = timings.predicted_ms;
						}
					},
					onComplete: () => {
						this.isGenerating = false;
					},
					onError: (error) => {
						if (error instanceof Error && error.name === 'AbortError') {
							// aborted by user
						} else {
							console.error('Notebook generation error:', error);
						}
						this.isGenerating = false;
					}
				},
				this.abortController.signal
			);
		} catch (error) {
			console.error('Notebook generation failed:', error);
			this.isGenerating = false;
		}
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
