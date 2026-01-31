import { ChatService } from '$lib/services/chat';
import { config } from '$lib/stores/settings.svelte';

export class NotebookStore {
    content = $state('');
    isGenerating = $state(false);
    abortController: AbortController | null = null;

    async generate(model?: string) {
        if (this.isGenerating) return;

        this.isGenerating = true;
        this.abortController = new AbortController();

        try {
            const currentConfig = config();
            await ChatService.sendCompletion(
                this.content,
                {
                    ...currentConfig,
                    model,
                    stream: true,
                    onChunk: (chunk) => {
                        this.content += chunk;
                    },
                    onComplete: () => {
                        this.isGenerating = false;
                    },
                    onError: (error) => {
                        console.error('Notebook generation error:', error);
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
