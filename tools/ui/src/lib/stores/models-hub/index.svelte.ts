/**
 * modelsHubStore - Model Hub browse state
 *
 * Owns the HuggingFace GGUF model list shown in the hub sidebar
 * (DialogModelsHub). The hub has no "nothing selected" screen: it always opens
 * a model, so `firstModel` drives the initial selection. By default the list
 * shows the official ggml-org GGUF models sorted by popularity (downloads);
 * search replaces the list with matching models across all of HuggingFace.
 * Detail data is loaded by ModelsHubModelDetails, not here.
 */

import { HuggingFaceService } from '$lib/services';
import type { HfModelInfo } from '$lib/types/huggingface';

class ModelsHubStore {
	/** Org whose models the sidebar lists by default. */
	private static readonly DEFAULT_AUTHOR = 'ggml-org';

	models = $state<HfModelInfo[]>([]);
	loading = $state(false);
	error = $state<string | null>(null);

	/** First model in the list - the hub auto-opens this one. */
	firstModel = $derived(this.models[0] ?? null);

	private fetched = false;
	private defaultModels: HfModelInfo[] = [];
	private searchRequestId = 0;

	/**
	 * Fetch the default list: official ggml-org GGUF models sorted by popularity
	 * (downloads). No-op when already loaded or in flight.
	 */
	async fetch(): Promise<void> {
		if (this.loading || this.fetched) return;

		this.loading = true;
		this.error = null;

		try {
			const results = await HuggingFaceService.search({
				author: ModelsHubStore.DEFAULT_AUTHOR,
				sort: 'downloads',
				limit: 50,
				full: true
			});

			// `models-moved` is a redirect placeholder, not a browsable model
			this.defaultModels = results.filter((m) => m.id !== 'ggml-org/models-moved');
			this.models = this.defaultModels;
			this.fetched = true;
		} catch (err) {
			this.error = err instanceof Error ? err.message : 'Failed to fetch models';
		} finally {
			this.loading = false;
		}
	}

	/**
	 * Replace the list with GGUF search results. An empty query restores the
	 * default list. The current list stays visible while a search is in
	 * flight; stale responses are dropped when a newer search starts.
	 */
	async search(query: string): Promise<void> {
		const trimmed = query.trim();

		this.searchRequestId++;

		if (!trimmed) {
			this.models = this.defaultModels;
			this.error = null;
			return;
		}

		const requestId = this.searchRequestId;

		try {
			const results = await HuggingFaceService.searchByQuery(trimmed, { limit: 50, full: true });

			if (requestId === this.searchRequestId) {
				this.models = results;
				this.error = null;
			}
		} catch (err) {
			if (requestId === this.searchRequestId) {
				this.error = err instanceof Error ? err.message : 'Search failed';
			}
		}
	}
}

export const modelsHubStore = new ModelsHubStore();
