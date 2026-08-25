/**
 * modelsHubStore - Model Hub browse state
 *
 * Owns the HuggingFace GGUF model list shown in the hub sidebar
 * (DialogModelsDiscover). The hub has no "nothing selected" screen: it always opens
 * a model, so `firstModel` drives the initial selection. By default the list
 * shows a curated set of official ggml-org GGUF models in a fixed display order;
 * search replaces the list with matching models across all of HuggingFace.
 * Detail data is loaded by ModelsDiscoverDetails, not here.
 */

import { HuggingFaceService } from '$lib/services';
import type { HfCatalogEntry, HfModelInfo } from '$lib/types/huggingface';

class ModelsHubStore {
	error = $state<string | null>(null);
	models = $state<HfModelInfo[]>([]);
	/** First model in the list - the hub auto-opens this one. */
	firstModel = $derived(this.models[0] ?? null);

	loading = $state(false);

	private catalog: HfCatalogEntry[] = [];
	private defaultModels: HfModelInfo[] = [];
	private fetched = false;
	private searchRequestId = 0;

	/**
	 * Catalog family description for a repo id, or undefined when the repo is
	 * not part of the catalog (e.g. a search result outside the curated list).
	 */
	descriptionFor(modelId: string): string | undefined {
		return this.catalog.find((entry) =>
			entry.sizes.some((size) => size.builds.some((build) => build.repo === modelId))
		)?.description;
	}

	/**
	 * Fetch the default list from the llama.app catalog, flattened to a flat
	 * list of ggml-org repo ids in catalog order (one per size). Each repo is
	 * fetched directly by ID, so the list is independent of download ranking.
	 * No-op when already loaded or in flight.
	 */
	async fetch(): Promise<void> {
		if (this.loading || this.fetched) return;

		this.loading = true;
		this.error = null;

		try {
			const catalog = await HuggingFaceService.getCatalog();

			this.catalog = catalog;
			const ids = this.catalogModelIds(catalog);

			// getDetails returns full metadata (downloads, likes, lastModified,
			// siblings, tags, gguf) for a single model.
			this.defaultModels = (
				await Promise.all(ids.map((id) => HuggingFaceService.getDetails(id)))
			).filter((m): m is HfModelInfo => m !== null);
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
			const results = await HuggingFaceService.searchByQuery(trimmed, { full: true, limit: 50 });

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

	/**
	 * Min/max GGUF file size (bytes) across the available quants for a repo,
	 * or undefined when the repo is not part of the catalog.
	 */
	sizeRangeFor(modelId: string): { min: number; max: number } | undefined {
		for (const entry of this.catalog) {
			for (const size of entry.sizes) {
				const builds = size.builds.filter((b) => b.repo === modelId);

				if (builds.length === 0) continue;

				const bytes = builds.map((b) => b.sizeBytes);

				return { max: Math.max(...bytes), min: Math.min(...bytes) };
			}
		}

		return undefined;
	}

	/**
	 * Flatten the catalog to a flat list of ggml-org repo ids, newest family
	 * first (by release date). Returns an empty array when the catalog is empty.
	 */
	private catalogModelIds(catalog: HfCatalogEntry[]): string[] {
		return [...catalog]
			.sort((a, b) => b.released.localeCompare(a.released))
			.flatMap((entry) =>
				entry.sizes.flatMap((size) => {
					const build = size.builds.find((b) => b.repo.startsWith('ggml-org/'));

					return build ? [build.repo] : [];
				})
			);
	}
}

export const modelsHubStore = new ModelsHubStore();
