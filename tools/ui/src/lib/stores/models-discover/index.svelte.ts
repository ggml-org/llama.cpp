/**
 * modelsDiscoverStore - Models Discover browse state
 *
 * Owns the HuggingFace GGUF list shown in the discover sidebar. The default list is
 * the curated catalog set; a search replaces it with matches across all of HuggingFace.
 * Both paths fetch the same fields, so a row renders the same either way.
 */

import { MODELS_DISCOVER_CATALOG_BATCH } from '$lib/constants';
import { HuggingFaceService } from '$lib/services';
import type {
	HfCatalogBuild,
	HfCatalogEntry,
	HfModelInfo,
	HfModelSibling
} from '$lib/types/huggingface';
import { isAuxSidecar } from '$lib/utils';
import { SvelteMap } from 'svelte/reactivity';

/** Min/max GGUF file size (bytes) across the quants of one repo. */
export interface ModelsDiscoverSizeRange {
	max: number;
	min: number;
}

class ModelsDiscoverStore {
	error = $state<string | null>(null);
	models = $state<HfModelInfo[]>([]);
	/** First model in the list - discover auto-opens this one. */
	firstModel = $derived(this.models[0] ?? null);

	loading = $state(false);
	/** True while a search is in flight; only the newest request owns this flag. */
	searching = $state(false);

	private catalog: HfCatalogEntry[] = [];
	/** Repo id -> size range, for catalog rows and lazily measured search rows. */
	private catalogSizeRanges = new SvelteMap<string, ModelsDiscoverSizeRange>();
	private defaultModels: HfModelInfo[] = [];
	private fetched = false;
	private searchRequestId = 0;
	/** In-flight `sizeRange()` lookups, keyed by repo id. */
	private sizeRangePending = new Map<string, Promise<ModelsDiscoverSizeRange | undefined>>();

	/**
	 * Cached size range for a repo, without measuring: the synchronous part of
	 * `sizeRange()`, for rendering a row before its measurement resolves.
	 */
	cachedSizeRangeFor(modelId: string): ModelsDiscoverSizeRange | undefined {
		return this.catalogSizeRanges.get(modelId);
	}

	/** Catalog family description; undefined for a repo outside the curated catalog. */
	descriptionFor(modelId: string): string | undefined {
		return this.catalog.find((entry) =>
			entry.sizes.some((size) => size.builds.some((build) => build.repo === modelId))
		)?.description;
	}

	/**
	 * Fill the list from the llama.app catalog, one repo per catalog size. Every repo
	 * is fetched with its file tree, so a catalog row carries the same data a search
	 * result does. No-op when already loaded or in flight.
	 */
	async fetch(): Promise<void> {
		if (this.loading || this.fetched) return;

		this.loading = true;
		this.error = null;

		try {
			const catalog = await HuggingFaceService.getCatalog();

			this.catalog = catalog;

			const builds = this.catalogBuilds();
			const failedBuilds: HfCatalogBuild[] = [];

			this.defaultModels = [];

			// fetch in small batches: not every repo hits the HF API at once,
			// and the rows land as they arrive instead of all at the end
			for (let i = 0; i < builds.length; i += MODELS_DISCOVER_CATALOG_BATCH) {
				const batch = await Promise.all(
					builds.slice(i, i + MODELS_DISCOVER_CATALOG_BATCH).map(async (build) => {
						const [info, tree] = await Promise.all([
							HuggingFaceService.getDetails(build.repo),
							HuggingFaceService.getTree(build.repo)
						]);

						return { build, info, tree };
					})
				);

				for (const { build, info, tree } of batch) {
					if (!info) {
						failedBuilds.push(build);

						continue;
					}

					this.catalogSizeRanges.set(build.repo, this.sizeRangeFor(build, tree));

					// the catalog repo id, not the HF response `id`, drives selection
					this.defaultModels.push({ ...info, id: build.repo, modelId: build.repo });
				}

				// show what has landed, unless a search owns the list right now
				if (!this.searching) {
					this.models = [...this.defaultModels];
				}
			}

			// stay incomplete when some repos failed, so the next mount retries
			// them; succeeded repos are served from the service cache for free
			this.fetched = failedBuilds.length === 0;
		} catch (err) {
			this.error = err instanceof Error ? err.message : 'Failed to fetch models';
		} finally {
			this.loading = false;
		}
	}

	/** Replace the list with search results; an empty query restores the default list. */
	async search(query: string): Promise<void> {
		const trimmed = query.trim();
		const requestId = ++this.searchRequestId;

		if (!trimmed) {
			this.searching = false;
			this.models = this.defaultModels;
			this.error = null;

			return;
		}

		this.searching = true;

		try {
			const results = await HuggingFaceService.searchByQuery(trimmed, { limit: 50 });

			if (requestId !== this.searchRequestId) return;

			this.models = results;
			this.error = null;
		} catch (err) {
			if (requestId !== this.searchRequestId) return;

			this.error = err instanceof Error ? err.message : 'Search failed';
		} finally {
			if (requestId === this.searchRequestId) this.searching = false;
		}
	}

	/**
	 * Size range for a repo not measured yet, fetched once and cached: a search result
	 * never goes through fetch(), and a row can be remounted while scrolling.
	 */
	sizeRange(modelId: string): Promise<ModelsDiscoverSizeRange | undefined> {
		const cached = this.catalogSizeRanges.get(modelId);

		if (cached) return Promise.resolve(cached);

		const pending = this.sizeRangePending.get(modelId);

		if (pending) return pending;

		const request = (async () => {
			const tree = await HuggingFaceService.getTree(modelId);
			const range = this.sizeRangeOfMainQuants(tree);

			if (range) this.catalogSizeRanges.set(modelId, range);

			return range;
		})()
			.catch(() => undefined)
			.finally(() => this.sizeRangePending.delete(modelId));

		this.sizeRangePending.set(modelId, request);

		return request;
	}

	/**
	 * Sizes the catalog lists under this repo. Never empty: an entry with no parsable
	 * size contributes the build's own.
	 */
	private buildSizeBytes(build: HfCatalogBuild): number[] {
		const sizes = this.catalog
			.flatMap((entry) => entry.sizes)
			.flatMap((size) => size.builds.filter((b) => b.repo === build.repo))
			.map((b) => b.sizeBytes ?? HuggingFaceService.parseSizeBytes(b.size))
			.filter((bytes): bytes is number => Boolean(bytes) && bytes > 0);

		return sizes.length > 0 ? sizes : [build.sizeBytes ?? 0];
	}

	/**
	 * One build per catalog size, newest family first. Prefers the official ggml-org
	 * repo, else the first build, so repos of other orgs are covered too.
	 */
	private catalogBuilds(): HfCatalogBuild[] {
		return [...this.catalog]
			.sort((a, b) => b.released.localeCompare(a.released))
			.flatMap((entry) =>
				entry.sizes.flatMap((size) => {
					const build = size.builds.find((b) => b.repo.startsWith('ggml-org/')) ?? size.builds[0];

					return build ? [build] : [];
				})
			);
	}

	/** Byte sizes of every non-sidecar quant file in a tree, shards collapsed. */
	private quantSizesOf(tree: HfModelSibling[]): number[] {
		return HuggingFaceService.collapseGgufShards(
			HuggingFaceService.filterByExtension(tree, '.gguf')
		)
			.filter((f) => {
				const { quant, sidecar } = HuggingFaceService.extractQuantMeta(f.path) ?? {};

				return Boolean(quant) && (sidecar === null || sidecar === undefined);
			})
			.map((f) => f.size ?? 0)
			.filter((size) => size > 0);
	}

	/**
	 * Size range of one catalog build, draft sidecars included so the range covers what
	 * gets downloaded. Falls back to the catalog's own sizes when the tree yielded nothing.
	 */
	private sizeRangeFor(build: HfCatalogBuild, tree: HfModelSibling[]): ModelsDiscoverSizeRange {
		const quantSizes = this.quantSizesOf(tree);

		if (quantSizes.length === 0) {
			const listed = this.buildSizeBytes(build);

			return { max: Math.max(...listed), min: Math.min(...listed) };
		}

		const draftSizes = tree
			.filter((f) => {
				const sidecar = HuggingFaceService.extractQuantMeta(f.path)?.sidecar;

				return sidecar !== null && sidecar !== undefined && !isAuxSidecar(sidecar);
			})
			.map((f) => f.size ?? 0)
			.filter((size) => size > 0);
		const extra = draftSizes.length > 0 ? Math.max(...draftSizes) : 0;

		return {
			max: Math.max(...quantSizes) + extra,
			min: Math.min(...quantSizes) + (draftSizes.length > 0 ? Math.min(...draftSizes) : 0)
		};
	}

	/** Size range across the main-model quants of a tree, draft sidecars excluded. */
	private sizeRangeOfMainQuants(tree: HfModelSibling[]): ModelsDiscoverSizeRange | undefined {
		const sizes = this.quantSizesOf(tree);

		if (sizes.length === 0) return undefined;

		return { max: Math.max(...sizes), min: Math.min(...sizes) };
	}
}

export const modelsDiscoverStore = new ModelsDiscoverStore();
