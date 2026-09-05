/**
 * modelsDiscoverStore - Models Discover browse state
 *
 * Owns the HuggingFace GGUF model list shown in the discover sidebar
 * (DialogModelsDiscover). By default the list is the curated catalog set;
 * search replaces it with matches across all of HuggingFace. Both paths fetch
 * the same fields, so a row renders the same badges and sizes either way.
 */

import { isAuxSidecar } from '$lib/constants';
import { HuggingFaceService } from '$lib/services';
import type {
	HfCatalogBuild,
	HfCatalogEntry,
	HfModelInfo,
	HfModelSibling
} from '$lib/types/huggingface';
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
	 * Fetch the default list from the llama.app catalog, one repo per catalog
	 * size in display order (newest family first). Every repo is fetched by ID
	 * with its file tree, so the rows carry chat-template capabilities, context
	 * length and a real size range - the same data a search result gets.
	 * No-op when already loaded or in flight.
	 */
	async fetch(): Promise<void> {
		if (this.loading || this.fetched) return;

		this.loading = true;
		this.error = null;

		try {
			const catalog = await HuggingFaceService.getCatalog();

			this.catalog = catalog;

			const builds = this.catalogBuilds();
			const fetched = await Promise.all(
				builds.map(async (build) => {
					const [info, tree] = await Promise.all([
						HuggingFaceService.getDetails(build.repo),
						HuggingFaceService.getTree(build.repo)
					]);

					return { build, info, tree };
				})
			);
			const models: HfModelInfo[] = [];

			for (const { build, info, tree } of fetched) {
				if (!info) continue;

				this.catalogSizeRanges.set(build.repo, this.sizeRangeFor(build, tree));

				// the catalog repo id, not the HF response `id`, drives selection
				models.push({ ...info, id: build.repo, modelId: build.repo } as HfModelInfo);
			}

			this.defaultModels = models;
			this.models = models;
			this.fetched = true;
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
	 * Size range for a repo not measured yet - a search result, or a catalog
	 * repo whose tree came back empty. Fetches the file tree once per repo and
	 * caches it, so remounting a row (scrolling, searching back) is free.
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
	 * Bytes of every quant the catalog lists under this repo, parsed from the
	 * `size` strings when a build carries no `sizeBytes`. Never empty: an
	 * unparsable entry contributes the build's own size.
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
	 * One build per catalog size, newest family first (by release date).
	 * Prefers the official ggml-org repo, falling back to the first build so
	 * families published only by other orgs (mistralai, unsloth) still show up.
	 * Returns an empty array when the catalog is empty.
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
	 * Size range of one catalog build: every quant in the repo's file tree, plus
	 * the draft sidecars those files carry (mtp, dflash, ...) so the downloaded
	 * model fits within the range. Falls back to the catalog `size` / `sizeBytes`
	 * strings when the tree yielded nothing (partial fetch, sharded-only repo).
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

	/**
	 * Size range across the main-model quants of a file tree, draft sidecars
	 * excluded (they only widen the range when a row advertises them).
	 */
	private sizeRangeOfMainQuants(tree: HfModelSibling[]): ModelsDiscoverSizeRange | undefined {
		const sizes = this.quantSizesOf(tree);

		if (sizes.length === 0) return undefined;

		return { max: Math.max(...sizes), min: Math.min(...sizes) };
	}
}

export const modelsDiscoverStore = new ModelsDiscoverStore();
