/**
 * ModelPropsManager - Per-model props cache, modalities and thinking detection
 *
 * Owns the /props?model=<id> cache with TTL, the modality views over it,
 * and chat-template thinking detection. Created and owned by modelsStore;
 * the host owns the model lists that fetched modalities are mirrored onto.
 *
 * **API Inconsistency Workaround:**
 * In MODEL mode, `/props` returns modalities for the single model.
 * In ROUTER mode, `/props` has no modalities - must use `/props?model=<id>` per model.
 */

import { MODEL_PROPS_CACHE } from '$lib/constants';
import { FileTypeCategory, ModelModality } from '$lib/enums';
import { PropsService } from '$lib/services/props.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { serverStore } from '$lib/stores/server.svelte';
// deep imports, not the '$lib/utils' barrel: it re-exports modules that reach back
// into the stores, and going through it here would read a half-built module
import { TTLCache } from '$lib/utils/cache-ttl';
import { detectThinkingSupport } from '$lib/utils/chat-template-thinking-detector';
import { SvelteSet } from 'svelte/reactivity';

/**
 * The slice of modelsStore the manager reads. Kept narrow on purpose so it
 * cannot reach around the host's full surface; modelsStore implements this
 * structurally.
 */
export interface ModelPropsHost {
	/** Model rows the manager mirrors fetched modalities onto. */
	models: ModelOption[];
	readonly selectedModelName: string | null;
	readonly loadedModelIds: string[];
	isModelLoaded(modelId: string): boolean;
}

export class ModelPropsManager {
	/** Version counter for the cache - bumped on writes so $derived consumers recompute. */
	cacheVersion = $state(0);
	/**
	 * Model-specific props cache with TTL.
	 * Key: modelId, Value: props data including modalities.
	 */
	private cache = new TTLCache<string, ApiLlamaCppServerProps>({
		maxEntries: MODEL_PROPS_CACHE.MAX_ENTRIES,
		ttlMs: MODEL_PROPS_CACHE.TTL_MS
	});
	private fetching = new SvelteSet<string>();

	/**
	 * Whether the selected model's chat template supports thinking/reasoning.
	 * Uses heuristic detection on the model's chat_template from /props.
	 *
	 * - MODEL mode: the global /props already describes the single loaded model,
	 *   so its chat_template is used directly and no per-model cache is involved
	 * - ROUTER mode: fetches /props?model=<id> for the selected model (cached),
	 *   triggering an async fetch if not yet cached
	 */
	get supportsThinking(): boolean {
		if (!serverStore.isRouterMode) {
			return detectThinkingSupport(serverStore.props?.chat_template ?? '');
		}

		const modelId = this.host.selectedModelName;

		if (!modelId) return false;

		if (!this.cache.get(modelId)) {
			this.fetchModelProps(modelId);
		}

		const props = this.getModelProps(modelId);

		return detectThinkingSupport(props?.chat_template ?? '');
	}

	/** Map the router modalities, the only source available while a model is not loaded. */
	buildArchitectureModalities(
		architecture: ApiModelDataEntry['architecture']
	): ModelModalities | undefined {
		if (!architecture) return undefined;

		const inputs = architecture.input_modalities;

		return {
			audio: inputs.includes(FileTypeCategory.AUDIO),
			video: inputs.includes(FileTypeCategory.VIDEO),
			vision: inputs.includes(FileTypeCategory.IMAGE)
		};
	}

	/**
	 * Check if a specific model supports thinking.
	 * In MODEL mode the global /props describes the single loaded model.
	 * In ROUTER mode, fetches model props if not cached.
	 */
	checkModelSupportsThinking(modelId: string): boolean {
		if (!serverStore.isRouterMode) {
			return detectThinkingSupport(serverStore.props?.chat_template ?? '');
		}

		if (!modelId) return false;

		if (!this.cache.get(modelId)) {
			this.fetchModelProps(modelId);
		}

		const props = this.getModelProps(modelId);

		return detectThinkingSupport(props?.chat_template ?? '');
	}

	constructor(private host: ModelPropsHost) {}

	/** Fetch modalities for all loaded models from /props endpoint. */
	async fetchModalitiesForLoadedModels(): Promise<void> {
		const loadedModelIds = this.host.loadedModelIds;

		if (loadedModelIds.length === 0) return;

		const propsPromises = loadedModelIds.map((modelId) => this.fetchModelProps(modelId));

		try {
			const results = await Promise.all(propsPromises);

			this.host.models = this.host.models.map((model) => {
				const modelIndex = loadedModelIds.indexOf(model.model);

				if (modelIndex === -1) return model;

				const props = results[modelIndex];

				if (!props?.modalities) return model;

				return { ...model, modalities: this.buildModalities(props.modalities) };
			});

			this.cacheVersion++;
		} catch (error) {
			console.warn('Failed to fetch modalities for loaded models:', error);
		}
	}

	/**
	 * Fetch props for a specific model from /props endpoint.
	 * Uses caching to avoid redundant requests.
	 *
	 * In ROUTER mode, this only fetches props if the model is loaded,
	 * since unloaded models return 400 from /props endpoint.
	 *
	 * @param modelId - Model identifier to fetch props for
	 * @returns Props data or null if fetch failed or model not loaded
	 */
	async fetchModelProps(modelId: string): Promise<ApiLlamaCppServerProps | null> {
		const cached = this.cache.get(modelId);

		if (cached) return cached;

		if (serverStore.isRouterMode && !this.host.isModelLoaded(modelId)) {
			return null;
		}

		if (this.fetching.has(modelId)) return null;

		this.fetching.add(modelId);

		try {
			const props = await PropsService.fetchForModel(modelId);

			this.cache.set(modelId, props);
			this.cacheVersion++;

			return props;
		} catch (error) {
			console.warn(`Failed to fetch props for model ${modelId}:`, error);

			return null;
		} finally {
			this.fetching.delete(modelId);
		}
	}

	getModelContextSize(modelId: string): number | null {
		const props = this.getModelProps(modelId);
		const nCtx = props?.default_generation_settings?.n_ctx;

		return typeof nCtx === 'number' ? nCtx : null;
	}

	getModelModalities(modelId: string): ModelModalities | null {
		if (!serverStore.isRouterMode && serverStore.props?.modalities) {
			return this.buildModalities(serverStore.props.modalities);
		}

		const model = this.host.models.find((m) => m.model === modelId || m.id === modelId);

		if (model?.modalities) {
			return model.modalities;
		}

		const props = this.cache.get(modelId);

		if (props?.modalities) {
			return this.buildModalities(props.modalities);
		}

		return null;
	}

	getModelModalitiesArray(modelId: string): ModelModality[] {
		const modalities = this.getModelModalities(modelId);

		if (!modalities) return [];

		const result: ModelModality[] = [];

		if (modalities.vision) result.push(ModelModality.VISION);

		if (modalities.audio) result.push(ModelModality.AUDIO);

		if (modalities.video) result.push(ModelModality.VIDEO);

		return result;
	}

	getModelProps(modelId: string): ApiLlamaCppServerProps | null {
		return this.cache.get(modelId);
	}

	isModelPropsFetching(modelId: string): boolean {
		return this.fetching.has(modelId);
	}

	modelSupportsAudio(modelId: string): boolean {
		return this.getModelModalities(modelId)?.audio ?? false;
	}

	modelSupportsVideo(modelId: string): boolean {
		return this.getModelModalities(modelId)?.video ?? false;
	}

	modelSupportsVision(modelId: string): boolean {
		return this.getModelModalities(modelId)?.vision ?? false;
	}

	/**
	 * Update modalities for a specific model.
	 * Called when a model is loaded or when we need fresh modality data.
	 */
	async updateModelModalities(modelId: string): Promise<void> {
		const props = await this.fetchModelProps(modelId);

		if (!props?.modalities) return;

		this.host.models = this.host.models.map((model) =>
			model.model === modelId
				? { ...model, modalities: this.buildModalities(props.modalities!) }
				: model
		);

		this.cacheVersion++;
	}

	private buildModalities(
		modalities: NonNullable<ApiLlamaCppServerProps['modalities']>
	): ModelModalities {
		return {
			audio: modalities.audio ?? false,
			video: modalities.video ?? false,
			vision: modalities.vision ?? false
		};
	}
}
