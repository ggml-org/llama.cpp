import { SvelteSet } from 'svelte/reactivity';
import { ServerModelStatus, ModelModality } from '$lib/enums';
import { ModelsService, PropsService } from '$lib/services';
import { serverStore } from '$lib/stores/server.svelte';
import { TTLCache } from '$lib/utils';
import { MODEL_PROPS_CACHE_TTL_MS, MODEL_PROPS_CACHE_MAX_ENTRIES } from '$lib/constants/cache';

/**
 * modelsStore - Reactive store for model management in both MODEL and ROUTER modes
 *
 * This store manages:
 * - Available models list
 * - Selected model for new conversations
 * - Loaded models tracking (ROUTER mode)
 * - Model usage tracking per conversation
 * - Automatic unloading of unused models
 *
 * **Architecture & Relationships:**
 * - **ModelsService**: Stateless service for model API communication
 * - **PropsService**: Stateless service for props/modalities fetching
 * - **modelsStore** (this class): Reactive store for model state
 * - **conversationsStore**: Tracks which conversations use which models
 *
 * **API Inconsistency Workaround:**
 * In MODEL mode, `/props` returns modalities for the single model.
 * In ROUTER mode, `/props` has no modalities - must use `/props?model=<id>` per model.
 * This store normalizes this behavior so consumers don't need to know the server mode.
 *
 * **Key Features:**
 * - **MODEL mode**: Single model, always loaded
 * - **ROUTER mode**: Multi-model with load/unload capability
 * - **Auto-unload**: Automatically unloads models not used by any conversation
 * - **Lazy loading**: ensureModelLoaded() loads models on demand
 */
class ModelsStore {
	/**
	 *
	 *
	 * State
	 *
	 *
	 */

	models = $state<ModelOption[]>([]);
	routerModels = $state<ApiModelDataEntry[]>([]);
	loading = $state(false);
	updating = $state(false);
	error = $state<string | null>(null);
	selectedModelId = $state<string | null>(null);
	selectedModelName = $state<string | null>(null);

	private modelUsage = $state<Map<string, SvelteSet<string>>>(new Map());
	private modelLoadingStates = $state<Map<string, boolean>>(new Map());

	/**
	 * Model-specific props cache with TTL
	 * Key: modelId, Value: props data including modalities
	 * TTL: 10 minutes - props don't change frequently
	 */
	private modelPropsCache = new TTLCache<string, ApiLlamaCppServerProps>({
		ttlMs: MODEL_PROPS_CACHE_TTL_MS,
		maxEntries: MODEL_PROPS_CACHE_MAX_ENTRIES
	});
	private modelPropsFetching = $state<Set<string>>(new Set());

	/**
	 * Version counter for props cache - used to trigger reactivity when props are updated
	 */
	propsCacheVersion = $state(0);

	/**
	 *
	 *
	 * Computed Getters
	 *
	 *
	 */

	get selectedModel(): ModelOption | null {
		if (!this.selectedModelId) return null;
		return this.models.find((model) => model.id === this.selectedModelId) ?? null;
	}

	get loadedModelIds(): string[] {
		return this.routerModels
			.filter((m) => m.status.value === ServerModelStatus.LOADED)
			.map((m) => m.id);
	}

	get loadingModelIds(): string[] {
		return Array.from(this.modelLoadingStates.entries())
			.filter(([, loading]) => loading)
			.map(([id]) => id);
	}

	/**
	 * Get model name in MODEL mode (single model).
	 * Extracts from model_path or model_alias from server props.
	 * In ROUTER mode, returns null (model is per-conversation).
	 */
	get singleModelName(): string | null {
		if (serverStore.isRouterMode) return null;

		const props = serverStore.props;
		if (props?.model_alias) return props.model_alias;
		if (!props?.model_path) return null;

		return props.model_path.split(/(\\|\/)/).pop() || null;
	}

	/**
	 *
	 *
	 * Modalities
	 *
	 *
	 */

	/**
	 * Get modalities for a specific model
	 * Returns cached modalities from model props
	 */
	getModelModalities(modelId: string): ModelModalities | null {
		const model = this.models.find((m) => m.model === modelId || m.id === modelId);
		if (model?.modalities) {
			return model.modalities;
		}

		const props = this.modelPropsCache.get(modelId);
		if (props?.modalities) {
			return {
				vision: props.modalities.vision ?? false,
				audio: props.modalities.audio ?? false
			};
		}

		return null;
	}

	/**
	 * Check if a model supports vision modality
	 */
	modelSupportsVision(modelId: string): boolean {
		return this.getModelModalities(modelId)?.vision ?? false;
	}

	/**
	 * Check if a model supports audio modality
	 */
	modelSupportsAudio(modelId: string): boolean {
		return this.getModelModalities(modelId)?.audio ?? false;
	}

	/**
	 * Get model modalities as an array of ModelModality enum values
	 */
	getModelModalitiesArray(modelId: string): ModelModality[] {
		const modalities = this.getModelModalities(modelId);
		if (!modalities) return [];

		const result: ModelModality[] = [];

		if (modalities.vision) result.push(ModelModality.VISION);
		if (modalities.audio) result.push(ModelModality.AUDIO);

		return result;
	}

	/**
	 * Get props for a specific model (from cache)
	 */
	getModelProps(modelId: string): ApiLlamaCppServerProps | null {
		return this.modelPropsCache.get(modelId);
	}

	/**
	 * Get context size (n_ctx) for a specific model from cached props
	 */
	getModelContextSize(modelId: string): number | null {
		const props = this.getModelProps(modelId);
		const nCtx = props?.default_generation_settings?.n_ctx;

		return typeof nCtx === 'number' ? nCtx : null;
	}

	/**
	 * Get context size for the currently selected model or null if no model is selected
	 */
	get selectedModelContextSize(): number | null {
		if (!this.selectedModelName) return null;
		return this.getModelContextSize(this.selectedModelName);
	}

	/**
	 * Check if props are being fetched for a model
	 */
	isModelPropsFetching(modelId: string): boolean {
		return this.modelPropsFetching.has(modelId);
	}

	/**
	 *
	 *
	 * Status Queries
	 *
	 *
	 */

	isModelLoaded(modelId: string): boolean {
		const model = this.routerModels.find((m) => m.id === modelId);
		return model?.status.value === ServerModelStatus.LOADED || false;
	}

	isModelOperationInProgress(modelId: string): boolean {
		return this.modelLoadingStates.get(modelId) ?? false;
	}

	getModelStatus(modelId: string): ServerModelStatus | null {
		const model = this.routerModels.find((m) => m.id === modelId);
		return model?.status.value ?? null;
	}

	getModelUsage(modelId: string): SvelteSet<string> {
		return this.modelUsage.get(modelId) ?? new SvelteSet<string>();
	}

	isModelInUse(modelId: string): boolean {
		const usage = this.modelUsage.get(modelId);
		return usage !== undefined && usage.size > 0;
	}

	/**
	 *
	 *
	 * Data Fetching
	 *
	 *
	 */

	/**
	 * Fetch list of models from server and detect server role
	 * Also fetches modalities for MODEL mode (single model)
	 */
	async fetch(force = false): Promise<void> {
		if (this.loading) return;
		if (this.models.length > 0 && !force) return;

		this.loading = true;
		this.error = null;

		try {
			if (!serverStore.props) {
				await serverStore.fetch();
			}

			const response = await ModelsService.list();

			const models: ModelOption[] = response.data.map((item: ApiModelDataEntry, index: number) => {
				const details = response.models?.[index];
				const rawCapabilities = Array.isArray(details?.capabilities) ? details?.capabilities : [];
				const displayNameSource =
					details?.name && details.name.trim().length > 0 ? details.name : item.id;
				const displayName = this.toDisplayName(displayNameSource);

				return {
					id: item.id,
					name: displayName,
					model: details?.model || item.id,
					description: details?.description,
					capabilities: rawCapabilities.filter((value: unknown): value is string => Boolean(value)),
					details: details?.details,
					meta: item.meta ?? null
				} satisfies ModelOption;
			});

			this.models = models;

			// WORKAROUND: In MODEL mode, /props returns modalities for the single model,
			// but /v1/models doesn't include modalities. We bridge this gap here.
			const serverProps = serverStore.props;
			if (serverStore.isModelMode && this.models.length > 0 && serverProps?.modalities) {
				const modalities: ModelModalities = {
					vision: serverProps.modalities.vision ?? false,
					audio: serverProps.modalities.audio ?? false
				};
				this.modelPropsCache.set(this.models[0].model, serverProps);
				this.models = this.models.map((model, index) =>
					index === 0 ? { ...model, modalities } : model
				);
			}
		} catch (error) {
			this.models = [];
			this.error = error instanceof Error ? error.message : 'Failed to load models';
			throw error;
		} finally {
			this.loading = false;
		}
	}

	/**
	 * Fetch router models with full metadata (ROUTER mode only)
	 * This fetches the /models endpoint which returns status info for each model
	 */
	async fetchRouterModels(): Promise<void> {
		try {
			const response = await ModelsService.listRouter();
			this.routerModels = response.data;
			await this.fetchModalitiesForLoadedModels();
		} catch (error) {
			console.warn('Failed to fetch router models:', error);
			this.routerModels = [];
		}
	}

	/**
	 * Fetch props for a specific model from /props endpoint
	 * Uses caching to avoid redundant requests
	 *
	 * In ROUTER mode, this will only fetch props if the model is loaded,
	 * since unloaded models return 400 from /props endpoint.
	 *
	 * @param modelId - Model identifier to fetch props for
	 * @returns Props data or null if fetch failed or model not loaded
	 */
	async fetchModelProps(modelId: string): Promise<ApiLlamaCppServerProps | null> {
		const cached = this.modelPropsCache.get(modelId);
		if (cached) return cached;

		if (serverStore.isRouterMode && !this.isModelLoaded(modelId)) {
			return null;
		}

		if (this.modelPropsFetching.has(modelId)) return null;

		this.modelPropsFetching.add(modelId);

		try {
			const props = await PropsService.fetchForModel(modelId);
			this.modelPropsCache.set(modelId, props);
			return props;
		} catch (error) {
			console.warn(`Failed to fetch props for model ${modelId}:`, error);
			return null;
		} finally {
			this.modelPropsFetching.delete(modelId);
		}
	}

	/**
	 * Fetch modalities for all loaded models from /props endpoint
	 * This updates the modalities field in models array
	 */
	async fetchModalitiesForLoadedModels(): Promise<void> {
		const loadedModelIds = this.loadedModelIds;
		if (loadedModelIds.length === 0) return;

		const propsPromises = loadedModelIds.map((modelId) => this.fetchModelProps(modelId));

		try {
			const results = await Promise.all(propsPromises);

			// Update models with modalities
			this.models = this.models.map((model) => {
				const modelIndex = loadedModelIds.indexOf(model.model);
				if (modelIndex === -1) return model;

				const props = results[modelIndex];
				if (!props?.modalities) return model;

				const modalities: ModelModalities = {
					vision: props.modalities.vision ?? false,
					audio: props.modalities.audio ?? false
				};

				return { ...model, modalities };
			});

			this.propsCacheVersion++;
		} catch (error) {
			console.warn('Failed to fetch modalities for loaded models:', error);
		}
	}

	/**
	 * Update modalities for a specific model
	 * Called when a model is loaded or when we need fresh modality data
	 */
	async updateModelModalities(modelId: string): Promise<void> {
		try {
			const props = await this.fetchModelProps(modelId);
			if (!props?.modalities) return;

			const modalities: ModelModalities = {
				vision: props.modalities.vision ?? false,
				audio: props.modalities.audio ?? false
			};

			this.models = this.models.map((model) =>
				model.model === modelId ? { ...model, modalities } : model
			);

			this.propsCacheVersion++;
		} catch (error) {
			console.warn(`Failed to update modalities for model ${modelId}:`, error);
		}
	}

	/**
	 *
	 *
	 * Model Selection
	 *
	 *
	 */

	/**
	 * Select a model for new conversations
	 */
	async selectModelById(modelId: string): Promise<void> {
		if (!modelId || this.updating) return;
		if (this.selectedModelId === modelId) return;

		const option = this.models.find((model) => model.id === modelId);
		if (!option) throw new Error('Selected model is not available');

		this.updating = true;
		this.error = null;

		try {
			this.selectedModelId = option.id;
			this.selectedModelName = option.model;
		} finally {
			this.updating = false;
		}
	}

	/**
	 * Select a model by its model name (used for syncing with conversation model)
	 * @param modelName - Model name to select (e.g., "unsloth/gemma-3-12b-it-GGUF:latest")
	 */
	selectModelByName(modelName: string): void {
		const option = this.models.find((model) => model.model === modelName);
		if (option) {
			this.selectedModelId = option.id;
			this.selectedModelName = option.model;
		}
	}

	clearSelection(): void {
		this.selectedModelId = null;
		this.selectedModelName = null;
	}

	findModelByName(modelName: string): ModelOption | null {
		return this.models.find((model) => model.model === modelName) ?? null;
	}

	findModelById(modelId: string): ModelOption | null {
		return this.models.find((model) => model.id === modelId) ?? null;
	}

	hasModel(modelName: string): boolean {
		return this.models.some((model) => model.model === modelName);
	}

	/**
	 *
	 *
	 * Loading/Unloading Models
	 *
	 *
	 */

	/**
	 * WORKAROUND: Polling for model status after load/unload operations.
	 *
	 * Currently, the `/models/load` and `/models/unload` endpoints return success
	 * before the operation actually completes on the server. This means an immediate
	 * request to `/models` returns stale status (e.g., "loading" after load request,
	 * "loaded" after unload request).
	 *
	 * TODO: Remove this polling once llama-server properly waits for the operation
	 * to complete before returning success from `/load` and `/unload` endpoints.
	 * At that point, a single `fetchRouterModels()` call after the operation will
	 * be sufficient to get the correct status.
	 */

	/** Polling interval in ms for checking model status */
	private static readonly STATUS_POLL_INTERVAL = 500;
	/** Maximum polling attempts before giving up */
	private static readonly STATUS_POLL_MAX_ATTEMPTS = 60; // 30 seconds max

	/**
	 * Poll for expected model status after load/unload operation.
	 * Keeps polling until the model reaches the expected status or max attempts reached.
	 *
	 * @param modelId - Model identifier to check
	 * @param expectedStatus - Expected status to wait for
	 * @returns Promise that resolves when expected status is reached
	 */
	private async pollForModelStatus(
		modelId: string,
		expectedStatus: ServerModelStatus
	): Promise<void> {
		for (let attempt = 0; attempt < ModelsStore.STATUS_POLL_MAX_ATTEMPTS; attempt++) {
			await this.fetchRouterModels();

			const currentStatus = this.getModelStatus(modelId);
			if (currentStatus === expectedStatus) {
				return;
			}

			await new Promise((resolve) => setTimeout(resolve, ModelsStore.STATUS_POLL_INTERVAL));
		}

		console.warn(
			`Model ${modelId} did not reach expected status ${expectedStatus} after ${ModelsStore.STATUS_POLL_MAX_ATTEMPTS} attempts`
		);
	}

	/**
	 * Load a model (ROUTER mode)
	 * @param modelId - Model identifier to load
	 */
	async loadModel(modelId: string): Promise<void> {
		if (this.isModelLoaded(modelId)) {
			return;
		}

		if (this.modelLoadingStates.get(modelId)) return;

		this.modelLoadingStates.set(modelId, true);
		this.error = null;

		try {
			await ModelsService.load(modelId);
			await this.pollForModelStatus(modelId, ServerModelStatus.LOADED);

			await this.updateModelModalities(modelId);
		} catch (error) {
			this.error = error instanceof Error ? error.message : 'Failed to load model';
			throw error;
		} finally {
			this.modelLoadingStates.set(modelId, false);
		}
	}

	/**
	 * Unload a model (ROUTER mode)
	 * @param modelId - Model identifier to unload
	 */
	async unloadModel(modelId: string): Promise<void> {
		if (!this.isModelLoaded(modelId)) {
			return;
		}

		if (this.modelLoadingStates.get(modelId)) return;

		this.modelLoadingStates.set(modelId, true);
		this.error = null;

		try {
			await ModelsService.unload(modelId);

			await this.pollForModelStatus(modelId, ServerModelStatus.UNLOADED);
		} catch (error) {
			this.error = error instanceof Error ? error.message : 'Failed to unload model';
			throw error;
		} finally {
			this.modelLoadingStates.set(modelId, false);
		}
	}

	/**
	 * Ensure a model is loaded before use
	 * @param modelId - Model identifier to ensure is loaded
	 */
	async ensureModelLoaded(modelId: string): Promise<void> {
		if (this.isModelLoaded(modelId)) {
			return;
		}

		await this.loadModel(modelId);
	}

	/**
	 *
	 *
	 * Utilities
	 *
	 *
	 */

	private toDisplayName(id: string): string {
		const segments = id.split(/\\|\//);
		const candidate = segments.pop();

		return candidate && candidate.trim().length > 0 ? candidate : id;
	}

	clear(): void {
		this.models = [];
		this.routerModels = [];
		this.loading = false;
		this.updating = false;
		this.error = null;
		this.selectedModelId = null;
		this.selectedModelName = null;
		this.modelUsage.clear();
		this.modelLoadingStates.clear();
		this.modelPropsCache.clear();
		this.modelPropsFetching.clear();
	}

	/**
	 * Prune expired entries from caches.
	 * Call periodically for proactive memory cleanup.
	 */
	pruneExpiredCache(): number {
		return this.modelPropsCache.prune();
	}
}

export const modelsStore = new ModelsStore();

export const modelOptions = () => modelsStore.models;
export const routerModels = () => modelsStore.routerModels;
export const modelsLoading = () => modelsStore.loading;
export const modelsUpdating = () => modelsStore.updating;
export const modelsError = () => modelsStore.error;
export const selectedModelId = () => modelsStore.selectedModelId;
export const selectedModelName = () => modelsStore.selectedModelName;
export const selectedModelOption = () => modelsStore.selectedModel;
export const loadedModelIds = () => modelsStore.loadedModelIds;
export const loadingModelIds = () => modelsStore.loadingModelIds;
export const propsCacheVersion = () => modelsStore.propsCacheVersion;
export const singleModelName = () => modelsStore.singleModelName;
export const selectedModelContextSize = () => modelsStore.selectedModelContextSize;
