import { base } from '$app/paths';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';
import { toast } from 'svelte-sonner';
import { ServerModelStatus, ServerModelsSseEventType, ModelModality } from '$lib/enums';
import { ModelsService } from '$lib/services/models.service';
import { PropsService } from '$lib/services/props.service';
import { serverStore, isRouterMode } from '$lib/stores/server.svelte';
import {
	detectThinkingSupport,
	detectThinkingSupportWithReason
} from '$lib/utils/chat-template-thinking-detector';
import { TTLCache, getAuthHeaders } from '$lib/utils';
import {
	MODEL_PROPS_CACHE_TTL_MS,
	MODEL_PROPS_CACHE_MAX_ENTRIES,
	FAVORITE_MODELS_LOCALSTORAGE_KEY,
	API_MODELS,
	SSE_RECORD_SEPARATOR,
	SSE_LINE_SEPARATOR,
	SSE_DATA_PREFIX
} from '$lib/constants';

import { conversationsStore } from '$lib/stores/conversations.svelte';

/**
 * modelsStore - Reactive store for model management in both MODEL and ROUTER modes.
 *
 * **Architecture & Relationships:**
 * - **ModelsService**: Stateless service for model API communication
 * - **PropsService**: Stateless service for props/modalities fetching
 * - **modelsStore** (this class): Reactive store for model state
 * - **conversationsStore**: Tracks which conversations use which models
 *
 * **API Inconsistency Workaround:**
 * In MODEL mode, `/props` returns modalities for the single model.
 * In ROUTER mode, `/props` has no modalities — must use `/props?model=<id>` per model.
 * This store normalizes this behavior so consumers don't need to know the server mode.
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

	// Dedup concurrent fetch() callers — all awaiters share the same inflight promise.
	// Without this, ?model=<name> URL handler races an in-progress fetch and sees an empty list.
	private inflightFetch: Promise<void> | null = null;

	private modelUsage = $state<Map<string, SvelteSet<string>>>(new Map());
	private modelLoadingStates = new SvelteMap<string, boolean>();

	// /models/sse feed state, the single source of truth for status and load progress
	private statusAbort: AbortController | null = null;
	private statusReaderActive = false;
	private loadProgress = new SvelteMap<string, ModelLoadProgress>();
	private statusWaiters = new Map<
		string,
		{ target: ServerModelStatus; resolve: () => void; reject: (e: Error) => void }
	>();

	favoriteModelIds = $state<Set<string>>(this.loadFavoritesFromStorage());

	/**
	 * Model-specific props cache with TTL.
	 * Key: modelId, Value: props data including modalities.
	 * TTL: 10 minutes — props don't change frequently.
	 */
	private modelPropsCache = new TTLCache<string, ApiLlamaCppServerProps>({
		ttlMs: MODEL_PROPS_CACHE_TTL_MS,
		maxEntries: MODEL_PROPS_CACHE_MAX_ENTRIES
	});
	private modelPropsFetching = $state<Set<string>>(new Set());

	/**
	 * Version counter for props cache — used to trigger reactivity when props are updated.
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
		return this.models.find((m) => m.id === this.selectedModelId) ?? null;
	}

	get loadedModelIds(): string[] {
		return this.routerModels
			.filter(
				(m) =>
					m.status.value === ServerModelStatus.LOADED ||
					m.status.value === ServerModelStatus.SLEEPING
			)
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
		if (isRouterMode()) return null;

		const props = serverStore.props;
		if (props?.model_alias) return props.model_alias;
		if (!props?.model_path) return null;

		return props.model_path.split(/(\\|\/)/).pop() || null;
	}

	get selectedModelContextSize(): number | null {
		if (!this.selectedModelName) return null;
		return this.getModelContextSize(this.selectedModelName);
	}

	/**
	 *
	 *
	 * Modalities
	 *
	 *
	 */

	getModelModalities(modelId: string): ModelModalities | null {
		if (!isRouterMode() && serverStore.props?.modalities) {
			return this.buildModalities(serverStore.props.modalities);
		}

		const model = this.models.find((m) => m.model === modelId || m.id === modelId);
		if (model?.modalities) {
			return model.modalities;
		}

		const props = this.modelPropsCache.get(modelId);
		if (props?.modalities) {
			return this.buildModalities(props.modalities);
		}

		return null;
	}

	modelSupportsVision(modelId: string): boolean {
		return this.getModelModalities(modelId)?.vision ?? false;
	}

	modelSupportsAudio(modelId: string): boolean {
		return this.getModelModalities(modelId)?.audio ?? false;
	}

	modelSupportsVideo(modelId: string): boolean {
		return this.getModelModalities(modelId)?.video ?? false;
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
		return this.modelPropsCache.get(modelId);
	}

	getModelContextSize(modelId: string): number | null {
		const props = this.getModelProps(modelId);
		const nCtx = props?.default_generation_settings?.n_ctx;

		return typeof nCtx === 'number' ? nCtx : null;
	}

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

		return (
			model?.status.value === ServerModelStatus.LOADED ||
			model?.status.value === ServerModelStatus.SLEEPING
		);
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
	//
	// Thinking Support Detection
	//

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
		if (!isRouterMode()) {
			return detectThinkingSupport(serverStore.props?.chat_template ?? '');
		}

		const modelId = this.selectedModelName;
		if (!modelId) return false;

		if (!this.modelPropsCache.get(modelId)) {
			this.fetchModelProps(modelId);
		}
		const props = this.getModelProps(modelId);
		return detectThinkingSupport(props?.chat_template ?? '');
	}

	/**
	 * Check if a specific model supports thinking.
	 * In MODEL mode the global /props describes the single loaded model.
	 * In ROUTER mode, fetches model props if not cached.
	 */
	checkModelSupportsThinking(modelId: string): boolean {
		if (!isRouterMode()) {
			return detectThinkingSupport(serverStore.props?.chat_template ?? '');
		}

		if (!modelId) return false;

		if (!this.modelPropsCache.get(modelId)) {
			this.fetchModelProps(modelId);
		}

		const props = this.getModelProps(modelId);
		return detectThinkingSupport(props?.chat_template ?? '');
	}

	/**
	 * Detailed thinking support detection result with reason for debugging/UI.
	 */
	get thinkingSupportDetails(): { supported: boolean; reason: string } {
		if (!isRouterMode()) {
			return detectThinkingSupportWithReason(serverStore.props?.chat_template ?? '');
		}

		const modelId = this.selectedModelName;
		if (!modelId) {
			return { supported: false, reason: 'No model selected' };
		}

		if (!this.modelPropsCache.get(modelId)) {
			this.fetchModelProps(modelId);
		}
		const props = this.getModelProps(modelId);
		return detectThinkingSupportWithReason(props?.chat_template ?? '');
	}

	/**
	 *
	 *
	 * Data Fetching
	 *
	 *
	 */

	/**
	 * Fetch list of models from server and detect server role.
	 * Also fetches modalities for MODEL mode (single model).
	 */
	async fetch(force = false): Promise<void> {
		if (this.inflightFetch) return this.inflightFetch;
		if (this.models.length > 0 && !force) return;

		this.inflightFetch = this.runFetch();
		try {
			await this.inflightFetch;
		} finally {
			this.inflightFetch = null;
		}
	}

	private async runFetch(): Promise<void> {
		this.loading = true;
		this.error = null;

		try {
			if (!serverStore.props) {
				await serverStore.fetch();
			}

			const router = isRouterMode();

			if (router) {
				const response = await ModelsService.listRouter();

				this.routerModels = response.data;
				this.models = this.buildModelOptions(response);

				await this.fetchModalitiesForLoadedModels();

				const visible = this.getVisibleModels();

				if (visible.length === 1 && this.isModelLoaded(visible[0].model)) {
					this.selectModelById(visible[0].id);
				}
			} else {
				this.models = await this.fetchModelModeInternal();
			}
		} catch (error) {
			this.models = [];
			this.error = error instanceof Error ? error.message : 'Failed to load models';

			throw error;
		} finally {
			this.loading = false;
		}
	}

	/** Fetch models in MODEL mode (single model, standard OpenAI-compatible). */
	private async fetchModelModeInternal(): Promise<ModelOption[]> {
		const response = await ModelsService.list();

		return this.buildModelOptions(response);
	}

	/**
	 * Build ModelOption[] from an API response.
	 * Both MODEL and ROUTER modes share the same mapping logic;
	 * they differ only in which endpoint is called.
	 */
	private buildModelOptions(
		response: ApiModelListResponse | ApiRouterModelsListResponse
	): ModelOption[] {
		return response.data.map((item: ApiModelDataEntry, index: number) => {
			const details = response.models?.[index];
			const rawCapabilities = Array.isArray(details?.capabilities) ? details?.capabilities : [];
			const displayNameSource =
				details?.name && details.name.trim().length > 0 ? details.name : item.id;
			const modelId = details?.model || item.id;

			return {
				id: item.id,
				name: this.toDisplayName(displayNameSource),
				model: modelId,
				description: details?.description,
				capabilities: rawCapabilities.filter((value: unknown): value is string => Boolean(value)),
				details: details?.details,
				meta: item.meta ?? null,
				parsedId: ModelsService.parseModelId(modelId),
				aliases: item.aliases ?? [],
				tags: item.tags ?? []
			};
		});
	}

	/**
	 * Fetch router models with full metadata (ROUTER mode only).
	 * No-op in router mode — fetch() already calls listRouter() internally.
	 * Kept for API compatibility (e.g. handleOpenChange dropdown open handler).
	 */
	async fetchRouterModels(): Promise<void> {
		if (!isRouterMode()) return;

		try {
			const response = await ModelsService.listRouter();
			this.routerModels = response.data;
			await this.fetchModalitiesForLoadedModels();

			const visible = this.getVisibleModels();
			if (visible.length === 1 && this.isModelLoaded(visible[0].model)) {
				this.selectModelById(visible[0].id);
			}
		} catch (error) {
			console.warn('Failed to fetch router models:', error);
			this.routerModels = [];
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
			this.propsCacheVersion++;
			return props;
		} catch (error) {
			console.warn(`Failed to fetch props for model ${modelId}:`, error);
			return null;
		} finally {
			this.modelPropsFetching.delete(modelId);
		}
	}

	/** Fetch modalities for all loaded models from /props endpoint. */
	async fetchModalitiesForLoadedModels(): Promise<void> {
		const loadedModelIds = this.loadedModelIds;
		if (loadedModelIds.length === 0) return;

		const propsPromises = loadedModelIds.map((modelId) => this.fetchModelProps(modelId));

		try {
			const results = await Promise.all(propsPromises);

			this.models = this.models.map((model) => {
				const modelIndex = loadedModelIds.indexOf(model.model);
				if (modelIndex === -1) return model;

				const props = results[modelIndex];
				if (!props?.modalities) return model;

				return { ...model, modalities: this.buildModalities(props.modalities) };
			});

			this.propsCacheVersion++;
		} catch (error) {
			console.warn('Failed to fetch modalities for loaded models:', error);
		}
	}

	/**
	 * Update modalities for a specific model.
	 * Called when a model is loaded or when we need fresh modality data.
	 */
	async updateModelModalities(modelId: string): Promise<void> {
		const props = await this.fetchModelProps(modelId);
		if (!props?.modalities) return;

		this.models = this.models.map((model) =>
			model.model === modelId
				? { ...model, modalities: this.buildModalities(props.modalities!) }
				: model
		);

		this.propsCacheVersion++;
	}

	/**
	 * Filter to models visible in the UI (ui !== false).
	 */
	private getVisibleModels(): ModelOption[] {
		return this.models.filter((option) => this.getModelProps(option.model)?.ui !== false);
	}

	/**
	 * Gets the model name from the last assistant message in the active conversation.
	 * Used by both the chat page and settings page to maintain model consistency.
	 */
	getModelFromLastAssistantResponse(): string | null {
		const messages = conversationsStore.activeMessages;
		if (!messages || messages.length === 0) return null;

		for (let i = messages.length - 1; i >= 0; i--) {
			if (messages[i].model) {
				return messages[i].model;
			}
		}

		return null;
	}

	/**
	 * Auto-selects the model from the last assistant response if available and loaded.
	 * Returns true if a model was selected, false otherwise.
	 */
	async selectModelFromLastAssistantResponse(): Promise<boolean> {
		const lastModel = this.getModelFromLastAssistantResponse();
		if (!lastModel || this.selectedModelName === lastModel) return false;

		const matchingModel = this.models.find((option) => option.model === lastModel);
		if (!matchingModel || !this.isModelLoaded(lastModel)) return false;

		try {
			await this.selectModelById(matchingModel.id);
			console.log(`[modelsStore] Automatically selected model: ${lastModel} from last message`);
			return true;
		} catch (error) {
			console.warn('[modelsStore] Failed to automatically select model from last message:', error);
			return false;
		}
	}

	/**
	 * Auto-selects the first available model if none is selected.
	 * Prioritizes:
	 * 1. Model from active conversation's last assistant response (if loaded)
	 * 2. Model from active conversation's last assistant response (if not loaded)
	 * 3. First loaded model (not from active conversation)
	 * 4. A favorite model
	 * 5. First available model
	 */
	async ensureFirstModelSelected(): Promise<void> {
		if (this.selectedModelName) return;

		const availableModels = this.getVisibleModels();
		if (availableModels.length === 0) return;

		// Try to select model from last assistant response first
		const lastModel = this.getModelFromLastAssistantResponse();
		if (lastModel) {
			const lastModelOption = availableModels.find((m) => m.model === lastModel);
			if (lastModelOption) {
				await this.selectModelById(lastModelOption.id);
				if (this.isModelLoaded(lastModel)) {
					await this.fetchModelProps(lastModel);
				}
				return;
			}
		}

		// Try a loaded model first
		const loadedModel = availableModels.find((m) => this.isModelLoaded(m.model));
		if (loadedModel) {
			await this.selectModelById(loadedModel.id);
			await this.fetchModelProps(loadedModel.model);
			return;
		}

		// Try loading a favorite model
		const favorite = this.favoriteModelIds.values().next()?.value;
		if (favorite) {
			await this.selectModelById(favorite);
			return;
		}

		// Fall back to the first available model
		await this.selectModelById(availableModels[0].id);
	}

	/**
	 *
	 *
	 * Model Selection
	 *
	 *
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
	 * Select a model by its model name (used for syncing with conversation model).
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
		return (
			this.models.find(
				(model) =>
					model.model === modelName || model.id === modelName || model.aliases?.includes(modelName)
			) ?? null
		);
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
	 * Loading / Unloading Models
	 *
	 *
	 */

	// reconnect delay after the feed drops or the server is not ready yet
	private static readonly SSE_RECONNECT_MS = 1000;

	/**
	 * Open the /models/sse feed and keep it live with auto reconnect.
	 * Idempotent and router mode only. The feed drives status and progress,
	 * so it replaces any post-operation polling.
	 */
	subscribeStatus(): void {
		if (this.statusReaderActive) return;
		if (!isRouterMode()) return;

		this.statusReaderActive = true;
		this.statusAbort = new AbortController();
		void this.runStatusReader(this.statusAbort.signal);
	}

	/**
	 * Close the /models/sse feed and drop transient progress.
	 */
	unsubscribeStatus(): void {
		this.statusReaderActive = false;
		this.statusAbort?.abort();
		this.statusAbort = null;
		this.loadProgress.clear();
	}

	/**
	 * Current load progress for a model, or null when not loading.
	 */
	getLoadProgress(modelId: string): ModelLoadProgress | null {
		return this.loadProgress.get(modelId) ?? null;
	}

	/**
	 * Read the feed and reconnect until unsubscribed. Splits the byte stream
	 * into SSE records on the blank line boundary.
	 */
	private async runStatusReader(signal: AbortSignal): Promise<void> {
		const decoder = new TextDecoder();

		while (!signal.aborted) {
			try {
				const response = await fetch(`${base}${API_MODELS.SSE}`, {
					headers: getAuthHeaders(),
					signal
				});

				if (response.ok && response.body) {
					const reader = response.body.getReader();
					let buffer = '';

					while (!signal.aborted) {
						const { value, done } = await reader.read();
						if (done) break;

						buffer += decoder.decode(value, { stream: true });

						let boundary = buffer.indexOf(SSE_RECORD_SEPARATOR);
						while (boundary !== -1) {
							this.handleStatusRecord(buffer.slice(0, boundary));
							buffer = buffer.slice(boundary + SSE_RECORD_SEPARATOR.length);
							boundary = buffer.indexOf(SSE_RECORD_SEPARATOR);
						}
					}
				}
			} catch {
				// network drop or abort falls through to the reconnect delay
			}

			if (signal.aborted) return;

			await new Promise((resolve) => setTimeout(resolve, ModelsStore.SSE_RECONNECT_MS));
		}
	}

	/**
	 * Parse one SSE record. The payload rides in the data lines as a JSON
	 * envelope that carries its own model, event and data fields.
	 */
	private handleStatusRecord(record: string): void {
		const payload = record
			.split(SSE_LINE_SEPARATOR)
			.filter((line) => line.startsWith(SSE_DATA_PREFIX))
			.map((line) => line.slice(SSE_DATA_PREFIX.length).trim())
			.join(SSE_LINE_SEPARATOR);

		if (payload.length === 0) return;

		let envelope: ApiModelsSseEvent;
		try {
			envelope = JSON.parse(payload);
		} catch {
			return;
		}

		this.applyStatusEvent(envelope);
	}

	/**
	 * Route one feed record by event kind. Only the status_* events carry a
	 * status payload, models_reload triggers a list refresh, model_remove drops
	 * the row, download_* belong to the download surface, not here.
	 */
	private applyStatusEvent(event: ApiModelsSseEvent): void {
		switch (event.event) {
			case ServerModelsSseEventType.STATUS_CHANGE:
			case ServerModelsSseEventType.MODEL_STATUS:
			case ServerModelsSseEventType.STATUS_UPDATE:
				this.applyModelStatus(event);
				break;
			case ServerModelsSseEventType.MODELS_RELOAD:
				void this.fetchRouterModels();
				break;
			case ServerModelsSseEventType.MODEL_REMOVE:
				this.removeRouterModel(event.model);
				break;
			case ServerModelsSseEventType.DOWNLOAD_PROGRESS:
				break;
		}
	}

	/**
	 * Apply a status envelope: update the model row, track or clear progress,
	 * settle any pending load or unload awaiter.
	 */
	private applyModelStatus(event: ApiModelsSseEvent): void {
		const model = event.model;
		const data = event.data;
		if (!model || !data?.status) return;

		const status = data.status;

		this.setRouterModelStatus(model, status);

		if (status === ServerModelStatus.LOADING) {
			if (data.progress) this.loadProgress.set(model, data.progress);
		} else {
			this.loadProgress.delete(model);
		}

		if (status === ServerModelStatus.LOADED) {
			void this.updateModelModalities(model);
		}

		const failed =
			status === ServerModelStatus.FAILED ||
			(status === ServerModelStatus.UNLOADED && (data.exit_code ?? 0) !== 0);

		if (failed) {
			this.rejectStatus(model, new Error(`Model failed: ${this.toDisplayName(model)}`));
			return;
		}

		this.settleStatus(model, status);
	}

	/**
	 * Drop a model row reported gone by the feed and settle its awaiters.
	 */
	private removeRouterModel(modelId: string): void {
		if (this.routerModels.findIndex((m) => m.id === modelId) === -1) return;

		this.routerModels = this.routerModels.filter((m) => m.id !== modelId);
		this.loadProgress.delete(modelId);
		this.rejectStatus(modelId, new Error(`Model removed: ${this.toDisplayName(modelId)}`));
	}

	/**
	 * Update one model row status in place, reassigning to trigger reactivity.
	 */
	private setRouterModelStatus(modelId: string, status: ServerModelStatus): void {
		const idx = this.routerModels.findIndex((m) => m.id === modelId);
		if (idx === -1) return;

		const current = this.routerModels[idx];
		if (current.status.value === status) return;

		const next = [...this.routerModels];
		next[idx] = { ...current, status: { ...current.status, value: status } };
		this.routerModels = next;
	}

	/**
	 * Register an awaiter that resolves when the feed reports target status.
	 * One operation runs per model at a time, so one awaiter per model is kept.
	 */
	private waitForStatus(modelId: string, target: ServerModelStatus): Promise<void> {
		return new Promise((resolve, reject) => {
			this.statusWaiters.set(modelId, { target, resolve, reject });
		});
	}

	/**
	 * Resolve and drop the awaiter when the model reaches its target status.
	 */
	private settleStatus(modelId: string, status: ServerModelStatus): void {
		const waiter = this.statusWaiters.get(modelId);
		if (waiter && waiter.target === status) {
			this.statusWaiters.delete(modelId);
			waiter.resolve();
		}
	}

	/**
	 * Reject and drop the awaiter for a model.
	 */
	private rejectStatus(modelId: string, error: Error): void {
		const waiter = this.statusWaiters.get(modelId);
		if (waiter) {
			this.statusWaiters.delete(modelId);
			waiter.reject(error);
		}
	}

	async loadModel(modelId: string): Promise<void> {
		if (this.isModelLoaded(modelId)) return;
		if (this.modelLoadingStates.get(modelId)) return;

		this.modelLoadingStates.set(modelId, true);
		this.error = null;

		// the feed drives completion, so it must be live before the request
		this.subscribeStatus();

		const reachedLoaded = this.waitForStatus(modelId, ServerModelStatus.LOADED);
		reachedLoaded.catch(() => {});

		try {
			await ModelsService.load(modelId);
			await reachedLoaded;
			toast.success(`Model loaded: ${this.toDisplayName(modelId)}`);
		} catch (error) {
			this.rejectStatus(modelId, error instanceof Error ? error : new Error('load failed'));
			this.error = error instanceof Error ? error.message : 'Failed to load model';
			toast.error(`Failed to load model: ${this.toDisplayName(modelId)}`);
			throw error;
		} finally {
			this.modelLoadingStates.set(modelId, false);
		}
	}

	async unloadModel(modelId: string): Promise<void> {
		if (!this.isModelLoaded(modelId)) return;
		if (this.modelLoadingStates.get(modelId)) return;

		this.modelLoadingStates.set(modelId, true);
		this.error = null;

		this.subscribeStatus();

		const reachedUnloaded = this.waitForStatus(modelId, ServerModelStatus.UNLOADED);
		reachedUnloaded.catch(() => {});

		try {
			await ModelsService.unload(modelId);
			await reachedUnloaded;
			toast.info(`Model unloaded: ${this.toDisplayName(modelId)}`);
		} catch (error) {
			this.rejectStatus(modelId, error instanceof Error ? error : new Error('unload failed'));
			this.error = error instanceof Error ? error.message : 'Failed to unload model';
			toast.error(`Failed to unload model: ${this.toDisplayName(modelId)}`);
			throw error;
		} finally {
			this.modelLoadingStates.set(modelId, false);
		}
	}

	async ensureModelLoaded(modelId: string): Promise<void> {
		if (this.isModelLoaded(modelId)) return;
		await this.loadModel(modelId);
	}

	/**
	 *
	 *
	 * Favorites
	 *
	 *
	 */

	isFavorite(modelId: string): boolean {
		return this.favoriteModelIds.has(modelId);
	}

	toggleFavorite(modelId: string): void {
		const next = new SvelteSet(this.favoriteModelIds);

		if (next.has(modelId)) {
			next.delete(modelId);
		} else {
			next.add(modelId);
		}

		this.favoriteModelIds = next;

		try {
			localStorage.setItem(FAVORITE_MODELS_LOCALSTORAGE_KEY, JSON.stringify([...next]));
		} catch {
			toast.error('Failed to save favorite models to local storage');
		}
	}

	private loadFavoritesFromStorage(): Set<string> {
		try {
			const raw = localStorage.getItem(FAVORITE_MODELS_LOCALSTORAGE_KEY);
			return raw ? new Set(JSON.parse(raw) as string[]) : new Set();
		} catch {
			toast.error('Failed to load favorite models from local storage');
			return new Set();
		}
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

	private buildModalities(
		modalities: NonNullable<ApiLlamaCppServerProps['modalities']>
	): ModelModalities {
		return {
			vision: modalities.vision ?? false,
			audio: modalities.audio ?? false,
			video: modalities.video ?? false
		};
	}

	clear(): void {
		this.unsubscribeStatus();
		this.statusWaiters.forEach((waiter) => waiter.reject(new Error('Models store cleared')));
		this.statusWaiters.clear();
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
export const favoriteModelIds = () => modelsStore.favoriteModelIds;
export const supportsThinking = () => modelsStore.supportsThinking;
export const checkModelSupportsThinking = (modelId: string) =>
	modelsStore.checkModelSupportsThinking(modelId);
export const thinkingSupportDetails = () => modelsStore.thinkingSupportDetails;
