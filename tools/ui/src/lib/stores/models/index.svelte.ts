/**
 * modelsStore - Model management for MODEL and ROUTER modes
 *
 * Owns model lists, selection, favorites and load/unload state. Composes the
 * per-model props cache (modalities, thinking detection) as
 * {@link ModelsStore.props} and the /models/sse status feed as
 * {@link ModelsStore.status}; tracks which conversations use which models.
 */

import { browser } from '$app/environment';
import {
	FAVORITE_MODELS_LOCALSTORAGE_KEY,
	RECENT_MODEL_LIMIT,
	RECENT_MODELS_LOCALSTORAGE_KEY,
	SELECTED_MODEL_LOCALSTORAGE_KEY
} from '$lib/constants';
import { ServerModelStatus } from '$lib/enums';
import { ModelsService } from '$lib/services/models.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { backendsStore } from '$lib/stores/backends.svelte';
import { backendsModelsStore } from '$lib/stores/backendsModels.svelte';
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { type ModelPropsHost, ModelPropsManager } from '$lib/stores/models/props.svelte';
import { type ModelStatusHost, ModelStatusManager } from '$lib/stores/models/status.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { readModelContextLength } from '$lib/utils/backend';
import { getConversationModel } from '$lib/utils/conversation-utils';
import { backendIdFromModelId, qualifyModelId, rawModelId } from '$lib/utils/model-option-id';
import { SvelteSet } from 'svelte/reactivity';
import { toast } from 'svelte-sonner';

/** Selection kept from the last session, so a reload does not drop the picked model. */
function loadStoredSelection(): { id: string; model: string | null } | null {
	if (!browser) return null;

	try {
		const raw = localStorage.getItem(SELECTED_MODEL_LOCALSTORAGE_KEY);

		if (!raw) return null;

		const parsed = JSON.parse(raw) as { id?: unknown; model?: unknown };

		if (typeof parsed?.id !== 'string' || !parsed.id) return null;

		return { id: parsed.id, model: typeof parsed.model === 'string' ? parsed.model : null };
	} catch {
		return null;
	}
}

const storedSelection = loadStoredSelection();

/** Recently used backend-qualified ids, most recent first. */
function loadRecentModels(): string[] {
	if (!browser) return [];

	try {
		const raw = localStorage.getItem(RECENT_MODELS_LOCALSTORAGE_KEY);

		if (!raw) return [];

		const parsed = JSON.parse(raw) as unknown;

		return Array.isArray(parsed)
			? parsed.filter((id): id is string => typeof id === 'string').slice(0, RECENT_MODEL_LIMIT)
			: [];
	} catch {
		return [];
	}
}

class ModelsStore implements ModelPropsHost, ModelStatusHost {
	activeModels = $state<ModelOption[]>([]);
	error = $state<string | null>(null);
	favoriteModelIds = $state<Set<string>>(this.loadFavoritesFromStorage());
	loading = $state(false);
	recentModelIds = $state<string[]>(loadRecentModels());
	routerModels = $state<ApiModelDataEntry[]>([]);
	selectedModelId = $state<string | null>(storedSelection?.id ?? null);
	selectedModelName = $state<string | null>(storedSelection?.model ?? null);

	updating = $state(false);

	/** Per-model props cache, modalities and thinking detection, composed here. */
	private _props = new ModelPropsManager(this);

	/** Load/unload operations and the /models/sse status feed, composed here. */
	private _status = new ModelStatusManager(this);

	// Dedup concurrent fetch() callers — all awaiters share the same inflight promise.
	// Without this, ?model=<name> URL handler races an in-progress fetch and sees an empty list.
	private inflightFetch: Promise<void> | null = null;

	/** A restored selection loses to the active conversation's model, a fresh pick does not. */
	private selectionFromStorage = storedSelection !== null;

	/**
	 * Model the active conversation view resolves to. Router mode: the user's
	 * selection first, then the conversation's own model. Otherwise the single
	 * served model, from the models list or the server props as a fallback.
	 */
	get activeModelId(): string | null {
		if (!serverStore.isRouterMode) {
			// external backends expose a selectable list; prefer the user's pick
			const selected = this.selectedModelId
				? this.models.find((m) => m.id === this.selectedModelId)
				: undefined;

			if (selected) return selected.model;

			return this.models.length > 0 ? this.models[0].model : this.singleModelName;
		}

		const picked = this.selectedModelId && !this.selectionFromStorage ? this.selectedModelId : null;
		const selected = picked ? this.models.find((m) => m.id === picked) : undefined;

		if (selected) return selected.model;

		const conversationModel = getConversationModel(conversationsStore.activeMessages);

		if (conversationModel) {
			const model = this.models.find((m) => m.model === conversationModel);

			if (model) return model.model;
		}

		const restored = this.selectedModelId
			? this.models.find((m) => m.id === this.selectedModelId)
			: undefined;

		return restored?.model ?? null;
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

	/**
	 * Every selectable model across enabled backends. The active backend's
	 * models come from {@link activeModels}; the rest come from the background
	 * prefetch cache. Ids are backend-qualified so the same model name on two
	 * backends stays distinct.
	 */
	get models(): ModelOption[] {
		const activeBackendId = backendsStore.active.id;
		const merged: ModelOption[] = [];
		const seen = new SvelteSet<string>();
		const push = (option: ModelOption, backendId: string) => {
			const id = qualifyModelId(backendId, rawModelId(option.id));

			// a backend can be listed twice while a switch is in flight: the rows
			// of the previous backend are still in activeModels
			if (seen.has(id)) return;

			seen.add(id);
			merged.push({ ...option, backendId, id });
		};

		for (const option of this.activeModels) {
			// keep the backend an option was built for: rows from the previous
			// backend must not be relabelled while a switch is in flight
			push(option, option.backendId ?? activeBackendId);
		}

		for (const backend of backendsStore.enabled) {
			if (backend.id === activeBackendId) continue;

			for (const option of backendsModelsStore.get(backend.id).models) {
				push(option, option.backendId ?? backend.id);
			}
		}

		return merged;
	}

	get props() {
		return this._props;
	}

	get selectedModel(): ModelOption | null {
		if (!this.selectedModelId) return null;

		return this.models.find((m) => m.id === this.selectedModelId) ?? null;
	}

	get selectedModelContextSize(): number | null {
		if (!this.selectedModelName) return null;

		return this.props.getModelContextSize(this.selectedModelName);
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

	get status() {
		return this._status;
	}

	clearSelection(): void {
		this.selectedModelId = null;
		this.selectedModelName = null;
		this.persistSelection();
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
					await this.props.fetchModelProps(lastModel);
				}

				return;
			}
		}

		// Try a loaded model first
		const loadedModel = availableModels.find((m) => this.isModelLoaded(m.model));

		if (loadedModel) {
			await this.selectModelById(loadedModel.id);
			await this.props.fetchModelProps(loadedModel.model);

			return;
		}

		// Try a favorite model, but only one that exists on this backend: favorites
		// are shared across backends, so a stored id may belong to another one
		const favorite = this.favoriteModelIds.values().next()?.value;
		const favoriteOption = favorite
			? availableModels.find((m) => m.id === favorite || m.model === favorite)
			: undefined;

		if (favoriteOption) {
			await this.selectModelById(favoriteOption.id);

			return;
		}

		// Fall back to the first available model
		await this.selectModelById(availableModels[0].id);
	}

	/**
	 * Make the backend serving `modelName` active when it is not already.
	 * A conversation keeps the model that generated it, which can belong to a
	 * backend other than the active one.
	 */
	async ensureModelBackend(modelName: string): Promise<void> {
		const option = this.models.find((model) => model.model === modelName);

		if (!option?.backendId || option.backendId === backendsStore.active.id) return;

		await this.selectModelById(option.id);
	}

	/**
	 * Fetch list of models from server and detect server role.
	 * Also fetches modalities for MODEL mode (single model).
	 */
	async fetch(force = false): Promise<void> {
		if (this.inflightFetch) return this.inflightFetch;

		if (this.activeModels.length > 0 && !force) return;

		this.inflightFetch = this.runFetch();
		try {
			await this.inflightFetch;
		} finally {
			this.inflightFetch = null;
		}
	}

	/**
	 * Fetch models with full metadata (ROUTER mode only).
	 * No-op in MODEL mode - fetch() already calls list() internally.
	 * Kept for API compatibility (e.g. handleOpenChange dropdown open handler).
	 */
	async fetchRouterModels(): Promise<void> {
		if (!serverStore.isRouterMode) return;

		try {
			const response = await ModelsService.list();

			this.routerModels = response.data;
			// keep the selector options in sync: a downloaded / deleted model shows
			// up here too, not only in the router model rows
			this.activeModels = this.buildModelOptions(response);
			await this.props.fetchModalitiesForLoadedModels();

			const visible = this.getVisibleModels();

			if (visible.length === 1 && this.isModelLoaded(visible[0].model)) {
				this.selectModelById(visible[0].id);
			}
		} catch (error) {
			console.warn('Failed to fetch router models:', error);
			this.routerModels = [];
		}
	}

	findModelById(modelId: string): ModelOption | null {
		return this.models.find((model) => model.id === modelId) ?? null;
	}

	findModelByName(modelName: string): ModelOption | null {
		return (
			this.models.find(
				(model) =>
					model.model === modelName || model.id === modelName || model.aliases?.includes(modelName)
			) ?? null
		);
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

	getModelStatus(modelId: string): ServerModelStatus | null {
		const model = this.routerModels.find((m) => m.id === modelId);

		return model?.status.value ?? null;
	}

	hasModel(modelName: string): boolean {
		return this.models.some((model) => model.model === modelName);
	}

	isFavorite(modelId: string): boolean {
		return this.favoriteModelIds.has(modelId);
	}

	isModelLoaded(modelId: string): boolean {
		const model = this.routerModels.find((m) => m.id === modelId);

		return (
			model?.status.value === ServerModelStatus.LOADED ||
			model?.status.value === ServerModelStatus.SLEEPING
		);
	}

	/**
	 * Select a model. `recordRecent` marks a pick the user made in the selector, so
	 * automatic picks (startup default, conversation sync) stay out of the recency list.
	 */
	async selectModelById(modelId: string, options?: { recordRecent?: boolean }): Promise<void> {
		if (!modelId || this.updating) return;

		const backendId = backendIdFromModelId(modelId) ?? backendsStore.active.id;
		const rawId = rawModelId(modelId);
		// the selection is stored backend-qualified, matching the aggregated
		// model list, no matter which form the caller passed
		const qualifiedId = qualifyModelId(backendId, rawId);

		// a model from another backend makes that backend active first
		if (backendId !== backendsStore.active.id) {
			backendsStore.setActive(backendId);
			await backendsModelsStore.ensureLoaded(backendId);
			await this.switchBackend();
		}

		if (this.selectedModelId === qualifiedId) {
			if (options?.recordRecent) this.recordRecentModel(qualifiedId);

			return;
		}

		const option = this.activeModels.find((model) => model.id === rawId);

		if (!option) throw new Error('Selected model is not available');

		this.updating = true;
		this.error = null;

		try {
			this.selectedModelId = qualifiedId;
			this.selectedModelName = option.model;
			this.selectionFromStorage = false;
			this.persistSelection();

			if (options?.recordRecent) this.recordRecentModel(qualifiedId);
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
			void this.selectModelById(option.id);
		}
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
	 * Activate a backend for the selector tabs. Everything comes from memory:
	 * the model list and router rows are prefetched at startup and the local
	 * server state is kept while an external backend is active. The selection
	 * is left alone, switching tabs must not pick a model.
	 */
	async switchBackend(): Promise<void> {
		this.error = null;

		const backend = backendsStore.active;

		// local props describe the server the UI is served from; keep them while
		// an external backend is active instead of dropping and refetching
		if (backend.protocol === 'llama.cpp') {
			serverStore.restoreLocalState();
		} else {
			serverStore.cacheLocalState();
			serverStore.clear();
		}

		const cached = backendsModelsStore.get(backend.id);

		if (!cached.loaded) {
			// nothing prefetched for this backend (startup prefetch failed): load it once
			await this.fetch(true);

			return;
		}

		if (backend.protocol === 'llama.cpp' && !serverStore.props) {
			// first visit to the local tab in this session
			await serverStore.fetch({ background: true });
		}

		this.activeModels = cached.models;
		this.loading = false;

		// the local router rows carry the load statuses; the startup prefetch
		// already returned them, so a tab switch rebuilds the list from memory
		if (backend.protocol === 'llama.cpp' && this.routerModels.length === 0 && cached.raw) {
			this.routerModels = cached.raw.data;
			this.activeModels = this.buildModelOptions(cached.raw);
		}
	}

	toDisplayName(id: string): string {
		const segments = id.split(/\\|\//);
		const candidate = segments.pop();

		return candidate && candidate.trim().length > 0 ? candidate : id;
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

	/**
	 * Build ModelOption[] from an API response.
	 * Both MODEL and ROUTER modes share the same mapping logic;
	 * they differ only in which endpoint is called.
	 */
	private buildModelOptions(response: ApiModelsListResponse): ModelOption[] {
		const entries: {
			details?: ApiModelsListResponse['models'][number];
			item: ApiModelDataEntry;
		}[] = response.data.map((item: ApiModelDataEntry, index: number) => ({
			details: response.models?.[index],
			item
		}));

		return (
			entries
				// sidecar entries mark downloaded sidecar files, not loadable models
				.filter(({ item }) => !ModelsService.isSidecarEntry(item.id))
				// in-flight downloads are not usable models yet; the selector tracks
				// them in its "Download in progress" section instead
				.filter(({ item }) => item.status?.value !== ServerModelStatus.DOWNLOADING)
				.map(({ details, item }) => {
					const rawCapabilities = Array.isArray(details?.capabilities) ? details?.capabilities : [];
					const displayNameSource =
						details?.name && details.name.trim().length > 0 ? details.name : item.id;
					const modelId = details?.model || item.id;

					return {
						aliases: item.aliases ?? [],
						// stamp the backend here so the option keeps its origin even
						// after another backend becomes active
						backendId: backendsStore.active.id,
						capabilities: rawCapabilities.filter((value: unknown): value is string =>
							Boolean(value)
						),
						// external backends report the context in their listing, so the
						// gauge keeps working when the list is rebuilt on reload
						contextLength: readModelContextLength(item),
						description: details?.description,
						details: details?.details,
						id: item.id,
						meta: item.meta ?? null,
						modalities: this.props.buildArchitectureModalities(item.architecture),
						model: modelId,
						name: this.toDisplayName(displayNameSource),
						parsedId: ModelsService.parseModelId(modelId),
						tags: item.tags ?? []
					};
				})
		);
	}

	/** Fetch models in MODEL mode (single model, standard OpenAI-compatible). */
	private async fetchModelModeInternal(): Promise<ModelOption[]> {
		const response = await ModelsService.list();

		return this.buildModelOptions(response);
	}

	/**
	 * Filter to models visible in the UI (ui !== false).
	 */
	private getVisibleModels(): ModelOption[] {
		return this.activeModels.filter(
			(option) => this.props.getModelProps(option.model)?.ui !== false
		);
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

	private persistSelection(): void {
		if (!browser) return;

		try {
			if (!this.selectedModelId) {
				localStorage.removeItem(SELECTED_MODEL_LOCALSTORAGE_KEY);
			} else {
				localStorage.setItem(
					SELECTED_MODEL_LOCALSTORAGE_KEY,
					JSON.stringify({ id: this.selectedModelId, model: this.selectedModelName })
				);
			}
		} catch {
			console.warn('[ModelsStore] Failed to persist the model selection');
		}
	}

	/** Move a model to the front of the recently used list. */
	private recordRecentModel(qualifiedId: string): void {
		this.recentModelIds = [
			qualifiedId,
			...this.recentModelIds.filter((id) => id !== qualifiedId)
		].slice(0, RECENT_MODEL_LIMIT);

		if (!browser) return;

		try {
			localStorage.setItem(RECENT_MODELS_LOCALSTORAGE_KEY, JSON.stringify(this.recentModelIds));
		} catch {
			console.warn('[ModelsStore] Failed to persist the recently used models');
		}
	}

	private async runFetch(): Promise<void> {
		this.loading = true;
		this.error = null;

		try {
			if (!serverStore.props) {
				await serverStore.fetch();
			}

			const router = serverStore.isRouterMode;

			if (router) {
				const response = await ModelsService.list();

				this.routerModels = response.data;
				this.activeModels = this.buildModelOptions(response);

				await this.props.fetchModalitiesForLoadedModels();

				const visible = this.getVisibleModels();

				if (visible.length === 1 && this.isModelLoaded(visible[0].model)) {
					this.selectModelById(visible[0].id);
				}
			} else {
				this.activeModels = await this.fetchModelModeInternal();

				// external backends expose a selectable list; pick a default so the
				// first send and title generation have a model to target
				if (!serverStore.capabilities.props && !this.selectedModelName) {
					await this.ensureFirstModelSelected();
				}
			}
		} catch (error) {
			this.activeModels = [];
			this.error = error instanceof Error ? error.message : 'Failed to load models';

			throw error;
		} finally {
			this.loading = false;
		}
	}
}

export const modelsStore = new ModelsStore();
