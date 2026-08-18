import { FAVORITE_MODELS_LOCALSTORAGE_KEY } from '$lib/constants';
import { ServerModelStatus } from '$lib/enums';
import { ModelsService } from '$lib/services/models.service';
// direct imports between stores, not via the barrel, to avoid circular deps
import { conversationsStore } from '$lib/stores/conversations/index.svelte';
import { ModelPropsManager } from '$lib/stores/models/props.svelte';
import { ModelStatusManager } from '$lib/stores/models/status.svelte';
import { serverStore } from '$lib/stores/server.svelte';
import { getConversationModel } from '$lib/utils/conversation-utils';
import { SvelteSet } from 'svelte/reactivity';
import { toast } from 'svelte-sonner';

/**
 * modelsStore - Reactive store for model management in both MODEL and ROUTER modes.
 *
 * **Architecture & Relationships:**
 * - **ModelsService**: Stateless service for model API communication
 * - **modelsStore** (this class): Reactive store for model state
 * - **ModelPropsManager**: Props cache, modalities, thinking detection - composed as {@link ModelsStore.props}
 * - **ModelStatusManager**: Load/unload + /models/sse status feed - composed as {@link ModelsStore.status}
 * - **conversationsStore**: Tracks which conversations use which models
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

	/** Per-model props cache, modalities and thinking detection, composed here. */
	private _props = new ModelPropsManager(this);

	/** Load/unload operations and the /models/sse status feed, composed here. */
	private _status = new ModelStatusManager(this);

	get props() {
		return this._props;
	}

	get status() {
		return this._status;
	}

	favoriteModelIds = $state<Set<string>>(this.loadFavoritesFromStorage());

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
	 * Model the active conversation view resolves to. Router mode: the user's
	 * selection first, then the conversation's own model. Otherwise the single
	 * served model, from the models list or the server props as a fallback.
	 */
	get activeModelId(): string | null {
		if (!serverStore.isRouterMode) {
			return this.models.length > 0 ? this.models[0].model : this.singleModelName;
		}

		if (this.selectedModelId) {
			const selected = this.models.find((m) => m.id === this.selectedModelId);

			if (selected) return selected.model;
		}

		const conversationModel = getConversationModel(conversationsStore.activeMessages);

		if (conversationModel) {
			const model = this.models.find((m) => m.model === conversationModel);

			if (model) return model.model;
		}

		return null;
	}

	get selectedModelContextSize(): number | null {
		if (!this.selectedModelName) return null;

		return this.props.getModelContextSize(this.selectedModelName);
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

	getModelStatus(modelId: string): ServerModelStatus | null {
		const model = this.routerModels.find((m) => m.id === modelId);

		return model?.status.value ?? null;
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

			const router = serverStore.isRouterMode;

			if (router) {
				const response = await ModelsService.listRouter();

				this.routerModels = response.data;
				this.models = this.buildModelOptions(response);

				await this.props.fetchModalitiesForLoadedModels();

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
				aliases: item.aliases ?? [],
				capabilities: rawCapabilities.filter((value: unknown): value is string => Boolean(value)),
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
		});
	}

	/**
	 * Fetch router models with full metadata (ROUTER mode only).
	 * No-op in router mode — fetch() already calls listRouter() internally.
	 * Kept for API compatibility (e.g. handleOpenChange dropdown open handler).
	 */
	async fetchRouterModels(): Promise<void> {
		if (!serverStore.isRouterMode) return;

		try {
			const response = await ModelsService.listRouter();

			this.routerModels = response.data;
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

	/**
	 * Filter to models visible in the UI (ui !== false).
	 */
	private getVisibleModels(): ModelOption[] {
		return this.models.filter((option) => this.props.getModelProps(option.model)?.ui !== false);
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

	toDisplayName(id: string): string {
		const segments = id.split(/\\|\//);
		const candidate = segments.pop();

		return candidate && candidate.trim().length > 0 ? candidate : id;
	}
}

export const modelsStore = new ModelsStore();
