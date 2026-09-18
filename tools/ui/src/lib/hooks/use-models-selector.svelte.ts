import type { ModelItem } from '$lib/components/app/navigation/utils';
import {
	filterModelOptions,
	groupFavoriteOptions,
	groupModelOptions,
	groupProviderOptions
} from '$lib/components/app/navigation/utils';
import {
	CHAT_INPUT_FOCUS_SELECTOR,
	LOCAL_BACKEND_ID,
	REMOTE_PROVIDER_MODEL_LIMIT
} from '$lib/constants';
import { backendsModelsStore, backendsStore, modelsStore, serverStore } from '$lib/stores';
import type { ModelOption } from '$lib/types/models';
import { findBackendPreset } from '$lib/utils/backend';
import { rawModelId } from '$lib/utils/model-option-id';
import { onMount } from 'svelte';

/** Groups of the favorites tab, which lists favorites only. */
const EMPTY_GROUPS = { available: [], loaded: [], providers: [] };

export interface UseModelsSelectorOptions {
	currentModel: () => string | null;
	useGlobalSelection?: () => boolean;
	onModelChange?: () =>
		| ((
				modelId: string,
				modelName: string,
				backendId?: string
		  ) => Promise<boolean> | boolean | void)
		| undefined;
	onOpenChange?: (open: boolean) => void;
}

export interface UseModelsSelectorReturn {
	readonly options: ModelOption[];
	readonly loading: boolean;
	readonly updating: boolean;
	readonly activeId: string | null;
	readonly emptyMessage: string;
	readonly isMultiModel: boolean;
	readonly isRouter: boolean;
	readonly serverModel: string | null;
	readonly isHighlightedCurrentModelActive: boolean;
	readonly isCurrentModelInCache: boolean;
	readonly favoriteItems: ModelItem[];
	readonly filteredOptions: ModelOption[];
	readonly isEmpty: boolean;
	readonly isProviderView: boolean;
	readonly groupedFilteredOptions: ReturnType<typeof groupModelOptions>;
	readonly isLoadingModel: boolean;
	readonly searchTerm: string;
	readonly showModelDialog: boolean;
	readonly infoModelId: string | null;
	closeProvider(): void;
	openProvider(backendId: string): void;
	setSearchTerm(value: string): void;
	showBackendModels(backendId: string): Promise<void>;
	setShowModelDialog(value: boolean): void;
	handleInfoClick(modelName: string): void;
	handleSelect(modelId: string): Promise<void>;
	handleOpenChange(open: boolean): void;
	isFavorite(model: string): boolean;
	getDisplayOption(): ModelOption | undefined;
}

/**
 * Shared reactive state and logic for model selection.
 *
 * Used by both the desktop dropdown (`ModelsSelectorDropdown`)
 * and the mobile sheet (`ModelsSelectorSheet`) to avoid
 * duplicating store derivations, selection handling, and model loading.
 */
export function useModelsSelector(opts: UseModelsSelectorOptions): UseModelsSelectorReturn {
	/**
	 * Current view: the favorites of every backend, the local server's models, or
	 * the remote backends'. Favorites are the default while there is at least one.
	 */
	/** Remote backend drilled into from its section; null while browsing. */
	let providerViewId = $state<string | null>(null);

	const isProviderView = $derived(providerViewId !== null);
	const isLocalOption = (option: ModelOption) => option.backendId === LOCAL_BACKEND_ID;
	// every enabled backend's models are one list: favorites, then the local
	// server, then one section per remote provider
	const allOptions = $derived(
		modelsStore.models.filter((option) => {
			const modelProps = modelsStore.props.getModelProps(option.model);

			return modelProps?.ui !== false;
		})
	);
	const options = $derived(
		providerViewId ? allOptions.filter((option) => option.backendId === providerViewId) : allOptions
	);
	const loading = $derived(modelsStore.loading);
	const updating = $derived(modelsStore.updating);
	const activeId = $derived(modelsStore.selectedModelId);
	// Router mode and external backends both expose a selectable model list; only
	// a lone llama.cpp server without a router has nothing to choose from.
	const isMultiModel = $derived(
		serverStore.isRouterMode ||
			backendsStore.enabled.some((backend) => backend.id !== LOCAL_BACKEND_ID)
	);
	const isRouter = $derived(serverStore.isRouterMode);
	const serverModel = $derived(modelsStore.singleModelName);
	const currentModel = $derived(opts.currentModel());
	const onModelChange = $derived(opts.onModelChange?.());
	const isHighlightedCurrentModelActive = $derived.by(() => {
		if (!isRouter || !currentModel) return false;

		const currentOption = allOptions.find((option) => option.model === currentModel);

		return currentOption ? currentOption.id === activeId : false;
	});
	const isCurrentModelInCache = $derived.by(() => {
		if (!isRouter || !currentModel) return true;

		return allOptions.some((option) => option.model === currentModel);
	});

	let isLoadingModel = $state(false);
	let searchTerm = $state('');
	let showModelDialog = $state(false);
	let infoModelId = $state<string | null>(null);

	const filteredOptions = $derived(filterModelOptions(options, searchTerm));
	// favorites span every backend, so they come from the full option list
	const favoriteItems = $derived(
		groupFavoriteOptions(filterModelOptions(allOptions, searchTerm), modelsStore.favoriteModelIds)
	);
	const remoteProviders = $derived(
		backendsStore.enabled
			.filter((backend) => backend.id !== LOCAL_BACKEND_ID)
			.map((backend) => {
				const state = backendsModelsStore.get(backend.id);

				return {
					backendId: backend.id,
					catalog: state.models.length,
					error: state.error,
					loading: state.loading,
					name: backend.name,
					preset: findBackendPreset(backend.baseUrl)
				};
			})
	);
	const providerSections = $derived(
		groupProviderOptions(
			filteredOptions,
			remoteProviders,
			// a drill-in or a search reaches every model, the sections stay short
			providerViewId || searchTerm ? Infinity : REMOTE_PROVIDER_MODEL_LIMIT
		)
	);
	const groupedFilteredOptions = $derived.by(() => {
		if (isProviderView) {
			const sections = providerSections.filter((section) => section.backendId === providerViewId);

			return { ...EMPTY_GROUPS, providers: sections };
		}

		const local = groupModelOptions(filteredOptions.filter(isLocalOption), (m) =>
			modelsStore.isModelLoaded(m)
		);

		return { ...local, providers: providerSections };
	});
	const isEmpty = $derived(filteredOptions.length === 0 && favoriteItems.length === 0);
	const emptyMessage = $derived(searchTerm ? 'No models found.' : 'No models yet.');

	function handleInfoClick(modelName: string) {
		infoModelId = modelName;
		showModelDialog = true;
	}

	onMount(() => {
		modelsStore.fetch().catch((error) => {
			console.error('Unable to load models:', error);
		});
	});

	function handleOpenChange(open: boolean) {
		if (loading || updating) return;

		// a single-model llama.cpp server with no other backend has no list to
		// show, so the trigger opens the model info dialog instead
		if (!isMultiModel) {
			showModelDialog = open;

			return;
		}

		searchTerm = '';
		providerViewId = null;

		if (open && isRouter) {
			modelsStore.props.fetchModalitiesForLoadedModels();
		}

		opts.onOpenChange?.(open);
	}

	/**
	 * Switch the rendered view. Views are display only: the backend that serves
	 * requests follows the selected model, not the view.
	 */
	/** Drill into one remote backend's full model list. */
	function openProvider(backendId: string) {
		providerViewId = backendId;
		searchTerm = '';
	}

	function closeProvider() {
		providerViewId = null;
		searchTerm = '';
	}

	/** Refresh a backend's models, e.g. right after it was added. */
	async function showBackendModels(backendId: string): Promise<void> {
		await backendsModelsStore.ensureLoaded(backendId);
	}

	async function handleSelect(modelId: string) {
		// favorites live above the tabs and may belong to another backend, so the
		// lookup spans every enabled backend
		const option = allOptions.find((opt) => opt.id === modelId);

		if (!option) return;

		let shouldCloseMenu = true;

		if (onModelChange) {
			const result = await onModelChange(rawModelId(option.id), option.model, option.backendId);

			if (result === false) {
				shouldCloseMenu = false;
			}
		} else {
			await modelsStore.selectModelById(option.id);
		}

		if (shouldCloseMenu) {
			handleOpenChange(false);

			requestAnimationFrame(() => {
				const input = document.querySelector<HTMLElement>(CHAT_INPUT_FOCUS_SELECTOR);

				input?.focus({ preventScroll: true });
			});
		}

		// only the built-in server loads on request, and only in router mode
		const canLoadHere = option.backendId === LOCAL_BACKEND_ID && isRouter;

		if (!onModelChange && canLoadHere && !modelsStore.isModelLoaded(option.model)) {
			isLoadingModel = true;

			modelsStore.status
				.load(option.model)
				.catch((error) => console.error('Failed to load model:', error))
				.finally(() => (isLoadingModel = false));
		}
	}

	function getDisplayOption(): ModelOption | undefined {
		if (!isRouter) {
			// External backend: the selection is backend-scoped, so it wins over
			// the conversation's model, which may belong to another backend.
			if (!serverStore.capabilities.props) {
				const selected = activeId ? allOptions.find((option) => option.id === activeId) : undefined;

				if (selected) return selected;

				return currentModel
					? allOptions.find((option) => option.model === currentModel)
					: undefined;
			}

			const displayModel = serverModel || currentModel;

			if (displayModel) {
				return {
					capabilities: [],
					id: serverModel ? 'current' : 'offline-current',
					model: displayModel,
					name: displayModel.split('/').pop() || displayModel
				};
			}

			return undefined;
		}

		if (currentModel) {
			if (!isCurrentModelInCache) {
				return {
					capabilities: [],
					id: 'not-in-cache',
					model: currentModel,
					name: currentModel.split('/').pop() || currentModel
				};
			}

			return allOptions.find((option) => option.model === currentModel);
		}

		if (activeId) {
			return allOptions.find((option) => option.id === activeId);
		}

		return undefined;
	}

	return {
		get activeId() {
			return activeId;
		},

		closeProvider,

		get emptyMessage() {
			return emptyMessage;
		},

		get favoriteItems() {
			return favoriteItems;
		},

		get filteredOptions() {
			return filteredOptions;
		},

		getDisplayOption,

		get groupedFilteredOptions() {
			return groupedFilteredOptions;
		},

		handleInfoClick,

		handleOpenChange,

		handleSelect,

		get infoModelId() {
			return infoModelId;
		},

		get isCurrentModelInCache() {
			return isCurrentModelInCache;
		},

		get isEmpty() {
			return isEmpty;
		},

		isFavorite(model: string) {
			return modelsStore.favoriteModelIds.has(model);
		},

		get isHighlightedCurrentModelActive() {
			return isHighlightedCurrentModelActive;
		},

		get isLoadingModel() {
			return isLoadingModel;
		},

		get isMultiModel() {
			return isMultiModel;
		},

		get isProviderView() {
			return isProviderView;
		},

		get isRouter() {
			return isRouter;
		},

		get loading() {
			return loading;
		},

		openProvider,

		get options() {
			return options;
		},

		get searchTerm() {
			return searchTerm;
		},

		get serverModel() {
			return serverModel;
		},

		setSearchTerm(value: string) {
			searchTerm = value;
		},

		setShowModelDialog(value: boolean) {
			showModelDialog = value;
		},

		showBackendModels,

		get showModelDialog() {
			return showModelDialog;
		},

		get updating() {
			return updating;
		}
	};
}
