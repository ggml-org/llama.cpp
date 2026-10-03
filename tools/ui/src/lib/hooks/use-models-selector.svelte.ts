import { CHAT_INPUT_FOCUS_SELECTOR } from '$lib/constants';
import { modelsStore, serverStore, uiStore } from '$lib/stores';
import type { ModelOption } from '$lib/types/models';
import {
	filterModelOptions,
	groupFavoriteOptions,
	groupModelOptions,
	type ModelItem
} from '$lib/utils';
import { onMount } from 'svelte';
import { SvelteSet } from 'svelte/reactivity';

export interface UseModelsSelectorOptions {
	currentModel: () => string | null;
	useGlobalSelection?: () => boolean;
	onModelChange?: () =>
		| ((modelId: string, modelName: string) => Promise<boolean> | boolean | void)
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
	readonly loadedItems: ModelItem[];
	readonly filteredOptions: ModelOption[];
	readonly isEmpty: boolean;
	readonly groupedFilteredOptions: ReturnType<typeof groupModelOptions>;
	readonly isLoadingModel: boolean;
	readonly searchTerm: string;
	setSearchTerm(value: string): void;
	handleSelect(modelId: string): Promise<void>;
	handleOpenChange(open: boolean): void;
	isFavorite(model: string): boolean;
	getDisplayOption(): ModelOption | undefined;
}

/**
 * Shared reactive state and logic for model selection.
 *
 * Used by both the desktop dropdown (`ModelsSelectorDropdown`)
 * and the mobile trigger (`ModelsSelectorMobileTrigger`) to avoid
 * duplicating store derivations, selection handling, and model loading.
 */
export function useModelsSelector(opts: UseModelsSelectorOptions): UseModelsSelectorReturn {
	let isLoadingModel = $state(false);
	let searchTerm = $state('');

	const options = $derived(
		modelsStore.models.filter((option) => {
			const modelProps = modelsStore.props.getModelProps(option.model);

			return modelProps?.ui !== false;
		})
	);
	const loading = $derived(modelsStore.loading);
	const updating = $derived(modelsStore.updating);
	const activeId = $derived(modelsStore.selectedModelId);
	// a lone llama.cpp server without a router has nothing to choose from
	const isRouter = $derived(serverStore.isRouterMode);
	const serverModel = $derived(modelsStore.singleModelName);
	const currentModel = $derived(opts.currentModel());
	const onModelChange = $derived(opts.onModelChange?.());
	const isHighlightedCurrentModelActive = $derived.by(() => {
		if (!isRouter || !currentModel) return false;

		const currentOption = options.find((option) => option.model === currentModel);

		return currentOption ? currentOption.id === activeId : false;
	});
	const isCurrentModelInCache = $derived.by(() => {
		if (!isRouter || !currentModel) return true;

		return options.some((option) => option.model === currentModel);
	});
	// the search, the rows and the sections all read the visible set; only the current
	// model resolves against `options`, since it can be hidden and still selected
	const visibleOptions = $derived(options.filter((option) => !modelsStore.isHidden(option.id)));
	// one filter pass feeds the favorites, the loaded rows and the sections alike
	const filteredOptions = $derived(filterModelOptions(visibleOptions, searchTerm));
	const favoriteItems = $derived(
		groupFavoriteOptions(filteredOptions, modelsStore.favoriteModelIds)
	);
	const loadedItems = $derived(
		filteredOptions
			.filter((option) => modelsStore.isModelLoaded(option.model))
			.map((option) => ({ option }))
	);
	const loadedIds = $derived(new SvelteSet(loadedItems.map((item) => item.option.id)));
	// loaded models and favorites are listed once, at the top: the sections skip both
	const sectionOptions = $derived(
		filteredOptions.filter(
			(option) => !modelsStore.favoriteModelIds.has(option.model) && !loadedIds.has(option.id)
		)
	);
	const groupedFilteredOptions = $derived(groupModelOptions(sectionOptions));
	const isEmpty = $derived(
		filteredOptions.length === 0 && favoriteItems.length === 0 && loadedItems.length === 0
	);
	const emptyMessage = $derived(searchTerm ? 'No models found.' : 'No models yet.');

	// the manager takes the focus, so the selector closes instead of sitting behind it
	$effect(() => {
		if (!uiStore.manageModelsOpen) return;

		opts.onOpenChange?.(false);
	});

	onMount(() => {
		modelsStore.fetch().catch((error) => {
			console.error('Unable to load models:', error);
		});
	});

	function handleOpenChange(open: boolean) {
		if (loading || updating) return;

		// a single-model server has no list: the trigger opens the manager instead
		if (!isRouter) {
			if (open) uiStore.openModelsManager();

			return;
		}

		searchTerm = '';

		if (open && isRouter) {
			modelsStore.props.fetchModalitiesForLoadedModels();
		}

		opts.onOpenChange?.(open);
	}

	async function handleSelect(modelId: string) {
		const option = options.find((opt) => opt.id === modelId);

		if (!option) return;

		let shouldCloseMenu = true;

		if (onModelChange) {
			const result = await onModelChange(option.id, option.model);

			if (result === false) {
				shouldCloseMenu = false;
			}
		} else {
			await modelsStore.selectModelById(option.id, { recordRecent: true });
		}

		if (shouldCloseMenu) {
			handleOpenChange(false);

			requestAnimationFrame(() => {
				const input = document.querySelector<HTMLElement>(CHAT_INPUT_FOCUS_SELECTOR);

				input?.focus({ preventScroll: true });
			});
		}

		// only the built-in server loads on request, and only in router mode
		if (!onModelChange && isRouter && !modelsStore.isModelLoaded(option.model)) {
			isLoadingModel = true;

			modelsStore.status
				.load(option.model)
				.catch((error) => console.error('Failed to load model:', error))
				.finally(() => (isLoadingModel = false));
		}
	}

	function getDisplayOption(): ModelOption | undefined {
		if (!isRouter) {
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

			return options.find((option) => option.model === currentModel);
		}

		if (activeId) {
			return options.find((option) => option.id === activeId);
		}

		return undefined;
	}

	return {
		get activeId() {
			return activeId;
		},

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

		handleOpenChange,

		handleSelect,

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
			return isRouter;
		},

		get isRouter() {
			return isRouter;
		},

		get loadedItems() {
			return loadedItems;
		},

		get loading() {
			return loading;
		},

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

		get updating() {
			return updating;
		}
	};
}
