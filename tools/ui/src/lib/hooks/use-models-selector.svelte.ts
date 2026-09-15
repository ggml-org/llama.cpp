import { filterModelOptions, groupModelOptions } from '$lib/components/app/navigation/utils';
import { CHAT_INPUT_FOCUS_SELECTOR, LOCAL_BACKEND_ID } from '$lib/constants';
import { backendsModelsStore, backendsStore, modelsStore, serverStore } from '$lib/stores';
import type { Backend } from '$lib/types';
import type { ModelOption } from '$lib/types/models';
import { rawModelId } from '$lib/utils/model-option-id';
import { onMount } from 'svelte';

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
	readonly activeBackendId: string;
	readonly backends: Backend[];
	readonly isMultiModel: boolean;
	readonly isRouter: boolean;
	readonly serverModel: string | null;
	readonly isHighlightedCurrentModelActive: boolean;
	readonly isCurrentModelInCache: boolean;
	readonly filteredOptions: ModelOption[];
	readonly groupedFilteredOptions: ReturnType<typeof groupModelOptions>;
	readonly isLoadingModel: boolean;
	readonly searchTerm: string;
	readonly showModelDialog: boolean;
	readonly infoModelId: string | null;
	setSearchTerm(value: string): void;
	setShowModelDialog(value: boolean): void;
	handleInfoClick(modelName: string): void;
	handleBackendChange(backendId: string): Promise<void>;
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
	const activeBackendId = $derived(backendsStore.active.id);
	// every enabled backend's models are selectable; the switcher only picks
	// the backend new requests target
	const options = $derived(
		modelsStore.models.filter((option) => {
			const modelProps = modelsStore.props.getModelProps(option.model);

			return modelProps?.ui !== false;
		})
	);
	const loading = $derived(modelsStore.loading);
	const updating = $derived(modelsStore.updating);
	const activeId = $derived(modelsStore.selectedModelId);
	const backends = $derived(backendsStore.enabled);
	// Router mode and external backends both expose a selectable model list;
	// a single-model llama.cpp server does not.
	const isMultiModel = $derived(serverStore.isRouterMode || !serverStore.capabilities.props);
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

	let isLoadingModel = $state(false);
	let searchTerm = $state('');
	let showModelDialog = $state(false);
	let infoModelId = $state<string | null>(null);

	const filteredOptions = $derived(filterModelOptions(options, searchTerm));
	const groupedFilteredOptions = $derived(
		groupModelOptions(
			filteredOptions,
			modelsStore.favoriteModelIds,
			(m) => modelsStore.isModelLoaded(m),
			(option) => option.backendId === LOCAL_BACKEND_ID
		)
	);

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

		if (isRouter) {
			searchTerm = '';

			if (open) {
				modelsStore.fetchRouterModels().then(() => {
					modelsStore.props.fetchModalitiesForLoadedModels();
				});
			}

			opts.onOpenChange?.(open);
		} else {
			showModelDialog = open;
		}
	}

	async function handleBackendChange(backendId: string) {
		if (backendId === backendsStore.active.id) return;

		backendsStore.setActive(backendId);
		searchTerm = '';

		try {
			await backendsModelsStore.ensureLoaded(backendId);
			await modelsStore.switchBackend();
		} catch (error) {
			console.error('Failed to switch backend:', error);
		}
	}

	async function handleSelect(modelId: string) {
		const option = options.find((opt) => opt.id === modelId);

		if (!option) return;

		let shouldCloseMenu = true;

		if (onModelChange) {
			const result = await onModelChange(rawModelId(option.id), option.model);

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
			// External backend: the selection is backend-scoped, so it wins over
			// the conversation's model, which may belong to another backend.
			if (!serverStore.capabilities.props) {
				const selected = activeId ? options.find((option) => option.id === activeId) : undefined;

				if (selected) return selected;

				return currentModel ? options.find((option) => option.model === currentModel) : undefined;
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

			return options.find((option) => option.model === currentModel);
		}

		if (activeId) {
			return options.find((option) => option.id === activeId);
		}

		return undefined;
	}

	return {
		get activeBackendId() {
			return activeBackendId;
		},

		get activeId() {
			return activeId;
		},

		get backends() {
			return backends;
		},

		get filteredOptions() {
			return filteredOptions;
		},

		getDisplayOption,

		get groupedFilteredOptions() {
			return groupedFilteredOptions;
		},

		handleBackendChange,

		handleInfoClick,

		handleOpenChange,

		handleSelect,

		get infoModelId() {
			return infoModelId;
		},

		get isCurrentModelInCache() {
			return isCurrentModelInCache;
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

		get isRouter() {
			return isRouter;
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

		setShowModelDialog(value: boolean) {
			showModelDialog = value;
		},

		get showModelDialog() {
			return showModelDialog;
		},

		get updating() {
			return updating;
		}
	};
}
