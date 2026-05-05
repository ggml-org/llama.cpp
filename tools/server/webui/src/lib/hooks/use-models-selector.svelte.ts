import { onMount } from 'svelte';
import {
	modelsStore,
	modelOptions,
	modelsLoading,
	modelsUpdating,
	selectedModelId,
	singleModelName
} from '$lib/stores/models.svelte';
import { isRouterMode } from '$lib/stores/server.svelte';
import { filterModelOptions, groupModelOptions } from '$lib/components/app/models/utils';
import type { ModelOption } from '$lib/types/models';

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
	const options = $derived(
		modelOptions().filter((option) => {
			const modelProps = modelsStore.getModelProps(option.model);
			return modelProps?.webui !== false;
		})
	);
	const loading = $derived(modelsLoading());
	const updating = $derived(modelsUpdating());
	const activeId = $derived(selectedModelId());
	const isRouter = $derived(isRouterMode());
	const serverModel = $derived(singleModelName());

	const currentModel = $derived(opts.currentModel());
	const useGlobalSelection = $derived(opts.useGlobalSelection?.() ?? false);
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
		groupModelOptions(filteredOptions, modelsStore.favoriteModelIds, (m) =>
			modelsStore.isModelLoaded(m)
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
					modelsStore.fetchModalitiesForLoadedModels();
				});
			}

			opts.onOpenChange?.(open);
		} else {
			showModelDialog = open;
		}
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
			await modelsStore.selectModelById(option.id);
		}

		if (shouldCloseMenu) {
			handleOpenChange(false);

			requestAnimationFrame(() => {
				const textarea = document.querySelector<HTMLTextAreaElement>(
					'[data-slot="chat-form"] textarea'
				);
				textarea?.focus();
			});
		}

		if (!onModelChange && isRouter && !modelsStore.isModelLoaded(option.model)) {
			isLoadingModel = true;
			modelsStore
				.loadModel(option.model)
				.catch((error) => console.error('Failed to load model:', error))
				.finally(() => (isLoadingModel = false));
		}
	}

	function getDisplayOption(): ModelOption | undefined {
		if (!isRouter) {
			const displayModel = serverModel || currentModel;
			if (displayModel) {
				return {
					id: serverModel ? 'current' : 'offline-current',
					model: displayModel,
					name: displayModel.split('/').pop() || displayModel,
					capabilities: []
				};
			}
			return undefined;
		}

		if (useGlobalSelection && activeId) {
			const selected = options.find((option) => option.id === activeId);
			if (selected) return selected;
		}

		if (currentModel) {
			if (!isCurrentModelInCache) {
				return {
					id: 'not-in-cache',
					model: currentModel,
					name: currentModel.split('/').pop() || currentModel,
					capabilities: []
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
		get options() {
			return options;
		},
		get loading() {
			return loading;
		},
		get updating() {
			return updating;
		},
		get activeId() {
			return activeId;
		},
		get isRouter() {
			return isRouter;
		},
		get serverModel() {
			return serverModel;
		},
		get isHighlightedCurrentModelActive() {
			return isHighlightedCurrentModelActive;
		},
		get isCurrentModelInCache() {
			return isCurrentModelInCache;
		},
		get filteredOptions() {
			return filteredOptions;
		},
		get groupedFilteredOptions() {
			return groupedFilteredOptions;
		},
		get isLoadingModel() {
			return isLoadingModel;
		},
		get searchTerm() {
			return searchTerm;
		},
		get showModelDialog() {
			return showModelDialog;
		},
		get infoModelId() {
			return infoModelId;
		},
		setSearchTerm(value: string) {
			searchTerm = value;
		},
		setShowModelDialog(value: boolean) {
			showModelDialog = value;
		},
		handleInfoClick,
		handleSelect,
		handleOpenChange,
		isFavorite(model: string) {
			return modelsStore.favoriteModelIds.has(model);
		},
		getDisplayOption
	};
}
