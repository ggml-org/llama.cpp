import { filterModelOptions, groupModelOptions } from '$lib/components/app/models/utils';
import { CHAT_INPUT_FOCUS_SELECTOR } from '$lib/constants';
import { HuggingFaceService } from '$lib/services';
import { modelsStore, serverStore } from '$lib/stores';
import type { ModelOption } from '$lib/types/models';
import { onMount } from 'svelte';
import { SvelteMap, SvelteSet } from 'svelte/reactivity';

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
		modelsStore.models.filter((option) => {
			const modelProps = modelsStore.props.getModelProps(option.model);

			return modelProps?.ui !== false;
		})
	);
	const loading = $derived(modelsStore.loading);
	const updating = $derived(modelsStore.updating);
	const activeId = $derived(modelsStore.selectedModelId);
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
	let menuOpen = $state(false);

	// Base model org per base repo (e.g. `Qwen` for `ggml-org/Qwen3.8-27B-GGUF`),
	// resolved lazily from the HF card while the menu is open.
	const baseOrgs = new SvelteMap<string, string>();
	const filteredOptions = $derived(filterModelOptions(options, searchTerm));
	// Augment each option with its base model; the org falls back to the quant
	// org until the HF lookup resolves.
	const optionsWithBaseModel = $derived(
		filteredOptions.map((option) => {
			const repo = option.model.split(':')[0];
			const org = baseOrgs.get(repo) ?? option.parsedId?.orgName ?? repo.split('/')[0] ?? '';
			const name = option.parsedId?.modelName ?? option.name ?? option.model;

			return { ...option, baseModel: { name, org } };
		})
	);
	const groupedFilteredOptions = $derived(
		groupModelOptions(optionsWithBaseModel, modelsStore.favoriteModelIds, (m) =>
			modelsStore.isModelLoaded(m)
		)
	);

	// Fetch base model orgs for the visible repos while the menu is open. The
	// service caches per repo, so repeated opens never re-hit the HF API.
	$effect(() => {
		if (!menuOpen) return;

		const repos = new SvelteSet<string>();

		for (const option of options) repos.add(option.model.split(':')[0]);

		for (const repo of repos) {
			if (baseOrgs.has(repo)) continue;

			void HuggingFaceService.getBaseModel(repo).then((info) => {
				baseOrgs.set(repo, info?.org ?? repo.split('/')[0] ?? '');
			});
		}
	});

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

		menuOpen = open;

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

		isFavorite(model: string) {
			return modelsStore.favoriteModelIds.has(model);
		},

		get isHighlightedCurrentModelActive() {
			return isHighlightedCurrentModelActive;
		},

		get isLoadingModel() {
			return isLoadingModel;
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
