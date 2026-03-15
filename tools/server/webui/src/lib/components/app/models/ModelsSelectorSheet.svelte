<script lang="ts">
	import { onMount } from 'svelte';
	import { SvelteMap } from 'svelte/reactivity';
	import { ChevronDown, Loader2, Package } from '@lucide/svelte';
	import * as Sheet from '$lib/components/ui/sheet';
	import { cn } from '$lib/components/ui/utils';
	import {
		modelsStore,
		modelOptions,
		modelsLoading,
		modelsUpdating,
		selectedModelId,
		singleModelName
	} from '$lib/stores/models.svelte';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import {
		DialogModelInformation,
		SearchInput,
		TruncatedText,
		ModelsSelectorOption
	} from '$lib/components/app';
	import type { ModelOption } from '$lib/types/models';

	interface Props {
		class?: string;
		currentModel?: string | null;
		/** Callback when model changes. Return false to keep menu open (e.g., for validation failures) */
		onModelChange?: (modelId: string, modelName: string) => Promise<boolean> | boolean | void;
		disabled?: boolean;
		forceForegroundText?: boolean;
		/** When true, user's global selection takes priority over currentModel (for form selector) */
		useGlobalSelection?: boolean;
	}

	let {
		class: className = '',
		currentModel = null,
		onModelChange,
		disabled = false,
		forceForegroundText = false,
		useGlobalSelection = false
	}: Props = $props();

	let options = $derived(
		modelOptions().filter((option) => {
			const modelProps = modelsStore.getModelProps(option.model);
			return modelProps?.webui !== false;
		})
	);
	let loading = $derived(modelsLoading());
	let updating = $derived(modelsUpdating());
	let activeId = $derived(selectedModelId());
	let isRouter = $derived(isRouterMode());
	let serverModel = $derived(singleModelName());

	let isLoadingModel = $state(false);

	let isHighlightedCurrentModelActive = $derived(
		!isRouter || !currentModel
			? false
			: (() => {
					const currentOption = options.find((option) => option.model === currentModel);

					return currentOption ? currentOption.id === activeId : false;
				})()
	);

	let isCurrentModelInCache = $derived.by(() => {
		if (!isRouter || !currentModel) return true;

		return options.some((option) => option.model === currentModel);
	});

	let searchTerm = $state('');

	let filteredOptions: ModelOption[] = $derived.by(() => {
		const term = searchTerm.trim().toLowerCase();
		if (!term) return options;

		return options.filter(
			(option) =>
				option.model.toLowerCase().includes(term) ||
				option.name?.toLowerCase().includes(term) ||
				option.aliases?.some((alias: string) => alias.toLowerCase().includes(term)) ||
				option.tags?.some((tag: string) => tag.toLowerCase().includes(term))
		);
	});

	let groupedFilteredOptions = $derived.by(() => {
		const favIds = modelsStore.favouriteModelIds;
		const result: {
			orgName: string | null;
			isFavouritesGroup: boolean;
			isLoadedGroup: boolean;
			items: { option: ModelOption; flatIndex: number }[];
		}[] = [];

		// Loaded models group (top)
		const loadedItems: { option: ModelOption; flatIndex: number }[] = [];
		for (let i = 0; i < filteredOptions.length; i++) {
			if (modelsStore.isModelLoaded(filteredOptions[i].model)) {
				loadedItems.push({ option: filteredOptions[i], flatIndex: i });
			}
		}
		if (loadedItems.length > 0) {
			result.push({
				orgName: null,
				isFavouritesGroup: false,
				isLoadedGroup: true,
				items: loadedItems
			});
		}

		// Favourites group
		const loadedModelIds = new Set(loadedItems.map((item) => item.option.model));
		const favItems: { option: ModelOption; flatIndex: number }[] = [];
		for (let i = 0; i < filteredOptions.length; i++) {
			if (favIds.has(filteredOptions[i].model) && !loadedModelIds.has(filteredOptions[i].model)) {
				favItems.push({ option: filteredOptions[i], flatIndex: i });
			}
		}
		if (favItems.length > 0) {
			result.push({
				orgName: null,
				isFavouritesGroup: true,
				isLoadedGroup: false,
				items: favItems
			});
		}

		// Org groups (excluding loaded and favourites)
		const orgGroups = new SvelteMap<string, { option: ModelOption; flatIndex: number }[]>();
		for (let i = 0; i < filteredOptions.length; i++) {
			const option = filteredOptions[i];
			if (loadedModelIds.has(option.model) || favIds.has(option.model)) continue;
			const orgName = option.parsedId?.orgName ?? null;
			const key = orgName ?? '';
			if (!orgGroups.has(key)) orgGroups.set(key, []);
			orgGroups.get(key)!.push({ option, flatIndex: i });
		}
		for (const [orgName, items] of orgGroups) {
			result.push({
				orgName: orgName || null,
				isFavouritesGroup: false,
				isLoadedGroup: false,
				items
			});
		}

		return result;
	});

	let sheetOpen = $state(false);
	let showModelDialog = $state(false);
	let infoModelId = $state<string | null>(null);

	// key of the first "available" (non-loaded, non-favourite) group
	// used to render the "Available models" separator exactly once
	let firstAvailableOrgKey = $derived.by(() => {
		const g = groupedFilteredOptions.find((g) => !g.isLoadedGroup && !g.isFavouritesGroup);
		return g ? (g.orgName ?? '') : null;
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

		if (isRouter) {
			if (open) {
				sheetOpen = true;
				searchTerm = '';

				modelsStore.fetchRouterModels().then(() => {
					modelsStore.fetchModalitiesForLoadedModels();
				});
			} else {
				sheetOpen = false;
				searchTerm = '';
			}
		} else {
			showModelDialog = open;
		}
	}

	export function open() {
		handleOpenChange(true);
	}

	function handleSheetOpenChange(open: boolean) {
		if (!open) {
			handleOpenChange(false);
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
			if (serverModel) {
				return {
					id: 'current',
					model: serverModel,
					name: serverModel.split('/').pop() || serverModel,
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
</script>

<div class={cn('relative inline-flex flex-col items-end gap-1', className)}>
	{#if loading && options.length === 0 && isRouter}
		<div class="flex items-center gap-2 text-xs text-muted-foreground">
			<Loader2 class="h-3.5 w-3.5 animate-spin" />
			Loading models…
		</div>
	{:else if options.length === 0 && isRouter}
		<p class="text-xs text-muted-foreground">No models available.</p>
	{:else}
		{@const selectedOption = getDisplayOption()}

		{#if isRouter}
			<button
				type="button"
				class={cn(
					`inline-grid cursor-pointer grid-cols-[1fr_auto_1fr] items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60`,
					!isCurrentModelInCache
						? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
						: forceForegroundText
							? 'text-foreground'
							: isHighlightedCurrentModelActive
								? 'text-foreground'
								: 'text-muted-foreground',
					sheetOpen ? 'text-foreground' : ''
				)}
				style="max-width: min(calc(100cqw - 9rem), 20rem)"
				disabled={disabled || updating}
				onclick={() => handleOpenChange(true)}
			>
				<Package class="h-3.5 w-3.5" />

				<TruncatedText text={selectedOption?.model || 'Select model'} class="min-w-0 font-medium" />

				{#if updating || isLoadingModel}
					<Loader2 class="h-3 w-3.5 animate-spin" />
				{:else}
					<ChevronDown class="h-3 w-3.5" />
				{/if}
			</button>

			<Sheet.Root bind:open={sheetOpen} onOpenChange={handleSheetOpenChange}>
				<Sheet.Content side="bottom" class="max-h-[85vh] gap-1">
					<Sheet.Header>
						<Sheet.Title>Select Model</Sheet.Title>

						<Sheet.Description class="sr-only">
							Choose a model to use for the conversation
						</Sheet.Description>
					</Sheet.Header>

					<div class="flex flex-col gap-1 pb-4">
						<div class="mb-3 px-4">
							<SearchInput placeholder="Search models..." bind:value={searchTerm} />
						</div>

						<div class="max-h-[60vh] overflow-y-auto px-2">
							{#if !isCurrentModelInCache && currentModel}
								<button
									type="button"
									class="flex w-full cursor-not-allowed items-center rounded-md bg-red-400/10 px-3 py-2.5 text-left text-sm text-red-400"
									disabled
								>
									<span class="min-w-0 flex-1 truncate">
										{selectedOption?.name || currentModel}
									</span>
									<span class="ml-2 text-xs whitespace-nowrap opacity-70">(not available)</span>
								</button>
								<div class="my-1 h-px bg-border"></div>
							{/if}

							{#if filteredOptions.length === 0}
								<p class="px-3 py-3 text-center text-sm text-muted-foreground">No models found.</p>
							{/if}

							{#each groupedFilteredOptions as group (group.isLoadedGroup ? '__loaded__' : group.isFavouritesGroup ? '__favourites__' : group.orgName)}
								{#if group.isLoadedGroup}
									<p class="px-2 py-2 text-xs font-semibold text-muted-foreground/60 select-none">
										Loaded models
									</p>
								{:else if group.isFavouritesGroup}
									<p class="px-2 py-2 text-xs font-semibold text-muted-foreground/60 select-none">
										Favourite models
									</p>
								{:else}
									{#if (group.orgName ?? '') === firstAvailableOrgKey}
										<p class="px-2 py-2 text-xs font-semibold text-muted-foreground/60 select-none">
											Available models
										</p>
									{/if}
									{#if group.orgName}
										<p
											class="px-2 py-2 text-xs font-semibold text-muted-foreground/60 select-none [&:not(:first-child)]:mt-2"
										>
											{group.orgName}
										</p>
									{/if}
								{/if}

								{#each group.items as { option } (group.isLoadedGroup ? `loaded-${option.id}` : group.isFavouritesGroup ? `fav-${option.id}` : option.id)}
									{@const isSelected = currentModel === option.model || activeId === option.id}
									{@const isFav = modelsStore.favouriteModelIds.has(option.model)}
									<ModelsSelectorOption
										{option}
										{isSelected}
										isHighlighted={false}
										{isFav}
										showOrgName={group.isFavouritesGroup || group.isLoadedGroup}
										onSelect={handleSelect}
										onInfoClick={handleInfoClick}
										onMouseEnter={() => {}}
										onKeyDown={() => {}}
									/>
								{/each}
							{/each}
						</div>
					</div>
				</Sheet.Content>
			</Sheet.Root>
		{:else}
			<button
				class={cn(
					`inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60`,
					!isCurrentModelInCache
						? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
						: forceForegroundText
							? 'text-foreground'
							: isHighlightedCurrentModelActive
								? 'text-foreground'
								: 'text-muted-foreground'
				)}
				style="max-width: min(calc(100cqw - 6.5rem), 32rem)"
				onclick={() => handleOpenChange(true)}
				disabled={disabled || updating}
			>
				<Package class="h-3.5 w-3.5" />

				<TruncatedText text={selectedOption?.model || ''} class="min-w-0 font-medium" />

				{#if updating}
					<Loader2 class="h-3 w-3.5 animate-spin" />
				{/if}
			</button>
		{/if}
	{/if}
</div>

{#if showModelDialog}
	<DialogModelInformation bind:open={showModelDialog} modelId={infoModelId} />
{/if}
