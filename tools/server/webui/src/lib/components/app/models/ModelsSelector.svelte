<script lang="ts">
	import { onMount } from 'svelte';
	import { ChevronDown, Loader2, Package, Power } from '@lucide/svelte';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { cn } from '$lib/components/ui/utils';
	import {
		modelsStore,
		modelOptions,
		modelsLoading,
		modelsUpdating,
		selectedModelId,
		routerModels,
		singleModelName
	} from '$lib/stores/models.svelte';
	import { KeyboardKey, ServerModelStatus } from '$lib/enums';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import {
		DialogModelInformation,
		DropdownMenuSearchable,
		TruncatedText
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

	// Reactive router models state - needed for proper reactivity of status checks
	let currentRouterModels = $derived(routerModels());

	function getModelStatus(modelId: string): ServerModelStatus | null {
		const model = currentRouterModels.find((m) => m.id === modelId);
		return (model?.status?.value as ServerModelStatus) ?? null;
	}

	let isHighlightedCurrentModelActive = $derived(
		!isRouter || !currentModel
			? false
			: (() => {
					const currentOption = options.find((option) => option.model === currentModel);

					return currentOption ? currentOption.id === activeId : false;
				})()
	);

	let isCurrentModelInCache = $derived(() => {
		if (!isRouter || !currentModel) return true;

		return options.some((option) => option.model === currentModel);
	});

	let searchTerm = $state('');
	let highlightedIndex = $state<number>(-1);

	let filteredOptions: ModelOption[] = $derived(
		(() => {
			const term = searchTerm.trim().toLowerCase();
			if (!term) return options;

			return options.filter(
				(option) =>
					option.model.toLowerCase().includes(term) || option.name?.toLowerCase().includes(term)
			);
		})()
	);

	// Reset highlighted index when search term changes
	$effect(() => {
		void searchTerm;
		highlightedIndex = -1;
	});

	let isOpen = $state(false);
	let showModelDialog = $state(false);

	onMount(() => {
		modelsStore.fetch().catch((error) => {
			console.error('Unable to load models:', error);
		});
	});

	// Handle changes to the model selector dropdown or the model dialog, depending on if the server is in
	// router mode or not.
	function handleOpenChange(open: boolean) {
		if (loading || updating) return;

		if (isRouter) {
			if (open) {
				isOpen = true;
				searchTerm = '';
				highlightedIndex = -1;

				modelsStore.fetchRouterModels().then(() => {
					modelsStore.fetchModalitiesForLoadedModels();
				});
			} else {
				isOpen = false;
				searchTerm = '';
				highlightedIndex = -1;
			}
		} else {
			showModelDialog = open;
		}
	}

	export function open() {
		handleOpenChange(true);
	}

	function handleSearchKeyDown(event: KeyboardEvent) {
		if (event.isComposing) return;

		if (event.key === KeyboardKey.ARROW_DOWN) {
			event.preventDefault();
			if (filteredOptions.length === 0) return;

			if (highlightedIndex === -1 || highlightedIndex === filteredOptions.length - 1) {
				highlightedIndex = 0;
			} else {
				highlightedIndex += 1;
			}
		} else if (event.key === KeyboardKey.ARROW_UP) {
			event.preventDefault();
			if (filteredOptions.length === 0) return;

			if (highlightedIndex === -1 || highlightedIndex === 0) {
				highlightedIndex = filteredOptions.length - 1;
			} else {
				highlightedIndex -= 1;
			}
		} else if (event.key === KeyboardKey.ENTER) {
			event.preventDefault();
			if (highlightedIndex >= 0 && highlightedIndex < filteredOptions.length) {
				const option = filteredOptions[highlightedIndex];
				handleSelect(option.id);
			} else if (filteredOptions.length > 0) {
				// No selection - highlight first option
				highlightedIndex = 0;
			}
		}
	}

	async function handleSelect(modelId: string) {
		const option = options.find((opt) => opt.id === modelId);
		if (!option) return;

		let shouldCloseMenu = true;

		if (onModelChange) {
			// If callback provided, use it (for regenerate functionality)
			const result = await onModelChange(option.id, option.model);

			// If callback returns false, keep menu open (validation failed)
			if (result === false) {
				shouldCloseMenu = false;
			}
		} else {
			// Update global selection
			await modelsStore.selectModelById(option.id);

			// Load the model if not already loaded (router mode)
			if (isRouter && getModelStatus(option.model) !== ServerModelStatus.LOADED) {
				try {
					await modelsStore.loadModel(option.model);
				} catch (error) {
					console.error('Failed to load model:', error);
				}
			}
		}

		if (shouldCloseMenu) {
			handleOpenChange(false);

			// Focus the chat textarea after model selection
			requestAnimationFrame(() => {
				const textarea = document.querySelector<HTMLTextAreaElement>(
					'[data-slot="chat-form"] textarea'
				);
				textarea?.focus();
			});
		}
	}

	function getDisplayOption(): ModelOption | undefined {
		if (!isRouter) {
			if (serverModel) {
				return {
					id: 'current',
					model: serverModel,
					name: serverModel.split('/').pop() || serverModel,
					capabilities: [] // Empty array for single model mode
				};
			}

			return undefined;
		}

		// When useGlobalSelection is true (form selector), prioritize user selection
		// Otherwise (message display), prioritize currentModel
		if (useGlobalSelection && activeId) {
			const selected = options.find((option) => option.id === activeId);
			if (selected) return selected;
		}

		// Show currentModel (from message payload or conversation)
		if (currentModel) {
			if (!isCurrentModelInCache()) {
				return {
					id: 'not-in-cache',
					model: currentModel,
					name: currentModel.split('/').pop() || currentModel,
					capabilities: []
				};
			}

			return options.find((option) => option.model === currentModel);
		}

		// Fallback to user selection (for new chats before first message)
		if (activeId) {
			return options.find((option) => option.id === activeId);
		}

		// No selection - return undefined to show "Select model"
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
			<DropdownMenu.Root bind:open={isOpen} onOpenChange={handleOpenChange}>
				<DropdownMenu.Trigger
					disabled={disabled || updating}
					onclick={(e) => {
						e.preventDefault();
						e.stopPropagation();
					}}
				>
					<button
						type="button"
						class={cn(
							`inline-grid cursor-pointer grid-cols-[1fr_auto_1fr] items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60`,
							!isCurrentModelInCache()
								? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
								: forceForegroundText
									? 'text-foreground'
									: isHighlightedCurrentModelActive
										? 'text-foreground'
										: 'text-muted-foreground',
							isOpen ? 'text-foreground' : ''
						)}
						style="max-width: min(calc(100cqw - 9rem), 20rem)"
						disabled={disabled || updating}
					>
						<Package class="h-3.5 w-3.5" />

						<TruncatedText
							text={selectedOption?.model || 'Select model'}
							class="min-w-0 font-medium"
						/>

						{#if updating}
							<Loader2 class="h-3 w-3.5 animate-spin" />
						{:else}
							<ChevronDown class="h-3 w-3.5" />
						{/if}
					</button>
				</DropdownMenu.Trigger>

				<DropdownMenu.Content
					align="end"
					class="w-full max-w-[100vw] pt-0 sm:w-max sm:max-w-[calc(100vw-2rem)]"
				>
					<DropdownMenuSearchable
						bind:searchValue={searchTerm}
						placeholder="Search models..."
						onSearchKeyDown={handleSearchKeyDown}
						emptyMessage="No models found."
						isEmpty={filteredOptions.length === 0 && isCurrentModelInCache()}
					>
						<div class="models-list">
							{#if !isCurrentModelInCache() && currentModel}
								<!-- Show unavailable model as first option (disabled) -->
								<button
									type="button"
									class="flex w-full cursor-not-allowed items-center bg-red-400/10 p-2 text-left text-sm text-red-400"
									role="option"
									aria-selected="true"
									aria-disabled="true"
									disabled
								>
									<span
										class="min-w-0 flex-1 truncate text-left sm:overflow-visible sm:text-clip sm:whitespace-nowrap"
									>
										{selectedOption?.name || currentModel}
									</span>
									<span class="ml-2 text-xs whitespace-nowrap opacity-70">(not available)</span>
								</button>
								<div class="my-1 h-px bg-border"></div>
							{/if}
							{#if filteredOptions.length === 0}
								<p class="px-4 py-3 text-sm text-muted-foreground">No models found.</p>
							{/if}
							{#each filteredOptions as option, index (option.id)}
								{@const status = getModelStatus(option.model)}
								{@const isLoaded = status === ServerModelStatus.LOADED}
								{@const isLoading = status === ServerModelStatus.LOADING}
								{@const isSelected = currentModel === option.model || activeId === option.id}
								{@const isHighlighted = index === highlightedIndex}

								<div
									class={cn(
										'group flex w-full items-center gap-2 rounded-sm p-2 text-left text-sm transition focus:outline-none',
										'cursor-pointer hover:bg-muted focus:bg-muted',
										isSelected || isHighlighted
											? 'bg-accent text-accent-foreground'
											: 'hover:bg-accent hover:text-accent-foreground',
										isLoaded ? 'text-popover-foreground' : 'text-muted-foreground'
									)}
									role="option"
									aria-selected={isSelected || isHighlighted}
									tabindex="0"
									onclick={() => handleSelect(option.id)}
									onmouseenter={() => (highlightedIndex = index)}
									onkeydown={(e) => {
										if (e.key === 'Enter' || e.key === ' ') {
											e.preventDefault();
											handleSelect(option.id);
										}
									}}
								>
									<span
										class="min-w-0 flex-1 truncate text-left sm:overflow-visible sm:pr-2 sm:text-clip sm:whitespace-nowrap"
									>
										{option.model}
									</span>

									<div class="flex w-6 shrink-0 justify-center">
										{#if isLoading}
											<Tooltip.Root>
												<Tooltip.Trigger>
													<Loader2 class="h-4 w-4 animate-spin text-muted-foreground" />
												</Tooltip.Trigger>
												<Tooltip.Content class="z-[9999]">
													<p>Loading model...</p>
												</Tooltip.Content>
											</Tooltip.Root>
										{:else if isLoaded}
											<Tooltip.Root>
												<Tooltip.Trigger>
													<button
														type="button"
														class="relative flex h-4 w-4 items-center justify-center"
														onclick={(e) => {
															e.stopPropagation();
															modelsStore.unloadModel(option.model);
														}}
													>
														<span
															class="h-2 w-2 rounded-full bg-green-500 transition-opacity group-hover:opacity-0"
														></span>
														<Power
															class="absolute h-4 w-4 text-red-500 opacity-0 transition-opacity group-hover:opacity-100 hover:text-red-600"
														/>
													</button>
												</Tooltip.Trigger>
												<Tooltip.Content class="z-[9999]">
													<p>Unload model</p>
												</Tooltip.Content>
											</Tooltip.Root>
										{:else}
											<span class="h-2 w-2 rounded-full bg-muted-foreground/50"></span>
										{/if}
									</div>
								</div>
							{/each}
						</div>
					</DropdownMenuSearchable>
				</DropdownMenu.Content>
			</DropdownMenu.Root>
		{:else}
			<button
				class={cn(
					`inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60`,
					!isCurrentModelInCache()
						? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
						: forceForegroundText
							? 'text-foreground'
							: isHighlightedCurrentModelActive
								? 'text-foreground'
								: 'text-muted-foreground',
					isOpen ? 'text-foreground' : ''
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

{#if showModelDialog && !isRouter}
	<DialogModelInformation bind:open={showModelDialog} />
{/if}
