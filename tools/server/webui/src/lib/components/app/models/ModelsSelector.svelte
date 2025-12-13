<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { ChevronDown, EyeOff, Loader2, MicOff, Package, Power } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import * as Popover from '$lib/components/ui/popover';
	import { cn } from '$lib/components/ui/utils';
	import {
		modelsStore,
		modelOptions,
		modelsLoading,
		modelsUpdating,
		selectedModelId,
		routerModels,
		propsCacheVersion,
		singleModelName
	} from '$lib/stores/models.svelte';
	import { usedModalities, conversationsStore } from '$lib/stores/conversations.svelte';
	import { ServerModelStatus } from '$lib/enums';
	import { isRouterMode } from '$lib/stores/server.svelte';
	import { DialogModelInformation, SearchInput } from '$lib/components/app';
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
		/**
		 * When provided, only consider modalities from messages BEFORE this message.
		 * Used for regeneration - allows selecting models that don't support modalities
		 * used in later messages.
		 */
		upToMessageId?: string;
	}

	let {
		class: className = '',
		currentModel = null,
		onModelChange,
		disabled = false,
		forceForegroundText = false,
		useGlobalSelection = false,
		upToMessageId
	}: Props = $props();

	let options = $derived(modelOptions());
	let loading = $derived(modelsLoading());
	let updating = $derived(modelsUpdating());
	let activeId = $derived(selectedModelId());
	let isRouter = $derived(isRouterMode());
	let serverModel = $derived(singleModelName());

	// Reactive router models state - needed for proper reactivity of status checks
	let currentRouterModels = $derived(routerModels());

	let requiredModalities = $derived(
		upToMessageId ? conversationsStore.getModalitiesUpToMessage(upToMessageId) : usedModalities()
	);

	function getModelStatus(modelId: string): ServerModelStatus | null {
		const model = currentRouterModels.find((m) => m.id === modelId);
		return (model?.status?.value as ServerModelStatus) ?? null;
	}

	/**
	 * Checks if a model supports all modalities used in the conversation.
	 * Returns true if the model can be selected, false if it should be disabled.
	 */
	function isModelCompatible(option: ModelOption): boolean {
		void propsCacheVersion();

		const modelModalities = modelsStore.getModelModalities(option.model);

		if (!modelModalities) {
			const status = getModelStatus(option.model);

			if (status === ServerModelStatus.LOADED) {
				if (requiredModalities.vision || requiredModalities.audio) return false;
			}

			return true;
		}

		if (requiredModalities.vision && !modelModalities.vision) return false;
		if (requiredModalities.audio && !modelModalities.audio) return false;

		return true;
	}

	/**
	 * Gets missing modalities for a model.
	 * Returns object with vision/audio booleans indicating what's missing.
	 */
	function getMissingModalities(option: ModelOption): { vision: boolean; audio: boolean } | null {
		void propsCacheVersion();

		const modelModalities = modelsStore.getModelModalities(option.model);

		if (!modelModalities) {
			const status = getModelStatus(option.model);

			if (status === ServerModelStatus.LOADED) {
				const missing = {
					vision: requiredModalities.vision,
					audio: requiredModalities.audio
				};

				if (missing.vision || missing.audio) return missing;
			}

			return null;
		}

		const missing = {
			vision: requiredModalities.vision && !modelModalities.vision,
			audio: requiredModalities.audio && !modelModalities.audio
		};

		if (!missing.vision && !missing.audio) return null;

		return missing;
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
	let searchInputRef = $state<HTMLInputElement | null>(null);
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

	// Get indices of compatible options for keyboard navigation
	let compatibleIndices = $derived(
		filteredOptions
			.map((option, index) => (isModelCompatible(option) ? index : -1))
			.filter((i) => i !== -1)
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

	// Handle changes to the model selector pop-down or the model dialog, depending on if the server is in
	// router mode or not.
	function handleOpenChange(open: boolean) {
		if (loading || updating) return;

		if (isRouter) {
			if (open) {
				isOpen = true;
				searchTerm = '';
				highlightedIndex = -1;

				// Focus search input after popover opens
				tick().then(() => {
					requestAnimationFrame(() => searchInputRef?.focus());
				});

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

		if (event.key === 'ArrowDown') {
			event.preventDefault();
			if (compatibleIndices.length === 0) return;

			const currentPos = compatibleIndices.indexOf(highlightedIndex);
			if (currentPos === -1 || currentPos === compatibleIndices.length - 1) {
				highlightedIndex = compatibleIndices[0];
			} else {
				highlightedIndex = compatibleIndices[currentPos + 1];
			}
		} else if (event.key === 'ArrowUp') {
			event.preventDefault();
			if (compatibleIndices.length === 0) return;

			const currentPos = compatibleIndices.indexOf(highlightedIndex);
			if (currentPos === -1 || currentPos === 0) {
				highlightedIndex = compatibleIndices[compatibleIndices.length - 1];
			} else {
				highlightedIndex = compatibleIndices[currentPos - 1];
			}
		} else if (event.key === 'Enter') {
			event.preventDefault();
			if (highlightedIndex >= 0 && highlightedIndex < filteredOptions.length) {
				const option = filteredOptions[highlightedIndex];
				if (isModelCompatible(option)) {
					handleSelect(option.id);
				}
			} else if (compatibleIndices.length > 0) {
				// No selection - highlight first compatible option
				highlightedIndex = compatibleIndices[0];
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
			<Popover.Root bind:open={isOpen} onOpenChange={handleOpenChange}>
				<Popover.Trigger
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
					disabled={disabled || updating}
				>
					<Package class="h-3.5 w-3.5" />

					<span class="truncate font-medium">
						{selectedOption?.model || 'Select model'}
					</span>

					{#if updating}
						<Loader2 class="h-3 w-3.5 animate-spin" />
					{:else}
						<ChevronDown class="h-3 w-3.5" />
					{/if}
				</Popover.Trigger>

				<Popover.Content
					class="group/popover-content w-96 max-w-[calc(100vw-2rem)] p-0"
					align="end"
					sideOffset={8}
					collisionPadding={16}
				>
					<div class="flex max-h-[50dvh] flex-col overflow-hidden">
						<div
							class="order-1 shrink-0 border-b p-4 group-data-[side=top]/popover-content:order-2 group-data-[side=top]/popover-content:border-t group-data-[side=top]/popover-content:border-b-0"
						>
							<SearchInput
								id="model-search"
								placeholder="Search models..."
								bind:value={searchTerm}
								bind:ref={searchInputRef}
								onClose={() => handleOpenChange(false)}
								onKeyDown={handleSearchKeyDown}
							/>
						</div>
						<div
							class="models-list order-2 min-h-0 flex-1 overflow-y-auto group-data-[side=top]/popover-content:order-1"
						>
							{#if !isCurrentModelInCache() && currentModel}
								<!-- Show unavailable model as first option (disabled) -->
								<button
									type="button"
									class="flex w-full cursor-not-allowed items-center bg-red-400/10 px-4 py-2 text-left text-sm text-red-400"
									role="option"
									aria-selected="true"
									aria-disabled="true"
									disabled
								>
									<span class="truncate">{selectedOption?.name || currentModel}</span>
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
								{@const isCompatible = isModelCompatible(option)}
								{@const isHighlighted = index === highlightedIndex}
								{@const missingModalities = getMissingModalities(option)}

								<div
									class={cn(
										'group flex w-full items-center gap-2 px-4 py-2 text-left text-sm transition focus:outline-none',
										isCompatible
											? 'cursor-pointer hover:bg-muted focus:bg-muted'
											: 'cursor-not-allowed opacity-50',
										isSelected || isHighlighted
											? 'bg-accent text-accent-foreground'
											: isCompatible
												? 'hover:bg-accent hover:text-accent-foreground'
												: '',
										isLoaded ? 'text-popover-foreground' : 'text-muted-foreground'
									)}
									role="option"
									aria-selected={isSelected || isHighlighted}
									aria-disabled={!isCompatible}
									tabindex={isCompatible ? 0 : -1}
									onclick={() => isCompatible && handleSelect(option.id)}
									onmouseenter={() => (highlightedIndex = index)}
									onkeydown={(e) => {
										if (isCompatible && (e.key === 'Enter' || e.key === ' ')) {
											e.preventDefault();
											handleSelect(option.id);
										}
									}}
								>
									<span class="min-w-0 flex-1 truncate">{option.model}</span>

									{#if missingModalities}
										<span class="flex shrink-0 items-center gap-1 text-muted-foreground/70">
											{#if missingModalities.vision}
												<Tooltip.Root>
													<Tooltip.Trigger>
														<EyeOff class="h-3.5 w-3.5" />
													</Tooltip.Trigger>
													<Tooltip.Content class="z-[9999]">
														<p>No vision support</p>
													</Tooltip.Content>
												</Tooltip.Root>
											{/if}
											{#if missingModalities.audio}
												<Tooltip.Root>
													<Tooltip.Trigger>
														<MicOff class="h-3.5 w-3.5" />
													</Tooltip.Trigger>
													<Tooltip.Content class="z-[9999]">
														<p>No audio support</p>
													</Tooltip.Content>
												</Tooltip.Root>
											{/if}
										</span>
									{/if}

									{#if isLoading}
										<Tooltip.Root>
											<Tooltip.Trigger>
												<Loader2 class="h-4 w-4 shrink-0 animate-spin text-muted-foreground" />
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
													class="relative ml-2 flex h-4 w-4 shrink-0 items-center justify-center"
													onclick={(e) => {
														e.stopPropagation();
														modelsStore.unloadModel(option.model);
													}}
												>
													<span
														class="mr-2 h-2 w-2 rounded-full bg-green-500 transition-opacity group-hover:opacity-0"
													></span>
													<Power
														class="absolute mr-2 h-4 w-4 text-red-500 opacity-0 transition-opacity group-hover:opacity-100 hover:text-red-600"
													/>
												</button>
											</Tooltip.Trigger>
											<Tooltip.Content class="z-[9999]">
												<p>Unload model</p>
											</Tooltip.Content>
										</Tooltip.Root>
									{:else}
										<span class="mx-2 h-2 w-2 rounded-full bg-muted-foreground/50"></span>
									{/if}
								</div>
							{/each}
						</div>
					</div>
				</Popover.Content>
			</Popover.Root>
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

				<span class="truncate font-medium">
					{selectedOption?.model}
				</span>

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
