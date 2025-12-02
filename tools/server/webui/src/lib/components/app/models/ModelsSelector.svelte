<script lang="ts">
	import { onMount, tick } from 'svelte';
	import { ChevronDown, EyeOff, Loader2, MicOff, Package, Power } from '@lucide/svelte';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { cn } from '$lib/components/ui/utils';
	import { portalToBody } from '$lib/utils';
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
	import { DialogModelInformation } from '$lib/components/app';
	import {
		MENU_MAX_WIDTH,
		MENU_OFFSET,
		VIEWPORT_GUTTER
	} from '$lib/constants/floating-ui-constraints';

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

	let isOpen = $state(false);
	let showModelDialog = $state(false);
	let container: HTMLDivElement | null = null;
	let menuRef = $state<HTMLDivElement | null>(null);
	let triggerButton = $state<HTMLButtonElement | null>(null);
	let menuPosition = $state<{
		top: number;
		left: number;
		width: number;
		placement: 'top' | 'bottom';
		maxHeight: number;
	} | null>(null);

	onMount(async () => {
		try {
			await modelsStore.fetch();
		} catch (error) {
			console.error('Unable to load models:', error);
		}
	});

	function toggleOpen() {
		if (loading || updating) return;

		if (isRouter) {
			// Router mode: show dropdown
			if (isOpen) {
				closeMenu();
			} else {
				openMenu();
			}
		} else {
			// Single model mode: show dialog
			showModelDialog = true;
		}
	}

	async function openMenu() {
		if (loading || updating) return;

		isOpen = true;
		await tick();
		updateMenuPosition();
		requestAnimationFrame(() => updateMenuPosition());

		if (isRouter) {
			modelsStore.fetchRouterModels().then(() => {
				modelsStore.fetchModalitiesForLoadedModels();
			});
		}
	}

	export function open() {
		if (isRouter) {
			openMenu();
		} else {
			showModelDialog = true;
		}
	}

	function closeMenu() {
		if (!isOpen) return;

		isOpen = false;
		menuPosition = null;
	}

	function handlePointerDown(event: PointerEvent) {
		if (!container) return;

		const target = event.target as Node | null;

		if (target && !container.contains(target) && !(menuRef && menuRef.contains(target))) {
			closeMenu();
		}
	}

	function handleKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			closeMenu();
		}
	}

	function handleResize() {
		if (isOpen) {
			updateMenuPosition();
		}
	}

	function updateMenuPosition() {
		if (!isOpen || !triggerButton || !menuRef) return;

		const triggerRect = triggerButton.getBoundingClientRect();
		const viewportWidth = window.innerWidth;
		const viewportHeight = window.innerHeight;

		if (viewportWidth === 0 || viewportHeight === 0) return;

		const scrollWidth = menuRef.scrollWidth;
		const scrollHeight = menuRef.scrollHeight;

		const availableWidth = Math.max(0, viewportWidth - VIEWPORT_GUTTER * 2);
		const constrainedMaxWidth = Math.min(MENU_MAX_WIDTH, availableWidth || MENU_MAX_WIDTH);
		const safeMaxWidth =
			constrainedMaxWidth > 0 ? constrainedMaxWidth : Math.min(MENU_MAX_WIDTH, viewportWidth);
		const desiredMinWidth = Math.min(160, safeMaxWidth || 160);

		let width = Math.min(
			Math.max(triggerRect.width, scrollWidth, desiredMinWidth),
			safeMaxWidth || 320
		);

		const availableBelow = Math.max(
			0,
			viewportHeight - VIEWPORT_GUTTER - triggerRect.bottom - MENU_OFFSET
		);
		const availableAbove = Math.max(0, triggerRect.top - VIEWPORT_GUTTER - MENU_OFFSET);
		const viewportAllowance = Math.max(0, viewportHeight - VIEWPORT_GUTTER * 2);
		const fallbackAllowance = Math.max(1, viewportAllowance > 0 ? viewportAllowance : scrollHeight);

		function computePlacement(placement: 'top' | 'bottom') {
			const available = placement === 'bottom' ? availableBelow : availableAbove;
			const allowedHeight =
				available > 0 ? Math.min(available, fallbackAllowance) : fallbackAllowance;
			const maxHeight = Math.min(scrollHeight, allowedHeight);
			const height = Math.max(0, maxHeight);

			let top: number;
			if (placement === 'bottom') {
				const rawTop = triggerRect.bottom + MENU_OFFSET;
				const minTop = VIEWPORT_GUTTER;
				const maxTop = viewportHeight - VIEWPORT_GUTTER - height;
				if (maxTop < minTop) {
					top = minTop;
				} else {
					top = Math.min(Math.max(rawTop, minTop), maxTop);
				}
			} else {
				const rawTop = triggerRect.top - MENU_OFFSET - height;
				const minTop = VIEWPORT_GUTTER;
				const maxTop = viewportHeight - VIEWPORT_GUTTER - height;
				if (maxTop < minTop) {
					top = minTop;
				} else {
					top = Math.max(Math.min(rawTop, maxTop), minTop);
				}
			}

			return { placement, top, height, maxHeight };
		}

		const belowMetrics = computePlacement('bottom');
		const aboveMetrics = computePlacement('top');

		let metrics = belowMetrics;
		if (scrollHeight > belowMetrics.maxHeight && aboveMetrics.maxHeight > belowMetrics.maxHeight) {
			metrics = aboveMetrics;
		}

		let left = triggerRect.right - width;
		const maxLeft = viewportWidth - VIEWPORT_GUTTER - width;
		if (maxLeft < VIEWPORT_GUTTER) {
			left = VIEWPORT_GUTTER;
		} else {
			if (left > maxLeft) {
				left = maxLeft;
			}
			if (left < VIEWPORT_GUTTER) {
				left = VIEWPORT_GUTTER;
			}
		}

		menuPosition = {
			top: Math.round(metrics.top),
			left: Math.round(left),
			width: Math.round(width),
			placement: metrics.placement,
			maxHeight: Math.round(metrics.maxHeight)
		};
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
			closeMenu();
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

<svelte:window onresize={handleResize} />
<svelte:document onpointerdown={handlePointerDown} onkeydown={handleKeydown} />

<div class={cn('relative inline-flex flex-col items-end gap-1', className)} bind:this={container}>
	{#if loading && options.length === 0 && isRouter}
		<div class="flex items-center gap-2 text-xs text-muted-foreground">
			<Loader2 class="h-3.5 w-3.5 animate-spin" />
			Loading models…
		</div>
	{:else if options.length === 0 && isRouter}
		<p class="text-xs text-muted-foreground">No models available.</p>
	{:else}
		{@const selectedOption = getDisplayOption()}

		<div class="relative">
			<button
				type="button"
				class={cn(
					`inline-flex cursor-pointer items-center gap-1.5 rounded-sm bg-muted-foreground/10 px-1.5 py-1 text-xs transition hover:text-foreground focus:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-60`,
					!isCurrentModelInCache()
						? 'bg-red-400/10 !text-red-400 hover:bg-red-400/20 hover:text-red-400'
						: forceForegroundText
							? 'text-foreground'
							: isHighlightedCurrentModelActive
								? 'text-foreground'
								: 'text-muted-foreground',
					isOpen ? 'text-foreground' : '',
					className
				)}
				style="max-width: min(calc(100cqw - 6.5rem), 32rem)"
				aria-haspopup={isRouter ? 'listbox' : undefined}
				aria-expanded={isRouter ? isOpen : undefined}
				onclick={toggleOpen}
				bind:this={triggerButton}
				disabled={disabled || updating}
			>
				<Package class="h-3.5 w-3.5" />

				<span class="truncate font-medium">
					{selectedOption?.model || 'Select model'}
				</span>

				{#if updating}
					<Loader2 class="h-3 w-3.5 animate-spin" />
				{:else if isRouter}
					<ChevronDown class="h-3 w-3.5" />
				{/if}
			</button>

			{#if isOpen && isRouter}
				<div
					bind:this={menuRef}
					use:portalToBody
					class={cn(
						'fixed z-[1000] overflow-hidden rounded-md border bg-popover shadow-lg transition-opacity',
						menuPosition ? 'opacity-100' : 'pointer-events-none opacity-0'
					)}
					role="listbox"
					style:top={menuPosition ? `${menuPosition.top}px` : undefined}
					style:left={menuPosition ? `${menuPosition.left}px` : undefined}
					style:width={menuPosition ? `${menuPosition.width}px` : undefined}
					data-placement={menuPosition?.placement ?? 'bottom'}
				>
					<div
						class="overflow-y-auto py-1"
						style:max-height={menuPosition && menuPosition.maxHeight > 0
							? `${menuPosition.maxHeight}px`
							: undefined}
					>
						{#if !isCurrentModelInCache() && currentModel}
							<!-- Show unavailable model as first option (disabled) -->
							<button
								type="button"
								class="flex w-full cursor-not-allowed items-center bg-red-400/10 px-3 py-2 text-left text-sm text-red-400"
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
						{#each options as option (option.id)}
							{@const status = getModelStatus(option.model)}
							{@const isLoaded = status === ServerModelStatus.LOADED}
							{@const isLoading = status === ServerModelStatus.LOADING}
							{@const isSelected = currentModel === option.model || activeId === option.id}
							{@const isCompatible = isModelCompatible(option)}
							{@const missingModalities = getMissingModalities(option)}
							<div
								class={cn(
									'group flex w-full items-center gap-2 px-3 py-2 text-left text-sm transition focus:outline-none',
									isCompatible
										? 'cursor-pointer hover:bg-muted focus:bg-muted'
										: 'cursor-not-allowed opacity-50',
									isSelected
										? 'bg-accent text-accent-foreground'
										: isCompatible
											? 'hover:bg-accent hover:text-accent-foreground'
											: '',
									isLoaded ? 'text-popover-foreground' : 'text-muted-foreground'
								)}
								role="option"
								aria-selected={isSelected}
								aria-disabled={!isCompatible}
								tabindex={isCompatible ? 0 : -1}
								onclick={() => isCompatible && handleSelect(option.id)}
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
			{/if}
		</div>
	{/if}
</div>

{#if showModelDialog && !isRouter}
	<DialogModelInformation bind:open={showModelDialog} />
{/if}
