<script lang="ts">
	import { Checkbox } from '@ggml-org/llama-ui/ui/checkbox';
	import * as Tooltip from '@ggml-org/llama-ui/ui/tooltip';
	import { GitBranch, Loader2, MoreHorizontal, Square } from '@lucide/svelte';
	import {
		ContextMenuActions,
		ConversationActions,
		DropdownMenuActions
	} from '#lib/components/app/index.ts';
	import { TruncatedText } from '#lib/components/app/index.ts';
	import {
		FORK_TREE_DEPTH_PADDING,
		ICON_CLASS_DEFAULT,
		ROW_LONG_PRESS_MS,
		UI_DATA_ATTRS
	} from '#lib/constants/index.ts';
	import { useLongPress } from '#lib/hooks/use-long-press.svelte.ts';
	import { RouterService } from '#lib/services/router.service.ts';
	import {
		chatStore,
		deviceStore,
		settingsStore,
		sidebarDragStore,
		tabsStore,
		uiStore
	} from '#lib/stores/index.ts';
	import type { MenuAction } from '#lib/types/index.ts';
	import { hapticTap } from '#lib/utils/index.ts';
	import { onMount, tick } from 'svelte';

	interface Props {
		isActive?: boolean;
		depth?: number;
		conversation: DatabaseConversation;
		isSelectionMode?: boolean;
		isSelected?: boolean;
		onSelect?: (id: string) => void;
		onStop?: (id: string) => void;
		onToggleSelect?: (id: string) => void;
		onEnterSelectionMode?: (id: string) => void;
		onSelectionClick?: (id: string, options: { shiftKey: boolean }) => void;
		onRowMouseDown?: (id: string, event: MouseEvent) => void;
	}

	let {
		conversation,
		depth = 0,
		isActive = false,
		isSelected = false,
		isSelectionMode = false,
		onEnterSelectionMode,
		onRowMouseDown,
		onSelect,
		onSelectionClick,
		onStop,
		onToggleSelect
	}: Props = $props();

	let renderActionsDropdown = $state(false);
	let dropdownOpen = $state(false);
	let contextMenuOpen = $state(false);
	let rowElement = $state<HTMLButtonElement | null>(null);

	// a phone has no three-dot trigger: holding the row opens its menu, anchored to the row
	const longPress = useLongPress({
		enabled: () => deviceStore.isMobile && !isSelectionMode,
		onLongPress: openHoldMenu
	});

	// the hold itself taps the haptic engine
	async function openHoldMenu() {
		renderActionsDropdown = true;
		await tick();
		dropdownOpen = true;
	}

	let isLoading = $derived(chatStore.getAllLoadingChats().includes(conversation.id));
	let isLanding = $derived(sidebarDragStore.landingConvId === conversation.id);

	// the row's own controls (the fork link, stop, its menu) keep their press
	function handlePointerDown(event: PointerEvent) {
		if (isSelectionMode) return;

		const control = (event.target as Element).closest(
			'a, button, [role="button"], [data-slot$="-trigger"]'
		);

		if (control && control !== event.currentTarget) return;

		sidebarDragStore.pressConversation(event, conversation.id);
	}

	function handleStop(event: Event) {
		event.stopPropagation();
		onStop?.(conversation.id);
	}

	let canOpenInNewTab = $derived(Boolean(settingsStore.config.conversationTabs));

	function openInNewTab() {
		void tabsStore.openInNewTab(conversation.id);
	}

	// middle-click opens a background tab, like a link in a browser
	function handleAuxClick(event: MouseEvent) {
		if (event.button !== 1 || isSelectionMode || !canOpenInNewTab) return;

		event.preventDefault();
		openInNewTab();
	}

	function handleEnterSelectionMode() {
		onEnterSelectionMode?.(conversation.id);
	}

	function handleGlobalEditEvent(event: Event) {
		const customEvent = event as CustomEvent<{ conversationId: string }>;

		if (customEvent.detail.conversationId === conversation.id && isActive) {
			uiStore.openConversationRename(conversation.id);
		}
	}

	function handleMouseLeave() {
		if (!dropdownOpen) {
			renderActionsDropdown = false;
		}
	}

	function handleMouseOver() {
		if (isSelectionMode) return;

		renderActionsDropdown = true;
	}

	function handleSelect(event: MouseEvent) {
		if (isSelectionMode) {
			onSelectionClick?.(conversation.id, { shiftKey: event.shiftKey });
		} else if ((event.metaKey || event.ctrlKey) && canOpenInNewTab) {
			openInNewTab();
		} else {
			if (deviceStore.isMobile) hapticTap();

			onSelect?.(conversation.id);
		}
	}

	function handleCheckboxClick(event: MouseEvent) {
		event.stopPropagation();

		if (isSelectionMode) {
			onSelectionClick?.(conversation.id, { shiftKey: event.shiftKey });
		} else {
			onToggleSelect?.(conversation.id);
		}
	}

	function handleRowMouseDown(event: MouseEvent) {
		onRowMouseDown?.(conversation.id, event);
	}

	function handleCheckboxKeydown(event: KeyboardEvent) {
		if (event.key !== ' ' && event.key !== 'Enter') return;

		event.stopPropagation();
		event.preventDefault();

		if (isSelectionMode) {
			onSelectionClick?.(conversation.id, { shiftKey: event.shiftKey });
		} else {
			onToggleSelect?.(conversation.id);
		}
	}

	$effect(() => {
		if (!dropdownOpen) {
			renderActionsDropdown = false;
		}
	});

	onMount(() => {
		document.addEventListener('edit-active-conversation', handleGlobalEditEvent as EventListener);

		return () => {
			document.removeEventListener(
				'edit-active-conversation',
				handleGlobalEditEvent as EventListener
			);
		};
	});
</script>

<ConversationActions
	{conversation}
	{isActive}
	onEnterSelectionMode={onEnterSelectionMode ? handleEnterSelectionMode : undefined}
>
	{#snippet children(actions)}
		{@render row(actions)}
	{/snippet}
</ConversationActions>

{#snippet row(actions: MenuAction[])}
	<ContextMenuActions bind:open={contextMenuOpen} {actions} disabled={isSelectionMode}>
		<button
			bind:this={rowElement}
			style:--row-hold-ms="{ROW_LONG_PRESS_MS}ms"
			class={[
				'group flex min-h-9 w-full cursor-pointer items-center justify-between space-x-3 rounded-lg px-2 py-1.5 text-left select-none [-webkit-touch-callout:none] transition-[background-color,color,scale] hover:bg-foreground/10 hover:text-foreground max-md:min-h-12 max-md:rounded-2xl max-md:px-3.5 max-md:py-3 md:min-h-8 md:py-1',
				isLanding && 'invisible',
				isActive ? 'selected-row-surface text-accent-foreground' : 'text-foreground/75',
				contextMenuOpen && 'bg-foreground/10',
				isSelected && 'bg-primary/10 hover:bg-primary/15',
				isSelectionMode && 'is-selection-mode',
				// the row sinks while a finger holds it, and stays lit while its menu is open
				longPress.isPressing && 'scale-97 bg-foreground/10 duration-(--row-hold-ms) ease-out',
				deviceStore.isMobile && dropdownOpen && 'bg-foreground/10'
			]}
			{...{ [UI_DATA_ATTRS.CONVERSATION_ROW]: conversation.id }}
			onauxclick={handleAuxClick}
			onclick={(e) => handleSelect(e)}
			onclickcapture={longPress.handleClickCapture}
			oncontextmenu={longPress.handleContextMenu}
			ondragstart={(e) => e.preventDefault()}
			onfocusin={handleMouseOver}
			onfocusout={(e) => {
				if (!e.currentTarget.contains(e.relatedTarget as Node | null)) {
					handleMouseLeave();
				}
			}}
			onmousedown={(e) => handleRowMouseDown(e)}
			onmouseleave={handleMouseLeave}
			onmouseover={handleMouseOver}
			onpointercancel={longPress.handlePointerEnd}
			onpointerdown={(e) => {
				// a mouse drags the row, a finger holds it for its menu
				handlePointerDown(e);
				longPress.handlePointerDown(e);
			}}
			onpointermove={longPress.handlePointerMove}
			onpointerup={longPress.handlePointerEnd}
		>
			<div
				style:padding-left="{depth * FORK_TREE_DEPTH_PADDING}px"
				class="flex min-w-0 flex-1 items-center gap-2"
			>
				{#if isSelectionMode}
					<div
						aria-checked={isSelected}
						aria-label={isSelected
							? `Deselect ${conversation.name}`
							: `Select ${conversation.name}`}
						class="shrink-0"
						onclick={(e) => handleCheckboxClick(e)}
						onkeydown={handleCheckboxKeydown}
						role="checkbox"
						tabindex="-1"
					>
						<Checkbox
							aria-label={isSelected
								? `Deselect ${conversation.name}`
								: `Select ${conversation.name}`}
							checked={isSelected}
						/>
					</div>
				{/if}

				{#if depth > 0}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<!-- prevent another nested button element -->
							{#snippet child({ props })}
								<a
									{...props}
									class="flex shrink-0 items-center text-muted-foreground transition-colors hover:text-foreground"
									href={RouterService.chat(conversation.forkedFromConversationId)}
								>
									<GitBranch class="h-3.5 w-3.5" />
								</a>
							{/snippet}
						</Tooltip.Trigger>

						<Tooltip.Content>
							<p>See parent conversation</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}

				{#if isLoading}
					<Tooltip.Root>
						<Tooltip.Trigger>
							<div
								aria-label="Stop generation"
								class="stop-button flex {ICON_CLASS_DEFAULT} shrink-0 cursor-pointer items-center justify-center rounded text-muted-foreground transition-colors hover:text-foreground"
								onclick={handleStop}
								onkeydown={(e) => e.key === 'Enter' && handleStop(e)}
								role="button"
								tabindex="0"
							>
								<Loader2 class="loading-icon h-3.5 w-3.5 animate-spin" />

								<Square class="stop-icon hidden h-3 w-3 fill-current text-destructive" />
							</div>
						</Tooltip.Trigger>

						<Tooltip.Content>
							<p>Stop generation</p>
						</Tooltip.Content>
					</Tooltip.Root>
				{/if}

				<TruncatedText
					class="text-body-sm max-md:text-body"
					showTooltip={false}
					text={conversation.name}
				/>
			</div>

			{#if !isSelectionMode && renderActionsDropdown}
				<!-- a phone opens the menu by holding the row, so it shows no trigger -->
				<div class="actions flex items-center" hidden={deviceStore.isMobile}>
					<DropdownMenuActions
						bind:open={dropdownOpen}
						{actions}
						anchor={deviceStore.isMobile ? rowElement : null}
						triggerIcon={MoreHorizontal}
						triggerTooltip="More actions"
					/>
				</div>
			{/if}
		</button>
	</ContextMenuActions>
{/snippet}

<style>
	button {
		&.is-selection-mode :global([data-slot='dropdown-menu-trigger']) {
			display: none !important;
		}

		.stop-button {
			:global(.stop-icon) {
				display: none;
			}

			:global(.loading-icon) {
				display: block;
			}
		}
	}

	/* the row's menu waits for a hover where there is one; a touch screen shows it all along */
	@media (hover: hover) and (pointer: fine) {
		button :global([data-slot='dropdown-menu-trigger']:not([data-state='open'])) {
			opacity: 0;
		}

		button:is(:hover, :focus-within) :global([data-slot='dropdown-menu-trigger']) {
			opacity: 1;
		}

		button:hover {
			.stop-button {
				:global(.stop-icon) {
					display: block;
				}

				:global(.loading-icon) {
					display: none;
				}
			}
		}
	}
</style>
