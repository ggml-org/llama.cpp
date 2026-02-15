<script lang="ts">
	import { Trash2, Pencil, MoreHorizontal, Download, Loader2, Square } from '@lucide/svelte';
	import { ActionDropdown } from '$lib/components/app';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { getAllLoadingChats } from '$lib/stores/chat.svelte';
	import { conversationsStore } from '$lib/stores/conversations.svelte';
	import { onMount } from 'svelte';

	interface Props {
		isActive?: boolean;
		conversation: DatabaseConversation;
		handleMobileSidebarItemClick?: () => void;
		onDelete?: (id: string) => void;
		onEdit?: (id: string) => void;
		onSelect?: (id: string) => void;
		onStop?: (id: string) => void;
	}

	let {
		conversation,
		handleMobileSidebarItemClick,
		onDelete,
		onEdit,
		onSelect,
		onStop,
		isActive = false
	}: Props = $props();

	let renderActionsDropdown = $state(false);
	let dropdownOpen = $state(false);

	let isLoading = $derived(getAllLoadingChats().includes(conversation.id));

	function handleEdit(event: Event) {
		event.stopPropagation();
		onEdit?.(conversation.id);
	}

	function handleDelete(event: Event) {
		event.stopPropagation();
		onDelete?.(conversation.id);
	}

	function handleStop(event: Event) {
		event.stopPropagation();
		onStop?.(conversation.id);
	}

	function handleGlobalEditEvent(event: Event) {
		const customEvent = event as CustomEvent<{ conversationId: string }>;

		if (customEvent.detail.conversationId === conversation.id && isActive) {
			handleEdit(event);
		}
	}

	function handleMouseLeave() {
		if (!dropdownOpen) {
			renderActionsDropdown = false;
		}
	}

	function handleMouseOver() {
		renderActionsDropdown = true;
	}

	function handleSelect() {
		onSelect?.(conversation.id);
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

<!-- svelte-ignore a11y_mouse_events_have_key_events -->
<button
	class="group flex min-h-9 w-full cursor-pointer items-center justify-between space-x-3 rounded-lg px-3 py-1.5 text-left transition-colors hover:bg-foreground/10 {isActive
		? 'bg-foreground/5 text-accent-foreground'
		: ''}"
	onclick={handleSelect}
	onmouseover={handleMouseOver}
	onmouseleave={handleMouseLeave}
>
	<div class="flex min-w-0 flex-1 items-center gap-2">
		{#if isLoading}
			<Tooltip.Root>
				<Tooltip.Trigger>
					<div
						class="stop-button flex h-4 w-4 shrink-0 cursor-pointer items-center justify-center rounded text-muted-foreground transition-colors hover:text-foreground"
						onclick={handleStop}
						onkeydown={(e) => e.key === 'Enter' && handleStop(e)}
						role="button"
						tabindex="0"
						aria-label="Stop generation"
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

		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<span class="truncate text-sm font-medium" onclick={handleMobileSidebarItemClick}>
			{conversation.name}
		</span>
	</div>

	{#if renderActionsDropdown}
		<div class="actions flex items-center">
			<ActionDropdown
				triggerIcon={MoreHorizontal}
				triggerTooltip="More actions"
				bind:open={dropdownOpen}
				actions={[
					{
						icon: Pencil,
						label: 'Edit',
						onclick: handleEdit,
						shortcut: ['shift', 'cmd', 'e']
					},
					{
						icon: Download,
						label: 'Export',
						onclick: (e) => {
							e.stopPropagation();
							conversationsStore.downloadConversation(conversation.id);
						},
						shortcut: ['shift', 'cmd', 's']
					},
					{
						icon: Trash2,
						label: 'Delete',
						onclick: handleDelete,
						variant: 'destructive',
						shortcut: ['shift', 'cmd', 'd'],
						separator: true
					}
				]}
			/>
		</div>
	{/if}
</button>

<style>
	button {
		:global([data-slot='dropdown-menu-trigger']:not([data-state='open'])) {
			opacity: 0;
		}

		&:is(:hover) :global([data-slot='dropdown-menu-trigger']) {
			opacity: 1;
		}
		@media (max-width: 768px) {
			:global([data-slot='dropdown-menu-trigger']) {
				opacity: 1 !important;
			}
		}

		.stop-button {
			:global(.stop-icon) {
				display: none;
			}

			:global(.loading-icon) {
				display: block;
			}
		}

		&:is(:hover) .stop-button {
			:global(.stop-icon) {
				display: block;
			}

			:global(.loading-icon) {
				display: none;
			}
		}
	}
</style>
