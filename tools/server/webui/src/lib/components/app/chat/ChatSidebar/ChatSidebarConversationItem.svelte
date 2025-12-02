<script lang="ts">
	import { Trash2, Pencil, MoreHorizontal, Download, Loader2 } from '@lucide/svelte';
	import { ActionDropdown } from '$lib/components/app';
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
	}

	let {
		conversation,
		handleMobileSidebarItemClick,
		onDelete,
		onEdit,
		onSelect,
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
			<Loader2 class="h-3.5 w-3.5 shrink-0 animate-spin text-muted-foreground" />
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
	}
</style>
