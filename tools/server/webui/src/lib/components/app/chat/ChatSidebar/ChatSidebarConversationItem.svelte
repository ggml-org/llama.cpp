<script lang="ts">
	import { Trash2, Pencil, MoreHorizontal } from '@lucide/svelte';
	import { ActionDropdown } from '$lib/components/app';
	import { onMount } from 'svelte';

	interface Props {
		isActive?: boolean;
		conversation: DatabaseConversation;
		handleMobileSidebarItemClick?: () => void;
		onDelete?: (id: string) => void;
		onEdit?: (id: string) => void;
		onSelect?: (id: string) => void;
		showLastModified?: boolean;
	}

	let {
		conversation,
		handleMobileSidebarItemClick,
		onDelete,
		onEdit,
		onSelect,
		isActive = false,
		showLastModified = false
	}: Props = $props();

	let showDropdown = $state(false);

	function formatLastModified(timestamp: number) {
		const now = Date.now();
		const diff = now - timestamp;
		const minutes = Math.floor(diff / (1000 * 60));
		const hours = Math.floor(diff / (1000 * 60 * 60));
		const days = Math.floor(diff / (1000 * 60 * 60 * 24));

		if (minutes < 1) return 'Just now';
		if (minutes < 60) return `${minutes}m ago`;
		if (hours < 24) return `${hours}h ago`;
		return `${days}d ago`;
	}

	function handleEdit(event: Event) {
		event.stopPropagation();
		onEdit?.(conversation.id);
	}

	function handleDelete(event: Event) {
		event.stopPropagation();
		onDelete?.(conversation.id);
	}

	function handleSelect() {
		onSelect?.(conversation.id);
	}

	function handleGlobalEditEvent(event: Event) {
		const customEvent = event as CustomEvent<{ conversationId: string }>;
		if (customEvent.detail.conversationId === conversation.id && isActive) {
			handleEdit(event);
		}
	}

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

<button
	class="group flex w-full cursor-pointer items-center justify-between space-x-3 rounded-lg px-3 py-1.5 text-left transition-colors hover:bg-foreground/10 {isActive
		? 'bg-foreground/5 text-accent-foreground'
		: ''}"
	onclick={handleSelect}
>
	<!-- svelte-ignore a11y_click_events_have_key_events -->
	<!-- svelte-ignore a11y_no_static_element_interactions -->
	<div
		class="text flex min-w-0 flex-1 items-center space-x-3"
		onclick={handleMobileSidebarItemClick}
	>
		<div class="min-w-0 flex-1">
			<p class="truncate text-sm font-medium">{conversation.name}</p>

			{#if showLastModified}
				<div class="mt-2 flex flex-wrap items-center space-y-2 space-x-2">
					<span class="w-full text-xs text-muted-foreground">
						{formatLastModified(conversation.lastModified)}
					</span>
				</div>
			{/if}
		</div>
	</div>

	<div class="actions flex items-center">
		<ActionDropdown
			triggerIcon={MoreHorizontal}
			triggerTooltip="More actions"
			bind:open={showDropdown}
			actions={[
				{
					icon: Pencil,
					label: 'Edit',
					onclick: handleEdit,
					shortcut: ['shift', 'cmd', 'e']
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
