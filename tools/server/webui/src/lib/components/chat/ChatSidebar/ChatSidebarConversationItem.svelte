<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import { Trash2, Pencil, MoreHorizontal } from '@lucide/svelte';
	import type { DatabaseConversation } from '$lib/types/database';

	interface Props {
		conversation: DatabaseConversation;
		isActive?: boolean;
		onSelect?: (id: string) => void;
		onEdit?: (id: string) => void;
		onDelete?: (id: string) => void;
		showLastModified?: boolean;
	}

	let {
		conversation,
		isActive = false,
		onSelect,
		onEdit,
		onDelete,
		showLastModified = false
	}: Props = $props();

	let showDeleteDialog = $state(false);

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

	function handleSelect() {
		onSelect?.(conversation.id);
	}

	function handleEdit(event: Event) {
		event.stopPropagation();
		onEdit?.(conversation.id);
	}

	function handleDeleteClick(event: Event) {
		event.stopPropagation();
		// Alert dialog will handle the actual delete confirmation
	}

	function handleConfirmDelete() {
		onDelete?.(conversation.id);
	}
</script>

<button
	class="hover:bg-foreground/10 group flex w-full cursor-pointer items-center justify-between space-x-3 rounded-lg px-3 py-1.5 text-left transition-colors {isActive
		? 'bg-foreground/5 text-accent-foreground'
		: ''}"
	onclick={handleSelect}
>
	<div class="text flex min-w-0 flex-1 items-center space-x-3">
		<div class="min-w-0 flex-1">
			<p class="truncate text-sm font-medium">{conversation.name}</p>

			{#if showLastModified}
				<div class="mt-2 flex flex-wrap items-center space-x-2 space-y-2">
					<span class="text-muted-foreground w-full text-xs">
						{formatLastModified(conversation.lastModified)}
					</span>
				</div>
			{/if}
		</div>
	</div>

	<div class="actions flex items-center">
		<DropdownMenu.Root>
			<DropdownMenu.Trigger
				class="flex h-6 w-6 items-center justify-center rounded-md p-0 text-sm font-medium transition-colors hover:bg-accent hover:text-accent-foreground focus:bg-accent focus:text-accent-foreground focus:outline-none disabled:pointer-events-none disabled:opacity-50 data-[state=open]:bg-accent data-[state=open]:text-accent-foreground"
				onclick={(e) => e.stopPropagation()}
			>
				<MoreHorizontal class="h-3 w-3" />
				<span class="sr-only">More actions</span>
			</DropdownMenu.Trigger>
			<DropdownMenu.Content align="end" class="w-48">
				<DropdownMenu.Item onclick={handleEdit} class="flex items-center gap-2">
					<Pencil class="h-4 w-4" />
					Edit
				</DropdownMenu.Item>
				<DropdownMenu.Separator />
				<DropdownMenu.Item
					variant="destructive"
					class="flex items-center gap-2"
					onclick={(e) => {
						e.stopPropagation();
						showDeleteDialog = true;
					}}
				>
					<Trash2 class="h-4 w-4" />
					Delete
				</DropdownMenu.Item>
			</DropdownMenu.Content>
		</DropdownMenu.Root>

		<AlertDialog.Root bind:open={showDeleteDialog}>
			<AlertDialog.Content>
				<AlertDialog.Header>
					<AlertDialog.Title>Delete Conversation</AlertDialog.Title>
					<AlertDialog.Description>
						Are you sure you want to delete "{conversation.name}"? This action cannot be
						undone and will permanently remove all messages in this conversation.
					</AlertDialog.Description>
				</AlertDialog.Header>
				<AlertDialog.Footer>
					<AlertDialog.Cancel>Cancel</AlertDialog.Cancel>
					<AlertDialog.Action
						class="bg-destructive text-destructive-foreground"
						onclick={handleConfirmDelete}>Delete</AlertDialog.Action
					>
				</AlertDialog.Footer>
			</AlertDialog.Content>
		</AlertDialog.Root>
	</div>
</button>

<style lang="postcss">
	.actions {
		& > * {
			width: 0;
			opacity: 0;
			transition: all 0.2s ease;
		}

		button:hover & > * {
			width: auto;
			opacity: 1;
		}
	}
</style>
