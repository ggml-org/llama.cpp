<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import { Trash2, Pencil } from '@lucide/svelte';
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
	class="hover:bg-accent group flex w-full cursor-pointer items-center justify-between space-x-3 rounded-lg p-3 text-left transition-colors {isActive
		? 'bg-accent text-accent-foreground border-border border'
		: 'border border-transparent'}"
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

	<div class="actions flex items-center space-x-1">
		<Button size="sm" variant="ghost" class="h-6 w-6 p-0" onclick={handleEdit}>
			<Pencil class="h-3 w-3" />
		</Button>
		<AlertDialog.Root>
			<AlertDialog.Trigger onclick={handleDeleteClick}>
				<Button
					size="sm"
					variant="ghost"
					class="text-destructive hover:text-destructive h-6 w-6 p-0"
				>
					<Trash2 class="h-3 w-3" />
				</Button>
			</AlertDialog.Trigger>
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
		button & {
			width: 0;
			opacity: 0;
		}

		button:hover & {
			width: auto;
			opacity: 1;
		}
	}
</style>
