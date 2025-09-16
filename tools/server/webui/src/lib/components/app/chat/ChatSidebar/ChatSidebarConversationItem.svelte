<script lang="ts">
	import { Trash2, Pencil, MoreHorizontal } from '@lucide/svelte';
	import { ActionDropdown, ConfirmationDialog } from '$lib/components/app';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import Input from '$lib/components/ui/input/input.svelte';
	import { onMount } from 'svelte';

	interface Props {
		isActive?: boolean;
		conversation: DatabaseConversation;
		onDelete?: (id: string) => void;
		onEdit?: (id: string, name: string) => void;
		onSelect?: (id: string) => void;
		showLastModified?: boolean;
	}

	let {
		conversation,
		onDelete,
		onEdit,
		onSelect,
		isActive = false,
		showLastModified = false
	}: Props = $props();

	let editedName = $state('');
	let showDeleteDialog = $state(false);
	let showDropdown = $state(false);
	let showEditDialog = $state(false);

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

	function handleConfirmDelete() {
		onDelete?.(conversation.id);
	}

	function handleConfirmEdit() {
		if (!editedName.trim()) return;
		onEdit?.(conversation.id, editedName);
	}

	function handleEdit(event: Event) {
		event.stopPropagation();
		editedName = conversation.name;
		showEditDialog = true;
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
	<div class="text flex min-w-0 flex-1 items-center space-x-3">
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
					onclick: (e) => {
						e.stopPropagation();
						showDeleteDialog = true;
					},
					variant: 'destructive',
					shortcut: ['shift', 'cmd', 'd'],
					separator: true
				}
			]}
		/>

		<ConfirmationDialog
			bind:open={showDeleteDialog}
			title="Delete Conversation"
			description={`Are you sure you want to delete "${conversation.name}"? This action cannot be undone and will permanently remove all messages in this conversation.`}
			confirmText="Delete"
			cancelText="Cancel"
			variant="destructive"
			icon={Trash2}
			onConfirm={handleConfirmDelete}
			onCancel={() => (showDeleteDialog = false)}
		/>

		<AlertDialog.Root bind:open={showEditDialog}>
			<AlertDialog.Content>
				<AlertDialog.Header>
					<AlertDialog.Title>Edit Conversation Name</AlertDialog.Title>

					<AlertDialog.Description>
						<Input
							class="mt-4 text-foreground"
							onkeydown={(e) => {
								if (e.key === 'Enter') {
									e.preventDefault();
									handleConfirmEdit();
									showEditDialog = false;
								}
							}}
							placeholder="Enter a new name"
							type="text"
							bind:value={editedName}
						/>
					</AlertDialog.Description>
				</AlertDialog.Header>

				<AlertDialog.Footer>
					<AlertDialog.Cancel>Cancel</AlertDialog.Cancel>

					<AlertDialog.Action onclick={handleConfirmEdit}>Save</AlertDialog.Action>
				</AlertDialog.Footer>
			</AlertDialog.Content>
		</AlertDialog.Root>
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
	}
</style>
