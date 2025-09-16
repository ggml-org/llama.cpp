<script lang="ts">
	import { Edit, Copy, RefreshCw, Trash2 } from '@lucide/svelte';
	import { ActionButton, ConfirmationDialog } from '$lib/components/app';
	import ChatMessageBranchingControls from './ChatMessageBranchingControls.svelte';

	interface Props {
		message: DatabaseMessage;
		role: 'user' | 'assistant';
		justify: 'start' | 'end';
		actionsPosition: 'left' | 'right';
		siblingInfo?: ChatMessageSiblingInfo | null;
		showDeleteDialog: boolean;
		deletionInfo: {
			totalCount: number;
			userMessages: number;
			assistantMessages: number;
			messageTypes: string[];
		} | null;
		onCopy: () => void;
		onEdit?: () => void;
		onRegenerate?: () => void;
		onDelete: () => void;
		onConfirmDelete: () => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onShowDeleteDialogChange: (show: boolean) => void;
	}

	let {
		actionsPosition,
		deletionInfo,
		justify,
		message,
		onCopy,
		onEdit,
		onConfirmDelete,
		onDelete,
		onNavigateToSibling,
		onShowDeleteDialogChange,
		onRegenerate,
		role,
		siblingInfo = null,
		showDeleteDialog
	}: Props = $props();

	function handleConfirmDelete() {
		onConfirmDelete();
		onShowDeleteDialogChange(false);
	}
</script>

<div class="relative {justify === 'start' ? 'mt-2' : ''} flex h-6 items-center justify-{justify}">
	<div
		class="flex items-center text-xs text-muted-foreground transition-opacity group-hover:opacity-0"
	>
		{new Date(message.timestamp).toLocaleTimeString(undefined, {
			hour: '2-digit',
			minute: '2-digit'
		})}
	</div>

	<div
		class="absolute top-0 {actionsPosition === 'left'
			? 'left-0'
			: 'right-0'} flex items-center gap-2 opacity-0 transition-opacity group-hover:opacity-100"
	>
		{#if siblingInfo && siblingInfo.totalSiblings > 1}
			<ChatMessageBranchingControls {siblingInfo} {onNavigateToSibling} />
		{/if}

		<div
			class="pointer-events-none inset-0 flex items-center gap-1 opacity-0 transition-all duration-150 group-hover:pointer-events-auto group-hover:opacity-100"
		>
			<ActionButton icon={Copy} tooltip="Copy" onclick={onCopy} />

			{#if onEdit}
				<ActionButton icon={Edit} tooltip="Edit" onclick={onEdit} />
			{/if}

			{#if role === 'assistant' && onRegenerate}
				<ActionButton icon={RefreshCw} tooltip="Regenerate" onclick={onRegenerate} />
			{/if}

			<ActionButton icon={Trash2} tooltip="Delete" onclick={onDelete} />
		</div>
	</div>
</div>

<ConfirmationDialog
	bind:open={showDeleteDialog}
	title="Delete Message"
	description={deletionInfo && deletionInfo.totalCount > 1
		? `This will delete ${deletionInfo.totalCount} messages including: ${deletionInfo.userMessages} user message${deletionInfo.userMessages > 1 ? 's' : ''} and ${deletionInfo.assistantMessages} assistant response${deletionInfo.assistantMessages > 1 ? 's' : ''}. All messages in this branch and their responses will be permanently removed. This action cannot be undone.`
		: 'Are you sure you want to delete this message? This action cannot be undone.'}
	confirmText={deletionInfo && deletionInfo.totalCount > 1
		? `Delete ${deletionInfo.totalCount} Messages`
		: 'Delete'}
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleConfirmDelete}
	onCancel={() => onShowDeleteDialogChange(false)}
/>
