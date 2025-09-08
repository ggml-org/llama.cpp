<script lang="ts">
	import { Edit, Copy, RefreshCw, Trash2 } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as AlertDialog from '$lib/components/ui/alert-dialog';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import type { DatabaseMessage } from '$lib/types/database';
	import type { Component } from 'svelte';
	import ChatMessageBranchingControls from './ChatMessageBranchingControls.svelte';

	interface Props {
		message: DatabaseMessage;
		role: 'user' | 'assistant';
		justify: 'start' | 'end';
		actionsPosition: 'left' | 'right';
		siblingInfo?: MessageSiblingInfo | null;
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
			{@render actionButton({ icon: Copy, tooltip: 'Copy', onclick: onCopy })}

			{#if onEdit}
				{@render actionButton({ icon: Edit, tooltip: 'Edit', onclick: onEdit })}
			{/if}

			{#if role === 'assistant' && onRegenerate}
				{@render actionButton({ icon: RefreshCw, tooltip: 'Regenerate', onclick: onRegenerate })}
			{/if}

			{@render actionButton({ icon: Trash2, tooltip: 'Delete', onclick: onDelete })}
		</div>
	</div>
</div>

{#snippet actionButton(config: { icon: Component; tooltip: string; onclick: () => void })}
	<Tooltip.Root>
		<Tooltip.Trigger>
			<Button variant="ghost" size="sm" class="h-6 w-6 p-0" onclick={config.onclick}>
				{@const IconComponent = config.icon}

				<IconComponent class="h-3 w-3" />
			</Button>
		</Tooltip.Trigger>

		<Tooltip.Content>
			<p>{config.tooltip}</p>
		</Tooltip.Content>
	</Tooltip.Root>
{/snippet}

<AlertDialog.Root bind:open={showDeleteDialog}>
	<AlertDialog.Content
		onkeydown={(e) => {
			if (e.key === 'Enter') {
				e.preventDefault();
				handleConfirmDelete();
			}
		}}
	>
		<AlertDialog.Header>
			<AlertDialog.Title>Delete Message</AlertDialog.Title>

			<AlertDialog.Description>
				{#if deletionInfo && deletionInfo.totalCount > 1}
					<div class="space-y-2">
						<p>This will delete <strong>{deletionInfo.totalCount} messages</strong> including:</p>

						<ul class="ml-4 list-inside list-disc space-y-1 text-sm">
							{#if deletionInfo.userMessages > 0}
								<li>
									{deletionInfo.userMessages} user message{deletionInfo.userMessages > 1 ? 's' : ''}
								</li>
							{/if}

							{#if deletionInfo.assistantMessages > 0}
								<li>
									{deletionInfo.assistantMessages} assistant response{deletionInfo.assistantMessages >
									1
										? 's'
										: ''}
								</li>
							{/if}
						</ul>

						<p class="mt-2 text-sm text-muted-foreground">
							All messages in this branch and their responses will be permanently removed. This
							action cannot be undone.
						</p>
					</div>
				{:else}
					Are you sure you want to delete this message? This action cannot be undone.
				{/if}
			</AlertDialog.Description>
		</AlertDialog.Header>

		<AlertDialog.Footer>
			<AlertDialog.Cancel>Cancel</AlertDialog.Cancel>

			<AlertDialog.Action
				onclick={handleConfirmDelete}
				class="bg-destructive text-white hover:bg-destructive/80"
			>
				{#if deletionInfo && deletionInfo.totalCount > 1}
					Delete {deletionInfo.totalCount} Messages
				{:else}
					Delete
				{/if}
			</AlertDialog.Action>
		</AlertDialog.Footer>
	</AlertDialog.Content>
</AlertDialog.Root>
