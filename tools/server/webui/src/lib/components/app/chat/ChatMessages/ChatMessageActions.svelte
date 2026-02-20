<script lang="ts">
	import { Edit, Copy, RefreshCw, Trash2, ArrowRight } from '@lucide/svelte';
	import {
		ActionIcon,
		ChatMessageBranchingControls,
		DialogConfirmation
	} from '$lib/components/app';
	import { Switch } from '$lib/components/ui/switch';
	import { MessageRole } from '$lib/enums';

	interface Props {
		role: MessageRole.USER | MessageRole.ASSISTANT;
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
		onContinue?: () => void;
		onDelete: () => void;
		onConfirmDelete: () => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onShowDeleteDialogChange: (show: boolean) => void;
		showRawOutputSwitch?: boolean;
		rawOutputEnabled?: boolean;
		onRawOutputToggle?: (enabled: boolean) => void;
	}

	let {
		actionsPosition,
		deletionInfo,
		justify,
		onCopy,
		onEdit,
		onConfirmDelete,
		onContinue,
		onDelete,
		onNavigateToSibling,
		onShowDeleteDialogChange,
		onRegenerate,
		role,
		siblingInfo = null,
		showDeleteDialog,
		showRawOutputSwitch = false,
		rawOutputEnabled = false,
		onRawOutputToggle
	}: Props = $props();

	function handleConfirmDelete() {
		onConfirmDelete();
		onShowDeleteDialogChange(false);
	}
</script>

<div class="relative {justify === 'start' ? 'mt-2' : ''} flex h-6 items-center justify-between">
	<div
		class="{actionsPosition === 'left'
			? 'left-0'
			: 'right-0'} flex items-center gap-2 opacity-100 transition-opacity"
	>
		{#if siblingInfo && siblingInfo.totalSiblings > 1}
			<ChatMessageBranchingControls {siblingInfo} {onNavigateToSibling} />
		{/if}

		<div
			class="pointer-events-auto inset-0 flex items-center gap-1 opacity-100 transition-all duration-150"
		>
			<ActionIcon icon={Copy} tooltip="Copy" onclick={onCopy} />

			{#if onEdit}
				<ActionIcon icon={Edit} tooltip="Edit" onclick={onEdit} />
			{/if}

			{#if role === MessageRole.ASSISTANT && onRegenerate}
				<ActionIcon icon={RefreshCw} tooltip="Regenerate" onclick={() => onRegenerate()} />
			{/if}

			{#if role === MessageRole.ASSISTANT && onContinue}
				<ActionIcon icon={ArrowRight} tooltip="Continue" onclick={onContinue} />
			{/if}

			<ActionIcon icon={Trash2} tooltip="Delete" onclick={onDelete} />
		</div>
	</div>

	{#if showRawOutputSwitch}
		<div class="flex items-center gap-2">
			<span class="text-xs text-muted-foreground">Show raw output</span>
			<Switch
				checked={rawOutputEnabled}
				onCheckedChange={(checked) => onRawOutputToggle?.(checked)}
			/>
		</div>
	{/if}
</div>

<DialogConfirmation
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
