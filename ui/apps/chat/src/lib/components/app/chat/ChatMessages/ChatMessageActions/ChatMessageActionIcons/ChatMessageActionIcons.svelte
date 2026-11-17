<script lang="ts">
	import { ArrowRight, Copy, Edit, GitBranch, RefreshCw, Trash2 } from '@lucide/svelte';
	import {
		ActionIcon,
		ChatMessageActionIconsBranchingControls,
		DialogConfirmation
	} from '$lib/components/app';
	import { Checkbox } from '$lib/components/ui/checkbox';
	import Input from '$lib/components/ui/input/input.svelte';
	import Label from '$lib/components/ui/label/label.svelte';
	import { Switch } from '$lib/components/ui/switch';
	import { getChatMessageActionsContext, getChatMessageEditContext } from '$lib/contexts';
	import { conversationsStore } from '$lib/stores';
	import { untrack } from 'svelte';

	interface Props {
		justify: 'start' | 'end';
		actionsPosition: 'left' | 'right';
		onRegenerate?: () => void;
		onContinue?: () => void;
		showRawOutputSwitch?: boolean;
		rawOutputEnabled?: boolean;
		onRawOutputToggle?: (enabled: boolean) => void;
	}

	let {
		actionsPosition,
		justify,
		onContinue,
		onRawOutputToggle,
		onRegenerate,
		rawOutputEnabled = false,
		showRawOutputSwitch = false
	}: Props = $props();

	const messageActions = getChatMessageActionsContext();
	const editCtx = getChatMessageEditContext();

	let forkName = $state('');
	let forkIncludeAttachments = $state(true);

	function handleConfirmDelete() {
		messageActions.confirmDelete();
		messageActions.setShowDeleteDialog(false);
	}

	// fills the fork form each time the dialog opens, from the icon or the
	// message's context menu
	$effect(() => {
		if (!messageActions.showForkDialog) return;

		untrack(() => {
			forkName = `Fork of ${conversationsStore.activeConversation?.name ?? 'Conversation'}`;
			forkIncludeAttachments = true;
		});
	});

	function handleConfirmFork() {
		messageActions.forkConversation?.({
			includeAttachments: forkIncludeAttachments,
			name: forkName.trim()
		});
		messageActions.setShowForkDialog(false);
	}
</script>

<div class="relative {justify === 'start' ? 'mt-2' : ''} flex h-6 items-center justify-between">
	<div
		class="{actionsPosition === 'left'
			? 'left-0'
			: 'right-0'} flex items-center gap-2 opacity-100 transition-opacity"
	>
		{#if messageActions.siblingInfo && messageActions.siblingInfo.totalSiblings > 1}
			<ChatMessageActionIconsBranchingControls />
		{/if}

		<div
			class="pointer-events-auto inset-0 flex items-center gap-1 opacity-100 transition-all duration-150"
		>
			<ActionIcon icon={Copy} onclick={messageActions.copy} tooltip="Copy" />

			<ActionIcon icon={Edit} onclick={editCtx.startEdit} tooltip="Edit" />

			{#if onRegenerate}
				<ActionIcon icon={RefreshCw} onclick={() => onRegenerate()} tooltip="Regenerate" />
			{/if}

			{#if onContinue}
				<ActionIcon icon={ArrowRight} onclick={onContinue} tooltip="Continue" />
			{/if}

			{#if messageActions.forkConversation}
				<ActionIcon
					icon={GitBranch}
					onclick={() => messageActions.setShowForkDialog(true)}
					tooltip="Fork conversation"
				/>
			{/if}

			<ActionIcon icon={Trash2} onclick={messageActions.requestDelete} tooltip="Delete" />
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
	cancelText="Cancel"
	confirmText={messageActions.deletionInfo && messageActions.deletionInfo.totalCount > 1
		? `Delete ${messageActions.deletionInfo.totalCount} Messages`
		: 'Delete'}
	description={messageActions.deletionInfo && messageActions.deletionInfo.totalCount > 1
		? `This will delete ${messageActions.deletionInfo.totalCount} messages including: ${messageActions.deletionInfo.userMessages} user message${messageActions.deletionInfo.userMessages > 1 ? 's' : ''} and ${messageActions.deletionInfo.assistantMessages} assistant response${messageActions.deletionInfo.assistantMessages > 1 ? 's' : ''}. All messages in this branch and their responses will be permanently removed. This action cannot be undone.`
		: 'Are you sure you want to delete this message? This action cannot be undone.'}
	icon={Trash2}
	onCancel={() => messageActions.setShowDeleteDialog(false)}
	onConfirm={handleConfirmDelete}
	open={messageActions.showDeleteDialog}
	title="Delete Message"
	variant="destructive"
/>

<DialogConfirmation
	cancelText="Cancel"
	confirmText="Fork"
	description="Create a new conversation branching from this message."
	icon={GitBranch}
	onCancel={() => messageActions.setShowForkDialog(false)}
	onConfirm={handleConfirmFork}
	open={messageActions.showForkDialog}
	title="Fork Conversation"
>
	<div class="flex flex-col gap-4 py-2">
		<div class="flex flex-col gap-2">
			<Label for="fork-name">Title</Label>

			<Input
				bind:value={forkName}
				class="text-foreground"
				id="fork-name"
				placeholder="Enter fork name"
				type="text"
			/>
		</div>

		<div class="flex items-center gap-2">
			<Checkbox
				checked={forkIncludeAttachments}
				id="fork-attachments"
				onCheckedChange={(checked) => {
					forkIncludeAttachments = checked === true;
				}}
			/>

			<Label class="cursor-pointer text-sm font-normal" for="fork-attachments">
				Include all attachments
			</Label>
		</div>
	</div>
</DialogConfirmation>
