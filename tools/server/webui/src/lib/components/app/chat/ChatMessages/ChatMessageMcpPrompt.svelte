<script lang="ts">
	import { ChatMessageActions, ChatMessageMcpPromptContent } from '$lib/components/app';
	import { getMessageEditContext } from '$lib/contexts';
	import { MessageRole, McpPromptVariant } from '$lib/enums';
	import type { DatabaseMessageExtraMcpPrompt } from '$lib/types';
	import ChatMessageEditForm from './ChatMessageEditForm.svelte';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		mcpPrompt: DatabaseMessageExtraMcpPrompt;
		siblingInfo?: ChatMessageSiblingInfo | null;
		showDeleteDialog: boolean;
		deletionInfo: {
			totalCount: number;
			userMessages: number;
			assistantMessages: number;
			messageTypes: string[];
		} | null;
		onCopy: () => void;
		onEdit: () => void;
		onDelete: () => void;
		onConfirmDelete: () => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onShowDeleteDialogChange: (show: boolean) => void;
	}

	let {
		class: className = '',
		message,
		mcpPrompt,
		siblingInfo = null,
		showDeleteDialog,
		deletionInfo,
		onCopy,
		onEdit,
		onDelete,
		onConfirmDelete,
		onNavigateToSibling,
		onShowDeleteDialogChange
	}: Props = $props();

	// Get edit context
	const editCtx = getMessageEditContext();
</script>

<div
	aria-label="MCP Prompt message with actions"
	class="group flex flex-col items-end gap-3 md:gap-2 {className}"
	role="group"
>
	{#if editCtx.isEditing}
		<ChatMessageEditForm />
	{:else}
		<ChatMessageMcpPromptContent
			prompt={mcpPrompt}
			variant={McpPromptVariant.MESSAGE}
			class="w-full max-w-[80%]"
		/>

		{#if message.timestamp}
			<div class="max-w-[80%]">
				<ChatMessageActions
					actionsPosition="right"
					{deletionInfo}
					justify="end"
					{onConfirmDelete}
					{onCopy}
					{onDelete}
					{onEdit}
					{onNavigateToSibling}
					{onShowDeleteDialogChange}
					{siblingInfo}
					{showDeleteDialog}
					role={MessageRole.USER}
				/>
			</div>
		{/if}
	{/if}
</div>
