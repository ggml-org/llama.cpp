<script lang="ts">
	import { ChatMessageActions, ChatMessageMcpPromptContent } from '$lib/components/app';
	import { MessageRole, McpPromptVariant } from '$lib/enums';
	import type { DatabaseMessageExtraMcpPrompt } from '$lib/types';
	import ChatMessageEditForm from './ChatMessageEditForm.svelte';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		mcpPrompt: DatabaseMessageExtraMcpPrompt;
		isEditing: boolean;
		editedContent: string;
		editedExtras?: DatabaseMessageExtra[];
		editedUploadedFiles?: ChatUploadedFile[];
		siblingInfo?: ChatMessageSiblingInfo | null;
		showDeleteDialog: boolean;
		deletionInfo: {
			totalCount: number;
			userMessages: number;
			assistantMessages: number;
			messageTypes: string[];
		} | null;
		onCancelEdit: () => void;
		onSaveEdit: () => void;
		onSaveEditOnly?: () => void;
		onEditedContentChange: (content: string) => void;
		onEditedExtrasChange?: (extras: DatabaseMessageExtra[]) => void;
		onEditedUploadedFilesChange?: (files: ChatUploadedFile[]) => void;
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
		isEditing,
		editedContent,
		editedExtras = [],
		editedUploadedFiles = [],
		siblingInfo = null,
		showDeleteDialog,
		deletionInfo,
		onCancelEdit,
		onSaveEdit,
		onSaveEditOnly,
		onEditedContentChange,
		onEditedExtrasChange,
		onEditedUploadedFilesChange,
		onCopy,
		onEdit,
		onDelete,
		onConfirmDelete,
		onNavigateToSibling,
		onShowDeleteDialogChange
	}: Props = $props();
</script>

<div
	aria-label="MCP Prompt message with actions"
	class="group flex flex-col items-end gap-3 md:gap-2 {className}"
	role="group"
>
	{#if isEditing}
		<ChatMessageEditForm
			{editedContent}
			{editedExtras}
			{editedUploadedFiles}
			originalContent={message.content}
			originalExtras={message.extra}
			showSaveOnlyOption={!!onSaveEditOnly}
			{onCancelEdit}
			{onSaveEdit}
			{onSaveEditOnly}
			{onEditedContentChange}
			{onEditedExtrasChange}
			{onEditedUploadedFilesChange}
		/>
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
