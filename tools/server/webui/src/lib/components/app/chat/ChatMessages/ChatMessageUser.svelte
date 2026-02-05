<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import { ChatAttachmentsList, MarkdownContent } from '$lib/components/app';
	import { config } from '$lib/stores/settings.svelte';
	import ChatMessageActions from './ChatMessageActions.svelte';
	import ChatMessageEditForm from './ChatMessageEditForm.svelte';

	interface Props {
		class?: string;
		message: DatabaseMessage;
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
		onEditKeydown: (event: KeyboardEvent) => void;
		onEditedContentChange: (content: string) => void;
		onEditedExtrasChange?: (extras: DatabaseMessageExtra[]) => void;
		onEditedUploadedFilesChange?: (files: ChatUploadedFile[]) => void;
		onCopy: () => void;
		onEdit: () => void;
		onDelete: () => void;
		onConfirmDelete: () => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onShowDeleteDialogChange: (show: boolean) => void;
		textareaElement?: HTMLTextAreaElement;
	}

	let {
		class: className = '',
		message,
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
		onEditKeydown,
		onEditedContentChange,
		onEditedExtrasChange,
		onEditedUploadedFilesChange,
		onCopy,
		onEdit,
		onDelete,
		onConfirmDelete,
		onNavigateToSibling,
		onShowDeleteDialogChange,
		textareaElement = $bindable()
	}: Props = $props();

	let isMultiline = $state(false);
	let messageElement: HTMLElement | undefined = $state();
	const currentConfig = config();

	$effect(() => {
		if (!messageElement || !message.content.trim()) return;

		if (message.content.includes('\n')) {
			isMultiline = true;
			return;
		}

		const resizeObserver = new ResizeObserver((entries) => {
			for (const entry of entries) {
				const element = entry.target as HTMLElement;
				const estimatedSingleLineHeight = 24; // Typical line height for text-md

				isMultiline = element.offsetHeight > estimatedSingleLineHeight * 1.5;
			}
		});

		resizeObserver.observe(messageElement);

		return () => {
			resizeObserver.disconnect();
		};
	});
</script>

<div
	aria-label="User message with actions"
	class="group flex flex-col items-end gap-3 md:gap-2 {className}"
	role="group"
>
	{#if isEditing}
		<ChatMessageEditForm
			bind:textareaElement
			messageId={message.id}
			{editedContent}
			{editedExtras}
			{editedUploadedFiles}
			originalContent={message.content}
			originalExtras={message.extra}
			showSaveOnlyOption={!!onSaveEditOnly}
			{onCancelEdit}
			{onSaveEdit}
			{onSaveEditOnly}
			{onEditKeydown}
			{onEditedContentChange}
			{onEditedExtrasChange}
			{onEditedUploadedFilesChange}
		/>
	{:else}
		{#if message.extra && message.extra.length > 0}
			<div class="mb-2 max-w-[80%]">
				<ChatAttachmentsList attachments={message.extra} readonly={true} imageHeight="h-80" />
			</div>
		{/if}

		{#if message.content.trim()}
			<Card
				class="max-w-[80%] rounded-[1.125rem] border-none bg-primary px-3.75 py-1.5 text-primary-foreground data-[multiline]:py-2.5"
				data-multiline={isMultiline ? '' : undefined}
			>
				{#if currentConfig.renderUserContentAsMarkdown}
					<div bind:this={messageElement} class="text-md">
						<MarkdownContent
							class="markdown-user-content text-primary-foreground"
							content={message.content}
						/>
					</div>
				{:else}
					<span bind:this={messageElement} class="text-md whitespace-pre-wrap">
						{message.content}
					</span>
				{/if}
			</Card>
		{/if}

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
					role="user"
				/>
			</div>
		{/if}
	{/if}
</div>
