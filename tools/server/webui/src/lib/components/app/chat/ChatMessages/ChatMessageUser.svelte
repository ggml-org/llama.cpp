<script lang="ts">
	import { Check, X } from '@lucide/svelte';
	import { Card } from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { ChatAttachmentsList, MarkdownContent } from '$lib/components/app';
	import { INPUT_CLASSES } from '$lib/constants/input-classes';
	import { config } from '$lib/stores/settings.svelte';
	import ChatMessageActions from './ChatMessageActions.svelte';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		isEditing: boolean;
		editedContent: string;
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
		onEditKeydown: (event: KeyboardEvent) => void;
		onEditedContentChange: (content: string) => void;
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
		siblingInfo = null,
		showDeleteDialog,
		deletionInfo,
		onCancelEdit,
		onSaveEdit,
		onEditKeydown,
		onEditedContentChange,
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
		<div class="w-full max-w-[80%]">
			<textarea
				bind:this={textareaElement}
				bind:value={editedContent}
				class="min-h-[60px] w-full resize-none rounded-2xl px-3 py-2 text-sm {INPUT_CLASSES}"
				onkeydown={onEditKeydown}
				oninput={(e) => onEditedContentChange(e.currentTarget.value)}
				placeholder="Edit your message..."
			></textarea>

			<div class="mt-2 flex justify-end gap-2">
				<Button class="h-8 px-3" onclick={onCancelEdit} size="sm" variant="outline">
					<X class="mr-1 h-3 w-3" />

					Cancel
				</Button>

				<Button class="h-8 px-3" onclick={onSaveEdit} disabled={!editedContent.trim()} size="sm">
					<Check class="mr-1 h-3 w-3" />

					Send
				</Button>
			</div>
		</div>
	{:else}
		{#if message.extra && message.extra.length > 0}
			<div class="mb-2 max-w-[80%]">
				<ChatAttachmentsList attachments={message.extra} readonly={true} imageHeight="h-80" />
			</div>
		{/if}

		{#if message.content.trim()}
			<Card
				class="max-w-[80%] rounded-[1.125rem] bg-primary px-3.75 py-1.5 text-primary-foreground data-[multiline]:py-2.5"
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
