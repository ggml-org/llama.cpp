<script lang="ts">
	import { Edit, Copy, RefreshCw, Check, X, Trash2 } from '@lucide/svelte';
	import { Card } from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { ChatAttachmentsList, ChatMessageThinkingBlock, MarkdownContent } from '$lib/components/app';
	import { Tooltip, TooltipContent, TooltipTrigger } from '$lib/components/ui/tooltip';
	import MessageBranchingControls from './MessageBranchingControls.svelte';
	import type { MessageSiblingInfo } from '$lib/utils/branching';
	import {
		AlertDialog,
		AlertDialogAction,
		AlertDialogCancel,
		AlertDialogContent,
		AlertDialogDescription,
		AlertDialogFooter,
		AlertDialogHeader,
		AlertDialogTitle
	} from '$lib/components/ui/alert-dialog';
	import { copyToClipboard } from '$lib/utils/copy';
	import { parseThinkingContent } from '$lib/utils/thinking';
	import { getDeletionInfo } from '$lib/stores/chat.svelte';
	import { isLoading } from '$lib/stores/chat.svelte';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { fade } from 'svelte/transition';
	import { inputClasses } from '$lib/constants/input-classes';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		siblingInfo?: MessageSiblingInfo | null;
		onCopy?: (message: DatabaseMessage) => void;
		onDelete?: (message: DatabaseMessage) => void;
		onNavigateToSibling?: (siblingId: string) => void;
		onEditWithBranching?: (message: DatabaseMessage, newContent: string) => void;
		onRegenerateWithBranching?: (message: DatabaseMessage) => void;
	}

	let {
		class: className = '',
		message,
		siblingInfo = null,
		onCopy,
		onDelete,
		onNavigateToSibling,
		onEditWithBranching,
		onRegenerateWithBranching
	}: Props = $props();

	let showDeleteDialog = $state(false);
	let editedContent = $state(message.content);
	let isEditing = $state(false);
	let deletionInfo = $state<{ totalCount: number; userMessages: number; assistantMessages: number; messageTypes: string[] } | null>(null);
	let textareaElement: HTMLTextAreaElement | undefined = $state();

	const processingState = useProcessingState();

	let thinkingContent = $derived.by(() => {
		if (message.role === 'assistant') {
			if (message.thinking) {
				return message.thinking;
			}

			const parsed = parseThinkingContent(message.content);

			return parsed.thinking;
		}
		return null;
	});

	let messageContent = $derived.by(() => {
		if (message.role === 'assistant') {
			const parsed = parseThinkingContent(message.content);
			return parsed.cleanContent?.replace('<|channel|>analysis', '');
		}

		return message.content?.replace('<|channel|>analysis', '');
	});

	async function handleCopy() {
		await copyToClipboard(message.content, 'Message copied to clipboard');
		onCopy?.(message);
	}

	function handleCancelEdit() {
		isEditing = false;
		editedContent = message.content;
	}

	function handleEdit() {
		isEditing = true;
		editedContent = message.content;
		setTimeout(() => {
			if (textareaElement) {
				textareaElement.focus();
				textareaElement.setSelectionRange(
					textareaElement.value.length,
					textareaElement.value.length
				);
			}
		}, 0);
	}

	function handleEditKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			handleSaveEdit();
		} else if (event.key === 'Escape') {
			event.preventDefault();
			handleCancelEdit();
		}
	}

	function handleRegenerate() {
		onRegenerateWithBranching?.(message);
	}

	function handleSaveEdit() {
		if (editedContent.trim() !== message.content) {
			onEditWithBranching?.(message, editedContent.trim());
		}
		isEditing = false;
	}

	async function handleDelete() {
		deletionInfo = await getDeletionInfo(message.id);
		showDeleteDialog = true;
	}

	function handleConfirmDelete() {
		onDelete?.(message);
		showDeleteDialog = false;
	}

	$effect(() => {
		if (message.role === 'assistant' && !message.content && isLoading()) {
			processingState.startMonitoring();
		} else {
			processingState.stopMonitoring();
		}
	});
</script>

{#if message.role === 'user'}
	<div
		class="group flex flex-col items-end gap-2 {className}"
		role="group"
		aria-label="User message with actions"
	>
		{#if isEditing}
			<div class="w-full max-w-[80%]">
				<textarea
					bind:this={textareaElement}
					bind:value={editedContent}
					onkeydown={handleEditKeydown}
					class="min-h-[60px] w-full resize-none rounded-2xl px-3 py-2 text-sm {inputClasses}"
					placeholder="Edit your message..."
				></textarea>

				<div class="mt-2 flex justify-end gap-2">
					<Button variant="outline" size="sm" class="h-8 px-3" onclick={handleCancelEdit}>
						<X class="mr-1 h-3 w-3" />
						Cancel
					</Button>

					<Button
						size="sm"
						class="h-8 px-3"
						onclick={handleSaveEdit}
						disabled={!editedContent.trim() || editedContent === message.content}
					>
						<Check class="mr-1 h-3 w-3" />
						Send
					</Button>
				</div>
			</div>
		{:else}
			{#if message.extra && message.extra.length > 0}
				<div class="mb-2 max-w-[80%]">
					<ChatAttachmentsList
						attachments={message.extra}
						readonly={true}
						imageHeight="h-80"
					/>
				</div>
			{/if}

			{#if message.content.trim()}
				<Card class="bg-primary text-primary-foreground max-w-[80%] rounded-2xl px-2.5 py-1.5">
					<div class="text-md whitespace-pre-wrap">
						{message.content}
					</div>
				</Card>
			{/if}

			{#if message.timestamp}
				{@render timestampAndActions({ role: 'user', justify: 'end', actionsPosition: 'right' })}
			{/if}
		{/if}
	</div>
{:else}
	<div
		class="text-md leading-7.5 group w-full {className}"
		role="group"
		aria-label="Assistant message with actions"
	>
		{#if thinkingContent}
			<ChatMessageThinkingBlock 
				reasoningContent={thinkingContent} 
				isStreaming={!message.timestamp} 
				hasRegularContent={!!messageContent?.trim()}
			/>
		{/if}

		{#if message?.role === 'assistant' && !message.content && isLoading()}
			<div class="w-full max-w-[48rem] mt-6" in:fade>
				<div class="processing-container">
					<span class="processing-text">
						{processingState.getProcessingMessage()}
					</span>
					
					{#if processingState.shouldShowDetails()}
						<div class="processing-details">
							{#each processingState.getProcessingDetails() as detail}
								<span class="processing-detail">{detail}</span>
							{/each}
						</div>
					{/if}
				</div>
			</div>
		{/if}

		{#if message.role === 'assistant'}
			<MarkdownContent content={messageContent} />
		{:else}
			<div class="whitespace-pre-wrap text-sm">
				{messageContent}
			</div>
		{/if}

		{#if message.timestamp}
			{@render timestampAndActions({ role: 'assistant', justify: 'start', actionsPosition: 'left' })}
		{/if}
	</div>
{/if}

{#snippet messageActions(config?: { role: ChatRole })}
	<div
		class="pointer-events-none inset-0 flex items-center gap-1 opacity-0 transition-all duration-150 group-hover:pointer-events-auto group-hover:opacity-100"
	>
		{@render actionButton({ icon: Copy, tooltip: 'Copy', onclick: handleCopy })}
		
		{#if config?.role === 'user'}
			{@render actionButton({ icon: Edit, tooltip: 'Edit', onclick: handleEdit })}
		{:else if config?.role === 'assistant'}
			{@render actionButton({ icon: RefreshCw, tooltip: 'Regenerate', onclick: handleRegenerate })}
		{/if}
		
		{@render actionButton({ icon: Trash2, tooltip: 'Delete', onclick: handleDelete })}
	</div>
{/snippet}

{#snippet actionButton(config: { icon: any; tooltip: string; onclick: () => void })}
	<Tooltip>
		<TooltipTrigger>
			<Button variant="ghost" size="sm" class="h-6 w-6 p-0" onclick={config.onclick}>
				{@const IconComponent = config.icon}
				<IconComponent class="h-3 w-3" />
			</Button>
		</TooltipTrigger>

		<TooltipContent>
			<p>{config.tooltip}</p>
		</TooltipContent>
	</Tooltip>
{/snippet}

{#snippet timestampAndActions(config: { role: ChatRole; justify: 'start' | 'end'; actionsPosition: 'left' | 'right' })}
	<div class="relative {config.justify === 'start' ? 'mt-2' : ''} flex h-6 items-center justify-{config.justify}">
		<div class="flex items-center text-xs text-muted-foreground group-hover:opacity-0 transition-opacity">
			{new Date(message.timestamp).toLocaleTimeString(undefined, {
				hour: '2-digit',
				minute: '2-digit'
			})}
		</div>

		<div class="absolute {config.actionsPosition}-0 flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
			{#if siblingInfo && siblingInfo.totalSiblings > 1}
				<MessageBranchingControls 
					{siblingInfo} 
					{onNavigateToSibling}
				/>
			{/if}
			{@render messageActions({ role: config.role })}
		</div>
	</div>
{/snippet}

<AlertDialog bind:open={showDeleteDialog}>
	<AlertDialogContent>
		<AlertDialogHeader>
			<AlertDialogTitle>Delete Message</AlertDialogTitle>
			<AlertDialogDescription>
				{#if deletionInfo && deletionInfo.totalCount > 1}
					<div class="space-y-2">
						<p>This will delete <strong>{deletionInfo.totalCount} messages</strong> including:</p>

						<ul class="list-disc list-inside text-sm space-y-1 ml-4">
							{#if deletionInfo.userMessages > 0}
								<li>{deletionInfo.userMessages} user message{deletionInfo.userMessages > 1 ? 's' : ''}</li>
							{/if}

							{#if deletionInfo.assistantMessages > 0}
								<li>{deletionInfo.assistantMessages} assistant response{deletionInfo.assistantMessages > 1 ? 's' : ''}</li>
							{/if}
						</ul>

						<p class="text-sm text-muted-foreground mt-2">
							All messages in this branch and their responses will be permanently removed. This action cannot be undone.
						</p>
					</div>
				{:else}
					Are you sure you want to delete this message? This action cannot be undone.
				{/if}
			</AlertDialogDescription>
		</AlertDialogHeader>

		<AlertDialogFooter>
			<AlertDialogCancel>Cancel</AlertDialogCancel>

			<AlertDialogAction onclick={handleConfirmDelete} class="bg-destructive text-destructive-foreground hover:bg-destructive/90">
				{#if deletionInfo && deletionInfo.totalCount > 1}
					Delete {deletionInfo.totalCount} Messages
				{:else}
					Delete
				{/if}
			</AlertDialogAction>
		</AlertDialogFooter>
	</AlertDialogContent>
</AlertDialog>

<style>
	.processing-container {
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 0.5rem;
	}

	.processing-text {
		background: linear-gradient(90deg, var(--muted-foreground), var(--foreground), var(--muted-foreground));
		background-size: 200% 100%;
		background-clip: text;
		-webkit-background-clip: text;
		-webkit-text-fill-color: transparent;
		animation: shine 1s linear infinite;
		font-weight: 500;
		font-size: 0.875rem;
	}

	.processing-details {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.5rem;
		margin-top: 0;
	}

	.processing-detail {
		color: var(--muted-foreground);
		font-size: 0.75rem;
		padding: 0.25rem 0.5rem;
		background: var(--muted);
		border-radius: 0.5rem;
		font-family: ui-monospace, SFMono-Regular, "SF Mono", Consolas, "Liberation Mono", Menlo, monospace;
		white-space: nowrap;
		line-height: 1.2;
	}

	@keyframes shine {
		to {
			background-position: -200% 0;
		}
	}
</style>
