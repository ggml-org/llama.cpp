<script lang="ts">
	import { Edit, Copy, RefreshCw, Check, X, Trash2 } from '@lucide/svelte';
	import { Card } from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { ChatAttachmentsList, ChatMessageThinkingBlock, MarkdownContent } from '$lib/components/app';
	import { Tooltip, TooltipContent, TooltipTrigger } from '$lib/components/ui/tooltip';
	import {
		AlertDialog,
		AlertDialogAction,
		AlertDialogCancel,
		AlertDialogContent,
		AlertDialogDescription,
		AlertDialogFooter,
		AlertDialogHeader,
		AlertDialogTitle,
		AlertDialogTrigger
	} from '$lib/components/ui/alert-dialog';
	import { copyToClipboard } from '$lib/utils/copy';
	import { parseThinkingContent } from '$lib/utils/thinking';
	import { isLoading } from '$lib/stores/chat.svelte';
	import { useProcessingState } from '$lib/hooks/use-processing-state.svelte';
	import { fade } from 'svelte/transition';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		onEdit?: (message: DatabaseMessage) => void;
		onCopy?: (message: DatabaseMessage) => void;
		onRegenerate?: (message: DatabaseMessage) => void;
		onUpdateMessage?: (message: DatabaseMessage, newContent: string) => void;
		onDelete?: (message: DatabaseMessage) => void;
	}

	let {
		class: className = '',
		message,
		onEdit,
		onCopy,
		onRegenerate,
		onUpdateMessage,
		onDelete
	}: Props = $props();

	let isEditing = $state(false);
	let editedContent = $state(message.content);
	let textareaElement: HTMLTextAreaElement | undefined = $state();
	let showDeleteDialog = $state(false);

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
		onEdit?.(message);
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
		onRegenerate?.(message);
	}

	function handleSaveEdit() {
		if (editedContent.trim() !== message.content) {
			onUpdateMessage?.(message, editedContent.trim());
		}
		isEditing = false;
	}

	function handleDelete() {
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
					class="border-primary bg-foreground/3 dark:bg-muted text-foreground border-1 focus:ring-ring min-h-[60px] w-full resize-none rounded-2xl px-3 py-2 text-sm focus:outline-none focus:ring-1 focus:ring-offset-2"
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

			<div class="relative flex h-6 items-center">
				{@render messageActions({ role: 'user' })}
			</div>
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
			<div class="relative mt-2 flex h-6 items-center">
				{@render messageActions({ role: 'assistant' })}
			</div>
		{/if}
	</div>
{/if}

{#snippet messageActions(config?: { role: ChatRole })}
	<div
		class="pointer-events-none inset-0 flex items-center gap-1 opacity-0 transition-all duration-150 group-hover:pointer-events-auto group-hover:opacity-100"
	>
		<Tooltip>
			<TooltipTrigger>
				<Button variant="ghost" size="sm" class="h-6 w-6 p-0" onclick={handleCopy}>
					<Copy class="h-3 w-3" />
				</Button>
			</TooltipTrigger>

			<TooltipContent>
				<p>Copy</p>
			</TooltipContent>
		</Tooltip>
		{#if config?.role === 'user'}
			<Tooltip>
				<TooltipTrigger>
					<Button variant="ghost" size="sm" class="h-6 w-6 p-0" onclick={handleEdit}>
						<Edit class="h-3 w-3" />
					</Button>
				</TooltipTrigger>

				<TooltipContent>
					<p>Edit</p>
				</TooltipContent>
			</Tooltip>
		{:else if config?.role === 'assistant'}
			<Tooltip>
				<TooltipTrigger>
					<Button
						variant="ghost"
						size="sm"
						class="h-6 w-6 p-0"
						onclick={handleRegenerate}
					>
						<RefreshCw class="h-3 w-3" />
					</Button>
				</TooltipTrigger>

				<TooltipContent>
					<p>Regenerate</p>
				</TooltipContent>
			</Tooltip>
		{/if}

		<Tooltip>
			<TooltipTrigger>
				<Button variant="ghost" size="sm" class="h-6 w-6 p-0 hover:bg-destructive/10 hover:text-destructive" onclick={handleDelete}>
					<Trash2 class="h-3 w-3 text-destructive" />
				</Button>
			</TooltipTrigger>

			<TooltipContent>
				<p>Delete</p>
			</TooltipContent>
		</Tooltip>
	</div>

	{#if messageContent.trim().length > 0}
		<div
			class="{config?.role === 'user'
				? 'right-0'
				: 'left-0'} text-muted-foreground absolute text-xs transition-all duration-150 group-hover:pointer-events-none group-hover:opacity-0"
		>
			{message.timestamp
				? new Date(message.timestamp).toLocaleTimeString(undefined, {
						hour: '2-digit',
						minute: '2-digit'
					})
				: ''}
		</div>
	{/if}
{/snippet}

<AlertDialog bind:open={showDeleteDialog}>
	<AlertDialogContent>
		<AlertDialogHeader>
			<AlertDialogTitle>Delete Message</AlertDialogTitle>
			<AlertDialogDescription>
				Are you sure you want to delete this message? This action cannot be undone.
			</AlertDialogDescription>
		</AlertDialogHeader>
		<AlertDialogFooter>
			<AlertDialogCancel>Cancel</AlertDialogCancel>
			<AlertDialogAction onclick={handleConfirmDelete} class="bg-destructive text-destructive-foreground hover:bg-destructive/90">
				Delete
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
