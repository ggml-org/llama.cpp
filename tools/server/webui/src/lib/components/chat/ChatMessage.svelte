<script lang="ts">
	import { Edit, Copy, RefreshCw, Check, X } from '@lucide/svelte';
	import { Card } from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { ChatAttachmentsList, ChatThinkingBlock, MarkdownContent } from '$lib/components';
	import { Tooltip, TooltipContent, TooltipTrigger } from '$lib/components/ui/tooltip';
	import type { ChatRole } from '$lib/types/chat';
	import type { DatabaseMessage } from '$lib/types/database';
	import { copyToClipboard } from '$lib/utils/copy';
	import { parseThinkingContent } from '$lib/utils/thinking';

	interface Props {
		class?: string;
		message: DatabaseMessage;
		onEdit?: (message: DatabaseMessage) => void;
		onCopy?: (message: DatabaseMessage) => void;
		onRegenerate?: (message: DatabaseMessage) => void;
		onUpdateMessage?: (message: DatabaseMessage, newContent: string) => void;
	}

	let {
		class: className = '',
		message,
		onEdit,
		onCopy,
		onRegenerate,
		onUpdateMessage
	}: Props = $props();

	let isEditing = $state(false);
	let editedContent = $state(message.content);
	let textareaElement: HTMLTextAreaElement | undefined = $state();

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
			return parsed.cleanContent;
		}
		return message.content;
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
		if (editedContent.trim() && editedContent !== message.content) {
			onUpdateMessage?.(message, editedContent.trim());
		}
		isEditing = false;
	}
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
				<div class="max-w-[80%] mb-2">
					<ChatAttachmentsList 
						attachments={message.extra}
						readonly={true}
						imageHeight="h-80"
					/>
				</div>
			{/if}
			
			<Card class="bg-primary text-primary-foreground max-w-[80%] rounded-2xl px-2.5 py-1.5">
				<div class="text-md whitespace-pre-wrap">
					{message.content}
				</div>
			</Card>

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
			<ChatThinkingBlock thinking={thinkingContent} isStreaming={!message.timestamp} />
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
	</div>

	<div
		class="{config?.role === 'user'
			? 'right-0'
			: 'left-0'} text-muted-foreground absolute text-xs transition-all duration-150 group-hover:pointer-events-none group-hover:opacity-0"
	>
		{message.timestamp ? new Date(message.timestamp).toLocaleTimeString() : ''}
	</div>
{/snippet}