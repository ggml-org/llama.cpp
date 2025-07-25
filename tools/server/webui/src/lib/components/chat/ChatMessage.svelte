<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Tooltip, TooltipContent, TooltipTrigger } from '$lib/components/ui/tooltip';
	import { Textarea } from '$lib/components/ui/textarea';
	import { User, Bot, Edit, Copy, Trash2, RefreshCw, Check, X } from '@lucide/svelte';
	import type { ChatRole } from '$lib/types/chat';
	import type { DatabaseChatMessage } from '$lib/types/database';
	import ThinkingSection from './ThinkingSection.svelte';
	import MarkdownContent from './MarkdownContent.svelte';
	import { parseThinkingContent } from '$lib/utils/thinking';
	import { copyToClipboard } from '$lib/utils/copy';

	interface Props {
		class?: string;
		message: DatabaseChatMessage;
		onEdit?: (message: DatabaseChatMessage) => void;
		onDelete?: (message: DatabaseChatMessage) => void;
		onCopy?: (message: DatabaseChatMessage) => void;
		onRegenerate?: (message: DatabaseChatMessage) => void;
		onUpdateMessage?: (message: DatabaseChatMessage, newContent: string) => void;
	}

	let {
		class: className = '',
		message,
		onEdit,
		onDelete,
		onCopy,
		onRegenerate,
		onUpdateMessage
	}: Props = $props();

	// Editing state
	let isEditing = $state(false);
	let editedContent = $state(message.content);
	// Element reference (not reactive)
	let textareaElement: HTMLTextAreaElement;

	// Parse thinking content for assistant messages
	// Use separate derived values to prevent unnecessary re-renders
	let thinkingContent = $derived.by(() => {
		if (message.role === 'assistant') {
			// Prioritize message.thinking (from streaming) over parsed thinking
			if (message.thinking) {
				return message.thinking;
			}
			// Fallback to parsing content for complete messages
			const parsed = parseThinkingContent(message.content);
			return parsed.thinking;
		}
		return null;
	});

	let messageContent = $derived.by(() => {
		if (message.role === 'assistant') {
			// Always parse and clean the content to remove <think>...</think> blocks
			const parsed = parseThinkingContent(message.content);
			return parsed.cleanContent;
		}
		return message.content;
	});

	// Handle copy to clipboard
	async function handleCopy() {
		await copyToClipboard(message.content, 'Message copied to clipboard');
		onCopy?.(message);
	}

	// Handle edit action
	function handleEdit() {
		isEditing = true;
		editedContent = message.content;
		// Focus the textarea after it's rendered
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

	// Handle save edited message
	function handleSaveEdit() {
		if (editedContent.trim() && editedContent !== message.content) {
			onUpdateMessage?.(message, editedContent.trim());
		}
		isEditing = false;
	}

	// Handle cancel edit
	function handleCancelEdit() {
		isEditing = false;
		editedContent = message.content;
	}

	// Handle keyboard shortcuts in edit mode
	function handleEditKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			handleSaveEdit();
		} else if (event.key === 'Escape') {
			event.preventDefault();
			handleCancelEdit();
		}
	}

	// Handle regenerate action
	function handleRegenerate() {
		onRegenerate?.(message);
	}
</script>

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
		<!-- This will be implemented after fully migrating to SvelteKit -->

		<!-- <Tooltip>
			<TooltipTrigger>
				<Button
					variant="ghost"
					size="sm"
					class="text-destructive hover:text-destructive h-6 w-6 p-0"
					onclick={handleDelete}
				>
					<Trash2 class="h-3 w-3" />
				</Button>
			</TooltipTrigger>
			<TooltipContent>
				<p>Delete</p>
			</TooltipContent>
		</Tooltip> -->
	</div>

	<div
		class="{config?.role === 'user'
			? 'right-0'
			: 'left-0'} text-muted-foreground absolute text-xs transition-all duration-150 group-hover:pointer-events-none group-hover:opacity-0"
	>
		{message.timestamp ? new Date(message.timestamp).toLocaleTimeString() : ''}
	</div>
{/snippet}

{#if message.role === 'user'}
	<div
		class="group flex flex-col items-end gap-2 {className}"
		role="group"
		aria-label="User message with actions"
	>
		{#if isEditing}
			<!-- Editing mode -->
			<div class="w-full max-w-[80%]">
				<textarea
					bind:this={textareaElement}
					bind:value={editedContent}
					onkeydown={handleEditKeydown}
					class="border-primary bg-background text-foreground focus:ring-ring min-h-[60px] w-full resize-none rounded-2xl border-2 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-offset-2"
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
			<!-- Display mode -->
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
			<ThinkingSection thinking={thinkingContent} isStreaming={!message.timestamp} />
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
