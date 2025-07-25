<script lang="ts">
	import { Card } from '$lib/components/ui/card';
	import { Button } from '$lib/components/ui/button';
	import { Tooltip, TooltipContent, TooltipTrigger } from '$lib/components/ui/tooltip';
	import { User, Bot, Edit, Copy, Trash2, RefreshCw } from '@lucide/svelte';
	import type { ChatRole } from '$lib/types/chat';
	import type { ChatMessageData } from '$lib/types/chat';
	import ThinkingSection from './ThinkingSection.svelte';
	import MarkdownContent from './MarkdownContent.svelte';
	import { parseThinkingContent } from '$lib/utils/thinking';

	interface Props {
		class?: string;
		message: ChatMessageData;
		onEdit?: (message: ChatMessageData) => void;
		onDelete?: (message: ChatMessageData) => void;
		onCopy?: (message: ChatMessageData) => void;
		onRegenerate?: (message: ChatMessageData) => void;
	}

	let {
		class: className = '',
		message,
		onEdit,
		onDelete,
		onCopy,
		onRegenerate
	}: Props = $props();

	// Parse thinking content for assistant messages
	const parsedContent = $derived(() => {
		if (message.role === 'assistant') {
			const parsed = parseThinkingContent(message.content);
			return {
				thinking: message.thinking || parsed.thinking,
				content: parsed.cleanContent || message.content
			};
		}
		return { thinking: null, content: message.content };
	});

	// Handle copy to clipboard
	function handleCopy() {
		navigator.clipboard.writeText(message.content);
		onCopy?.(message);
	}

	// Handle edit action
	function handleEdit() {
		onEdit?.(message);
	}

	// Handle delete action
	function handleDelete() {
		onDelete?.(message);
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
		<Card class="bg-primary text-primary-foreground max-w-[80%] rounded-2xl px-2.5 py-1.5">
			<div class="text-md whitespace-pre-wrap">
				{message.content}
			</div>
		</Card>

		<div class="relative flex h-6 items-center">
			{@render messageActions({ role: 'user' })}
		</div>
	</div>
{:else}
	<div
		class="text-md leading-7.5 group w-full {className}"
		role="group"
		aria-label="Assistant message with actions"
	>
		{#if parsedContent().thinking}
			<ThinkingSection thinking={parsedContent().thinking || ''} />
		{/if}
		{#if message.role === 'assistant'}
			<MarkdownContent content={parsedContent().content} />
		{:else}
			<div class="whitespace-pre-wrap text-sm">
				{parsedContent().content}
			</div>
		{/if}

		{#if message.timestamp}
			<div class="relative mt-2 flex h-6 items-center">
				{@render messageActions({ role: 'assistant' })}
			</div>
		{/if}
	</div>
{/if}
