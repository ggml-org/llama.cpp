<script lang="ts">
	import { Wrench } from '@lucide/svelte';
	import ChevronsUpDownIcon from '@lucide/svelte/icons/chevrons-up-down';
	import * as Collapsible from '$lib/components/ui/collapsible/index.js';
	import { buttonVariants } from '$lib/components/ui/button/index.js';
	import { Card } from '$lib/components/ui/card';
	import ChatMessageToolCallItem from './ChatMessageToolCallItem.svelte';
	import type { ApiChatCompletionToolCall } from '$lib/types/api';

	interface Props {
		class?: string;
		toolCallContent: ApiChatCompletionToolCall[] | string | null;
	}

	let { class: className = '', toolCallContent }: Props = $props();
	let fallbackExpanded = $state(false);

	const toolCalls = $derived.by(() => (Array.isArray(toolCallContent) ? toolCallContent : null));
	const fallbackContent = $derived.by(() =>
		typeof toolCallContent === 'string' ? toolCallContent : null
	);
</script>

{#if toolCalls && toolCalls.length > 0}
	<div class="mb-6 flex flex-col gap-3 {className}">
		{#each toolCalls as toolCall, index (toolCall.id ?? `${index}`)}
			<ChatMessageToolCallItem {toolCall} {index} />
		{/each}
	</div>
{:else if fallbackContent}
	<Collapsible.Root bind:open={fallbackExpanded} class="mb-6 {className}">
		<Card class="gap-0 border-muted bg-muted/30 py-0">
			<Collapsible.Trigger class="flex cursor-pointer items-center justify-between p-3">
				<div class="flex items-center gap-2 text-muted-foreground">
					<Wrench class="h-4 w-4" />

					<span class="text-sm font-medium">Tool calls</span>
				</div>

				<div
					class={buttonVariants({
						variant: 'ghost',
						size: 'sm',
						class: 'h-6 w-6 p-0 text-muted-foreground hover:text-foreground'
					})}
				>
					<ChevronsUpDownIcon class="h-4 w-4" />

					<span class="sr-only">Toggle tool call content</span>
				</div>
			</Collapsible.Trigger>

			<Collapsible.Content>
				<div class="border-t border-muted px-3 pb-3">
					<div class="pt-3">
						<pre class="tool-call-content">{fallbackContent}</pre>
					</div>
				</div>
			</Collapsible.Content>
		</Card>
	</Collapsible.Root>
{/if}

<style>
	.tool-call-content {
		font-family: var(--font-mono);
		font-size: 0.75rem;
		line-height: 1.25rem;
		white-space: pre-wrap;
		word-break: break-word;
		margin: 0;
	}
</style>
