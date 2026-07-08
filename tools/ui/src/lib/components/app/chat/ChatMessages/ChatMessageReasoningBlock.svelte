<script lang="ts">
	import { Lightbulb } from '@lucide/svelte';
	import { CollapsibleContentBlock, MarkdownContent } from '$lib/components/app';
	import { AgenticSectionType } from '$lib/enums';
	import type { DatabaseMessageExtra } from '$lib/types';
	import type { AgenticSection } from '$lib/utils';

	interface Props {
		section: AgenticSection;
		open: boolean;
		isStreaming: boolean;
		renderThinkingAsMarkdown: boolean;
		hasReasoningError?: boolean;
		attachments?: DatabaseMessageExtra[];
		onToggle?: () => void;
	}

	let {
		section,
		open,
		isStreaming,
		renderThinkingAsMarkdown,
		hasReasoningError = false,
		attachments,
		onToggle
	}: Props = $props();

	const isPending = $derived(section.type === AgenticSectionType.REASONING_PENDING);
	const title = $derived(isPending && isStreaming ? 'Reasoning...' : 'Reasoning');
	const subtitle = $derived.by(() => {
		if (isPending && !isStreaming) {
			return hasReasoningError ? 'Error' : 'Cancelled';
		}
		if (section.wasInterrupted) {
			return hasReasoningError ? 'Error' : 'Cancelled';
		}
		return isStreaming ? '' : undefined;
	});
	const shimmerTitle = $derived(isPending && isStreaming);
</script>

<CollapsibleContentBlock
	{open}
	class="my-2"
	icon={Lightbulb}
	iconClass="h-3.5 w-3.5"
	{title}
	{subtitle}
	rawContent={section.content}
	{isStreaming}
	{shimmerTitle}
	{onToggle}
>
	{#if renderThinkingAsMarkdown}
		<MarkdownContent content={section.content} {attachments} />
	{:else}
		<div class="text-[13px] leading-relaxed break-words whitespace-pre-wrap text-foreground/90">
			{section.content}
		</div>
	{/if}
</CollapsibleContentBlock>
