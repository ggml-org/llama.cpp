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

	let scrollEl: HTMLDivElement | undefined = $state();
	const SCROLL_BOTTOM_THRESHOLD_PX = 32;

	$effect(() => {
		// Re-run when content grows while reasoning is in progress; pin to the
		// bottom unless the user has scrolled up to read earlier content.
		void section.content;
		if (!scrollEl || !isPending || !isStreaming) return;

		const distanceFromBottom = scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight;
		if (distanceFromBottom <= SCROLL_BOTTOM_THRESHOLD_PX) {
			scrollEl.scrollTop = scrollEl.scrollHeight;
		}
	});
</script>

<CollapsibleContentBlock
	{open}
	class="my-2"
	icon={Lightbulb}
	iconClass="h-3.5 w-3.5"
	{title}
	{subtitle}
	{shimmerTitle}
	{onToggle}
>
	<div bind:this={scrollEl} class="reasoning-content" class:is-streaming={isPending}>
		{#if renderThinkingAsMarkdown}
			<MarkdownContent content={section.content} {attachments} />
		{:else}
			<div class="text-[13px] leading-relaxed break-words whitespace-pre-wrap text-foreground/90">
				{section.content}
			</div>
		{/if}
	</div>
</CollapsibleContentBlock>

<style>
	.reasoning-content.is-streaming {
		max-height: 28rem;
		overflow-y: auto;
		overscroll-behavior: contain;
		scrollbar-gutter: stable;
		padding-right: 0.25rem;
	}
</style>
