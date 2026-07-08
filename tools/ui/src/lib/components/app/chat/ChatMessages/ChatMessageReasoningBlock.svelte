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

	// Generous stickiness: anything within this many pixels of the bottom is
	// considered "at the bottom" and continues to follow new content. Larger
	// than the chat main view's 10px threshold because reasoning fires lots of
	// small incremental DOM writes that easily drift a few pixels off bottom.
	const SCROLL_BOTTOM_THRESHOLD_PX = 64;

	let userScrolledUp = $state(false);
	let lastScrollTop = 0;
	let pendingFrame: number | null = null;

	function isAtBottom(): boolean {
		if (!scrollEl) return false;
		return (
			scrollEl.scrollHeight - scrollEl.clientHeight - scrollEl.scrollTop <=
			SCROLL_BOTTOM_THRESHOLD_PX
		);
	}

	function scrollToBottomOnFrame() {
		if (pendingFrame !== null || !scrollEl || userScrolledUp) return;
		pendingFrame = requestAnimationFrame(() => {
			pendingFrame = null;
			// Re-check at paint time: the user may have scrolled between
			// scheduling the frame and the frame firing.
			if (scrollEl && !userScrolledUp && isAtBottom()) {
				scrollEl.scrollTop = scrollEl.scrollHeight;
			}
		});
	}

	function handleScrollEvent() {
		if (!scrollEl) return;
		const isScrollingUp = scrollEl.scrollTop < lastScrollTop;
		if (isScrollingUp && !isAtBottom()) {
			userScrolledUp = true;
		} else if (isAtBottom()) {
			userScrolledUp = false;
		}
		lastScrollTop = scrollEl.scrollTop;
	}

	$effect(() => {
		// Primary trigger: content updates directly. Coalesced via RAF so a
		// burst of chunks within the same paint frame results in one scroll.
		void section.content;
		if (!scrollEl || !isPending || !isStreaming) return;
		scrollToBottomOnFrame();
	});

	$effect(() => {
		// Secondary trigger: any DOM mutation inside the scroll region. This
		// catches layout changes that don't touch section.content directly,
		// e.g. markdown re-parsing turning plain text into a list, code blocks
		// expanding as syntax highlighting settles, image loads, etc.
		if (!scrollEl || !isPending || !isStreaming) return;

		const observer = new MutationObserver(() => scrollToBottomOnFrame());
		observer.observe(scrollEl, {
			childList: true,
			subtree: true,
			characterData: true
		});

		return () => observer.disconnect();
	});

	$effect(() => {
		// Reasoning just ended - reset sticky state so the next round starts
		// pinned to the bottom again, even if the user scrolled away.
		if (!isPending) {
			userScrolledUp = false;
			lastScrollTop = 0;
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
	<div
		bind:this={scrollEl}
		class="reasoning-content"
		class:is-streaming={isPending}
		onscroll={handleScrollEvent}
	>
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
