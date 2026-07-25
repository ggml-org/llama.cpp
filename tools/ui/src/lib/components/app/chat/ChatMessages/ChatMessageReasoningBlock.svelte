<script lang="ts">
	import { Lightbulb } from '@lucide/svelte';
	import { CollapsibleContentBlock, MarkdownContent } from '$lib/components/app';
	import { AgenticSectionType } from '$lib/enums';
	import { REASONING_SCROLL_AT_BOTTOM_THRESHOLD_PX } from '$lib/constants/auto-scroll';
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

	const REASONING_HEADER = 'Reasoning';
	const REASONING_HEADER_PENDING = 'Reasoning...';
	const REASONING_SUBTITLE_ERROR = 'Error';
	const REASONING_SUBTITLE_CANCELLED = 'Cancelled';

	const isPending = $derived(section.type === AgenticSectionType.REASONING_PENDING);
	const title = $derived(isPending && isStreaming ? REASONING_HEADER_PENDING : REASONING_HEADER);
	const subtitle = $derived.by(() => {
		if (isPending && !isStreaming) {
			return hasReasoningError ? REASONING_SUBTITLE_ERROR : REASONING_SUBTITLE_CANCELLED;
		}
		if (section.wasInterrupted) {
			return hasReasoningError ? REASONING_SUBTITLE_ERROR : REASONING_SUBTITLE_CANCELLED;
		}
		return isStreaming ? '' : undefined;
	});
	const shimmerTitle = $derived(isPending && isStreaming);

	let scrollEl: HTMLDivElement | undefined = $state();

	const SCROLL_BOTTOM_THRESHOLD_PX = REASONING_SCROLL_AT_BOTTOM_THRESHOLD_PX;

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
			// User may scroll between scheduling and paint.
			if (scrollEl && !userScrolledUp) {
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
		void section.content;
		if (!scrollEl || !isPending || !isStreaming) return;
		scrollToBottomOnFrame();
	});

	$effect(() => {
		// Layout shifts that don't change section.content (markdown re-parse,
		// syntax-highlight settle, image loads).
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
		// Pin to bottom at the start of each round.
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
			<MarkdownContent content={section.content} class="text-muted-foreground" {attachments} />
		{:else}
			<div
				class="text-[13px] leading-relaxed wrap-break-word whitespace-pre-wrap text-muted-foreground"
			>
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
