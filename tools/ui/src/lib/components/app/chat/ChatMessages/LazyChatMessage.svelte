<script lang="ts">
	import ChatMessage from './ChatMessage/ChatMessage.svelte';
	import { chatStore } from '$lib/stores';
	import type { ChatMessageActions } from '$lib/types';

	interface Props {
		chatActions: ChatMessageActions;
		class?: string;
		/** Mount right away instead of waiting for the row to near the viewport. */
		eager?: boolean;
		isLastAssistantMessage?: boolean;
		isLastUserMessage?: boolean;
		message: DatabaseMessage;
		nextAssistantMessage?: DatabaseMessage | null;
		siblingInfo?: ChatMessageSiblingInfo | null;
		toolMessages?: DatabaseMessage[];
	}

	let {
		chatActions,
		class: className = '',
		eager = false,
		isLastAssistantMessage = false,
		isLastUserMessage = false,
		message,
		nextAssistantMessage = null,
		siblingInfo = null,
		toolMessages = []
	}: Props = $props();

	// A mounted message row is a whole component tree (contexts, effects,
	// collapsibles, markdown blocks), and the cycle collector, GC and layout
	// invalidation keep walking every live object and DOM node, even for
	// rows the user never scrolls to. Mount the real tree only when the row
	// approaches the viewport; until then the row is an empty placeholder
	// that reserves its size through content-visibility.
	let mounted = $state(false);
	let wrapperEl: HTMLDivElement | undefined = $state();

	// eager rows render in the same pass as the list, so a conversation
	// landing at its bottom never paints empty placeholders there; latch it,
	// since a row stops being eager once newer messages push it up
	let shouldMount = $derived(mounted || eager);

	$effect(() => {
		if (eager) mounted = true;
	});

	$effect(() => {
		if (shouldMount || !wrapperEl) return;

		const observer = new IntersectionObserver(
			(entries) => {
				if (entries.some((entry) => entry.isIntersecting)) {
					mounted = true;
					observer.disconnect();
				}
			},
			// pre-mount a couple of viewport heights ahead of the scroll
			// position so a fast scroll never meets an empty row
			{ rootMargin: '200% 0px' }
		);

		observer.observe(wrapperEl);

		return () => observer.disconnect();
	});

	// Flows that target a row by id (pending edit) expect the message
	// component and its effects to exist; mount the target row first
	$effect(() => {
		if (chatStore.pendingEditMessageId === message.id) {
			mounted = true;
		}
	});
</script>

<div
	bind:this={wrapperEl}
	class:chat-message--eager={eager}
	class:chat-message--synthetic={Boolean(message.isSynthetic)}
	class="chat-message"
>
	{#if shouldMount}
		<ChatMessage
			{chatActions}
			class={className}
			{isLastAssistantMessage}
			{isLastUserMessage}
			{message}
			{nextAssistantMessage}
			{siblingInfo}
			{toolMessages}
		/>
	{/if}
</div>

<style>
	/*
	 * The browser skips layout and paint for messages outside the
	 * viewport. contain-intrinsic-size reuses the last rendered size
	 * once known; 500px sizes messages that have never been rendered.
	 */
	.chat-message {
		--chat-message-intrinsic-size: 500px;
		content-visibility: auto;
		contain-intrinsic-size: auto var(--chat-message-intrinsic-size);
	}

	/*
	 * Synthetic rows (e.g. the working-directory change) are small, so an
	 * accurate placeholder keeps the injected row from inflating the
	 * auto-scroll offset; the 500px default is for ordinary bubbles.
	 */
	.chat-message--synthetic {
		--chat-message-intrinsic-size: 40px;
	}

	/*
	 * A freshly created row is skipped for a frame until its viewport
	 * proximity is determined, so the rows a conversation lands on would paint
	 * as empty placeholders. They sit at the bottom anyway, so lay them out.
	 */
	.chat-message--eager {
		content-visibility: visible;
	}
</style>
