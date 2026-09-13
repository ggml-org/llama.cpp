<script lang="ts">
	import ChatMessage from './ChatMessage/ChatMessage.svelte';
	import { chatStore } from '$lib/stores';
	import type { ChatMessageActions } from '$lib/types';

	interface Props {
		chatActions: ChatMessageActions;
		class?: string;
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

	$effect(() => {
		if (mounted || !wrapperEl) return;

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
	class:chat-message--synthetic={Boolean(message.isSynthetic)}
	class="chat-message"
>
	{#if mounted}
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
</style>
