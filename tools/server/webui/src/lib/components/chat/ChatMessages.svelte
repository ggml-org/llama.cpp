<script lang="ts">
	import type { Message } from '$lib/types/database';
	import { updateMessage, regenerateMessage } from '$lib/stores/chat.svelte';
	import ChatMessage from './ChatMessage.svelte';
	interface Props {
		class?: string;
		messages?: Message[];
		isLoading?: boolean;
	}

	let { class: className, messages = [], isLoading = false }: Props = $props();

	let scrollContainer = $state<HTMLDivElement>();
	let shouldAutoScroll = $state(true);
	let lastScrollHeight = $state(0);

	// Check if user is at the bottom of the scroll container
	function isAtBottom(): boolean {
		if (!scrollContainer) return false;
		const { scrollTop, scrollHeight, clientHeight } = scrollContainer;
		// Consider "at bottom" if within 100px of the bottom
		return scrollHeight - scrollTop - clientHeight < 100;
	}

	// Handle scroll events to detect manual scrolling
	function handleScroll() {
		if (!scrollContainer) return;

		// If user scrolled to bottom, re-enable auto-scroll
		if (isAtBottom()) {
			shouldAutoScroll = true;
		} else {
			// If user scrolled up manually, disable auto-scroll
			shouldAutoScroll = false;
		}
	}

	// Auto-scroll when messages change or loading state changes
	$effect(() => {
		if (scrollContainer && shouldAutoScroll) {
			const currentScrollHeight = scrollContainer.scrollHeight;

			// Only scroll if content has actually changed (new messages or loading)
			if (currentScrollHeight !== lastScrollHeight) {
				setTimeout(() => {
					if (scrollContainer && shouldAutoScroll) {
						scrollContainer.scrollTop = scrollContainer.scrollHeight;
						lastScrollHeight = scrollContainer.scrollHeight;
					}
				}, 0);
			}
		}
	});

	// Update lastScrollHeight when messages change
	$effect(() => {
		if (scrollContainer) {
			lastScrollHeight = scrollContainer.scrollHeight;
		}
	});
</script>

<div class="flex h-full flex-col {className}">
	<div
		bind:this={scrollContainer}
		class="bg-background flex-1 overflow-y-auto p-4"
		onscroll={handleScroll}
	>
		<div class="mb-48 mt-16 space-y-6">
			{#each messages as message}
				<ChatMessage
					class="mx-auto w-full max-w-[56rem]"
					{message}
					onUpdateMessage={async (msg, newContent) => {
						await updateMessage(msg.id, newContent);
					}}
					onRegenerate={async (msg) => {
						await regenerateMessage(msg.id);
					}}
				/>
			{/each}
		</div>
	</div>
</div>
