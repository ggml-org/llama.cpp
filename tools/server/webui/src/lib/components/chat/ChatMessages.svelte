<script lang="ts">
	import type { ChatMessageData } from '$lib/types/chat';
	import ChatMessage from './ChatMessage.svelte';
	import ChatMessageLoading from './ChatMessageLoading.svelte';
	interface Props {
		class?: string;
		messages?: ChatMessageData[];
		isLoading?: boolean;
	}

	let { class: className, messages = [], isLoading = false }: Props = $props();

	let scrollContainer = $state<HTMLDivElement>();

	$effect(() => {
		if (scrollContainer && (messages.length > 0 || isLoading)) {
			setTimeout(() => {
				if (scrollContainer) {
					scrollContainer.scrollTop = scrollContainer.scrollHeight;
				}
			}, 0);
		}
	});
</script>

<div class="flex h-full flex-col {className}">
	<div bind:this={scrollContainer} class="bg-background flex-1 overflow-y-auto p-4">
		<div class="mb-48 mt-16 space-y-4">
			{#each messages as message}
				<ChatMessage class="mx-auto w-full max-w-[56rem]" {message} />
			{/each}

			{#if isLoading}
				<ChatMessageLoading />
			{/if}
		</div>
	</div>
</div>
