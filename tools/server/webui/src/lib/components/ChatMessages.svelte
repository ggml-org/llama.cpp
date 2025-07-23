<script lang="ts">
	import ChatMessage from './ChatMessage.svelte';
	import ChatMessageLoading from './ChatMessageLoading.svelte';
	let {
		messages = [],
		isLoading = false
	}: {
		messages?: ChatMessageData[];
		isLoading?: boolean;
	} = $props();

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

<div class="flex h-full flex-col">
	<div bind:this={scrollContainer} class="bg-background flex-1 overflow-y-auto p-4">
		<div class="space-y-4">
			{#each messages as message, i}
				<ChatMessage {message} />
			{/each}

			{#if isLoading}
				<ChatMessageLoading />
			{/if}
		</div>
	</div>
</div>
