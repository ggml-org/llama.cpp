<script lang="ts">
	import { ChatScreen, ChatHeader } from '$lib/components';
	import { chatMessages, isLoading } from '$lib/stores/chat.svelte';
	import { slide } from 'svelte/transition';

	// Show header only when there are messages or loading
	const hasMessages = $derived(chatMessages.length > 0 || isLoading);
</script>

{#if hasMessages}
	<div in:slide={{ duration: 400, axis: 'y' }}>
		<ChatHeader />
	</div>
{/if}

<ChatScreen />
