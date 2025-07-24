<script lang="ts">
	import ChatScreen from '$lib/components/chat/ChatScreen.svelte';
	import { chatStore, isInitialized } from '$lib/stores/chat.svelte';
	import { onMount } from 'svelte';

	// Initialize the chat store when the app loads
	onMount(async () => {
		if (!isInitialized) {
			await chatStore.initialize();
		}
	});

	// Clear any active chat when on home page
	onMount(() => {
		chatStore.clearActiveChat();
	});
</script>

<svelte:head>
	<title>llama.cpp - AI Chat Interface</title>
</svelte:head>

<!-- Home page shows empty state with welcome message -->
<ChatScreen showCenteredEmpty={true} />
