<script lang="ts">
	import ChatScreen from '$lib/components/chat/ChatScreen.svelte';
	import { chatStore, isInitialized } from '$lib/stores/chat.svelte';
	import { page } from '$app/stores';
	import { onMount } from 'svelte';

	onMount(async () => {
		if (!isInitialized) {
			await chatStore.initialize();
		}
	});

	onMount(() => {
		chatStore.clearActiveChat();
	});

	// Check if we're in new chat mode
	const isNewChatMode = $derived($page.url.searchParams.get('new_chat') === 'true');
</script>

<svelte:head>
	<title>llama.cpp - AI Chat Interface</title>
</svelte:head>

<ChatScreen showCenteredEmpty={true} />
