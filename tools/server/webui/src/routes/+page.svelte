<script lang="ts">
	import ChatScreen from '$lib/components/chat/ChatScreen.svelte';
	import { chatStore, isInitialized } from '$lib/stores/chat.svelte';
	import { onMount } from 'svelte';

	onMount(async () => {
		if (!isInitialized) {
			await chatStore.initialize();
		}
	});

	onMount(() => {
		chatStore.clearActiveConversation();
	});
</script>

<svelte:head>
	<title>llama.cpp - AI Chat Interface</title>
</svelte:head>

<ChatScreen showCenteredEmpty={true} />
