<script lang="ts">
	import { page } from '$app/stores';
	import { onMount } from 'svelte';
	import { goto } from '$app/navigation';
	import ChatScreen from '$lib/components/chat/ChatScreen.svelte';
	import ChatHeader from '$lib/components/chat/ChatHeader.svelte';
	import { chatStore, activeChat, activeChatMessages, isLoading } from '$lib/stores/chat.svelte';
	import { slide } from 'svelte/transition';

	// Get chat ID from URL params
	const chatId = $derived($page.params.id);

	// Load chat when ID changes
	$effect(() => {
		if (chatId) {
			// Load the chat asynchronously
			(async () => {
				const success = await chatStore.loadChat(chatId);
				if (!success) {
					// Chat not found, redirect to home
					await goto('/');
				}
			})();
		}
	});
</script>

<svelte:head>
	<title>{activeChat?.name || 'Chat'} - llama.cpp</title>
</svelte:head>

<!-- Chat Header (slides in when chat is active) -->
{#if activeChat() && (activeChatMessages().length > 0 || isLoading())}
	<div in:slide={{ duration: 300, axis: 'y' }}>
		<ChatHeader />
	</div>
{/if}

<!-- Main Chat Screen -->
<ChatScreen />
