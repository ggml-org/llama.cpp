<script lang="ts">
	import { page } from '$app/stores';
	import { goto } from '$app/navigation';
	import { beforeNavigate } from '$app/navigation';
	import { onDestroy } from 'svelte';
	import ChatScreen from '$lib/components/chat/ChatScreen.svelte';
	import ChatHeader from '$lib/components/chat/ChatHeader.svelte';
	import {
		chatStore,
		activeConversation,
		activeMessages,
		isLoading,
		stopGeneration,
		gracefulStop
	} from '$lib/stores/chat.svelte';
	import { slide } from 'svelte/transition';

	// Get chat ID from URL params
	const chatId = $derived($page.params.id);
	let currentChatId: string | undefined = undefined;

	// Navigation guard to handle streaming abortion
	beforeNavigate(async ({ cancel, to }) => {
		// Check if we're currently streaming a response
		if (isLoading()) {
			console.log(
				'Navigation detected while streaming - aborting stream and saving partial response'
			);

			// Cancel navigation temporarily to allow cleanup
			cancel();

			// Gracefully stop generation and save partial response
			await gracefulStop();

			// Now proceed with navigation
			if (to?.url) {
				await goto(to.url.pathname + to.url.search);
			}
		}
	});

	// Load chat when ID changes
	$effect(() => {
		if (chatId && chatId !== currentChatId) {
			// If we're switching chats and currently streaming, abort first
			if (isLoading()) {
				console.log('Chat switch detected while streaming - aborting stream');
				stopGeneration();
			}

			currentChatId = chatId;

			// Load the chat asynchronously
			(async () => {
				const success = await chatStore.loadConversation(chatId);
				if (!success) {
					// Chat not found, redirect to home
					await goto('/');
				}
			})();
		}
	});

	// Handle page unload (refresh, close tab, etc.)
	$effect(() => {
		if (typeof window !== 'undefined') {
			const handleBeforeUnload = (event: BeforeUnloadEvent) => {
				if (isLoading()) {
					console.log('Page unload detected while streaming - aborting stream');
					stopGeneration();
					// Note: We can't wait for async operations in beforeunload
					// but stopGeneration() will attempt to save synchronously
				}
			};

			window.addEventListener('beforeunload', handleBeforeUnload);

			return () => {
				window.removeEventListener('beforeunload', handleBeforeUnload);
			};
		}
	});

	// Cleanup on component destroy
	onDestroy(() => {
		if (isLoading()) {
			console.log('Component destroying while streaming - aborting stream');
			stopGeneration();
		}
	});
</script>

<svelte:head>
	<title>{activeConversation()?.name || 'Chat'} - llama.cpp</title>
</svelte:head>

{#if activeConversation() && (activeMessages().length > 0 || isLoading())}
	<ChatHeader />
{/if}

<ChatScreen />
