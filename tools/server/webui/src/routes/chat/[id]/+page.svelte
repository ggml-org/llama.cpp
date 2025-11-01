<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { ChatScreen } from '$lib/components/app';
	import {
		chatStore,
		activeConversation,
		isLoading,
		stopGeneration
	} from '$lib/stores/chat.svelte';

	let chatId = $derived(page.params.id);
	let currentChatId: string | undefined = undefined;

	$effect(() => {
		if (chatId && chatId !== currentChatId) {
			currentChatId = chatId;

			// Skip loading if this conversation is already active (e.g., just created)
			if (activeConversation()?.id === chatId) {
				return;
			}

			(async () => {
				const success = await chatStore.loadConversation(chatId);

				if (!success) {
					await goto('#/');
				}
			})();
		}
	});

	$effect(() => {
		if (typeof window !== 'undefined') {
			const handleBeforeUnload = () => {
				if (isLoading()) {
					console.log('Page unload detected while streaming - aborting stream');
					stopGeneration();
				}
			};

			window.addEventListener('beforeunload', handleBeforeUnload);

			return () => {
				window.removeEventListener('beforeunload', handleBeforeUnload);
			};
		}
	});
</script>

<svelte:head>
	<title>{activeConversation()?.name || 'Chat'} - llama.cpp</title>
</svelte:head>

<ChatScreen />
