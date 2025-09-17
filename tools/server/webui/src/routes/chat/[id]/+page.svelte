<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { beforeNavigate } from '$app/navigation';
	import { ChatScreen } from '$lib/components/app';
	import {
		chatStore,
		activeConversation,
		isLoading,
		stopGeneration,
		gracefulStop
	} from '$lib/stores/chat.svelte';
	import { onDestroy } from 'svelte';

	let chatId = $derived(page.params.id);
	let currentChatId: string | undefined = undefined;

	beforeNavigate(async ({ cancel, to }) => {
		if (isLoading()) {
			console.log(
				'Navigation detected while streaming - aborting stream and saving partial response'
			);

			cancel();

			await gracefulStop();

			if (to?.url) {
				await goto(to.url.pathname + to.url.search);
			}
		}
	});

	$effect(() => {
		if (chatId && chatId !== currentChatId) {
			if (isLoading()) {
				console.log('Chat switch detected while streaming - aborting stream');
				stopGeneration();
			}

			currentChatId = chatId;

			(async () => {
				const success = await chatStore.loadConversation(chatId);

				if (!success) {
					await goto('/');
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

	onDestroy(() => {
		if (isLoading()) {
			stopGeneration();
		}
	});
</script>

<svelte:head>
	<title>{activeConversation()?.name || 'Chat'} - llama.cpp</title>
</svelte:head>

<ChatScreen />
