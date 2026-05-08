<script lang="ts">
	import { goto, replaceState } from '$app/navigation';
	import { page } from '$app/state';
	import { afterNavigate } from '$app/navigation';
	import { DialogModelNotAvailable } from '$lib/components/app';
	import { ROUTES } from '$lib/constants/routes';
	import { chatStore, isLoading } from '$lib/stores/chat.svelte';
	import { conversationsStore, activeConversation } from '$lib/stores/conversations.svelte';
	import { modelsStore, modelOptions } from '$lib/stores/models.svelte';

	let chatId = $derived(page.params.id);
	let currentChatId: string | undefined = undefined;

	// URL parameters for prompt and model selection
	let qParam = $derived(page.url.searchParams.get('q'));
	let modelParam = $derived(page.url.searchParams.get('model'));

	// Dialog state for model not available error
	let showModelNotAvailable = $state(false);
	let requestedModelName = $state('');
	let availableModelNames = $derived(modelOptions().map((m) => m.model));

	// Track if URL params have been processed for this chat
	let urlParamsProcessed = $state(false);

	/**
	 * Clear URL params after message is sent to prevent re-sending on refresh
	 */
	function clearUrlParams() {
		const url = new URL(page.url);
		url.searchParams.delete('q');
		url.searchParams.delete('model');
		replaceState(url.toString(), {});
	}

	async function handleUrlParams() {
		// Ensure models are loaded first
		await modelsStore.fetch();

		// Handle model parameter - select model if provided
		if (modelParam) {
			const model = modelsStore.findModelByName(modelParam);
			if (model) {
				try {
					await modelsStore.selectModelById(model.id);
				} catch (error) {
					console.error('Failed to select model:', error);
					requestedModelName = modelParam;
					showModelNotAvailable = true;
					return;
				}
			} else {
				// Model not found - show error dialog
				requestedModelName = modelParam;
				showModelNotAvailable = true;
				return;
			}
		}

		// Handle ?q= parameter - send message in current conversation
		if (qParam !== null) {
			await chatStore.sendMessage(qParam);
			// Clear URL params after message is sent
			clearUrlParams();
		} else if (modelParam) {
			// Clear params even if no message was sent (just model selection)
			clearUrlParams();
		}

		urlParamsProcessed = true;
	}

	afterNavigate(() => {
		setTimeout(() => {
			void modelsStore.selectModelFromLastAssistantResponse();
		}, 100);
	});

	$effect(() => {
		if (chatId && chatId !== currentChatId) {
			currentChatId = chatId;
			urlParamsProcessed = false; // Reset for new chat

			// Skip loading if this conversation is already active (e.g., just created)
			if (activeConversation()?.id === chatId) {
				// Still handle URL params even if conversation is active
				if ((qParam !== null || modelParam !== null) && !urlParamsProcessed) {
					handleUrlParams();
				}
				return;
			}

			(async () => {
				const success = await conversationsStore.loadConversation(chatId);
				if (success) {
					chatStore.syncLoadingStateForChat(chatId);

					// Handle URL params after conversation is loaded
					if ((qParam !== null || modelParam !== null) && !urlParamsProcessed) {
						await handleUrlParams();
					}
				} else {
					await goto(ROUTES.START);
				}
			})();
		}
	});

	$effect(() => {
		if (typeof window !== 'undefined') {
			const handleBeforeUnload = () => {
				if (isLoading()) {
					console.log('Page unload detected while streaming - aborting stream');
					chatStore.stopGeneration();
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

<DialogModelNotAvailable
	bind:open={showModelNotAvailable}
	modelName={requestedModelName}
	availableModels={availableModelNames}
/>
