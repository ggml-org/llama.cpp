<script lang="ts">
	import { beforeNavigate, goto, replaceState } from '$app/navigation';
	import { page } from '$app/state';
	import { afterNavigate } from '$app/navigation';
	import { Trash2 } from '@lucide/svelte';
	import { DialogConfirmation, DialogModelNotAvailable } from '$lib/components/app';
	import { APP_NAME, ROUTES } from '$lib/constants';
	import { MessageRole } from '$lib/enums';
	import { chatStore, isLoading } from '$lib/stores/chat.svelte';
	import {
		activeConversation,
		activeMessages,
		conversationsStore
	} from '$lib/stores/conversations.svelte';
	import { modelsStore, modelOptions } from '$lib/stores/models.svelte';

	let chatId = $derived(page.params.id);
	let currentChatId: string | undefined = undefined;

	let qParam = $derived(page.url.searchParams.get('q'));
	let modelParam = $derived(page.url.searchParams.get('model'));

	let showModelNotAvailable = $state(false);
	let requestedModelName = $state('');
	let availableModelNames = $derived(modelOptions().map((m) => m.model));

	let urlParamsProcessed = $state(false);

	// Pending navigation interrupted by the discard-conversation confirmation.
	// The original URL is replayed once the user confirms and the conversation is deleted.
	let pendingNavigationUrl: URL | null = null;

	let showDiscardDialog = $state(false);

	function clearUrlParams() {
		const url = new URL(page.url);
		url.searchParams.delete('q');
		url.searchParams.delete('model');
		replaceState(url.toString(), {});
	}

	async function handleUrlParams() {
		await modelsStore.fetch();

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
				requestedModelName = modelParam;
				showModelNotAvailable = true;
				return;
			}
		}

		if (qParam !== null) {
			await chatStore.sendMessage(qParam);
			clearUrlParams();
		} else if (modelParam) {
			clearUrlParams();
		}

		urlParamsProcessed = true;
	}

	afterNavigate(() => {
		setTimeout(() => {
			void modelsStore.selectModelFromLastAssistantResponse();
		}, 100);
	});

	// A conversation with only a system message is an unsaved draft; leaving
	// it should prompt before discarding the work.
	function isUnsavedSystemOnlyConversation(): boolean {
		const messages = activeMessages();
		return !isLoading() && messages.length === 1 && messages[0].role === MessageRole.SYSTEM;
	}

	beforeNavigate(({ to, cancel }) => {
		if (to?.url && to.url.href !== page.url.href && isUnsavedSystemOnlyConversation()) {
			pendingNavigationUrl = to.url;
			showDiscardDialog = true;
			cancel();
		}
	});

	async function handleDiscardConfirm() {
		showDiscardDialog = false;
		const targetUrl = pendingNavigationUrl;
		pendingNavigationUrl = null;

		if (!chatId) return;

		// Skip the store's default NEW_CHAT navigation so we can jump straight to
		// the user's original destination without a start-page flash.
		await conversationsStore.deleteConversation(chatId, { skipNavigation: true });

		const target = targetUrl && targetUrl.href !== page.url.href ? targetUrl : ROUTES.START;
		await goto(target);
	}

	function handleDiscardCancel() {
		showDiscardDialog = false;
		pendingNavigationUrl = null;
	}

	$effect(() => {
		if (chatId && chatId !== currentChatId) {
			currentChatId = chatId;
			urlParamsProcessed = false;

			if (activeConversation()?.id === chatId) {
				void chatStore.discoverActiveStream(chatId);
				if ((qParam !== null || modelParam !== null) && !urlParamsProcessed) {
					handleUrlParams();
				}
				return;
			}

			(async () => {
				const success = await conversationsStore.loadConversation(chatId);
				if (!success) {
					await goto(ROUTES.START);
					return;
				}
				chatStore.syncLoadingStateForChat(chatId);
				await chatStore.discoverActiveStream(chatId);

				if ((qParam !== null || modelParam !== null) && !urlParamsProcessed) {
					await handleUrlParams();
				}
			})();
		}
	});

	$effect(() => {
		if (typeof window === 'undefined' || typeof document === 'undefined') return;

		// Re-run discovery on visibilitychange to catch races where the initial
		// mount probe missed an active session
		const onVisibility = () => {
			if (document.visibilityState !== 'visible') return;
			if (!chatId) return;
			void chatStore.discoverActiveStream(chatId);
		};
		document.addEventListener('visibilitychange', onVisibility);
		return () => document.removeEventListener('visibilitychange', onVisibility);
	});
</script>

<svelte:head>
	<title>{activeConversation()?.name || 'Chat'} - {APP_NAME}</title>
</svelte:head>

<DialogModelNotAvailable
	bind:open={showModelNotAvailable}
	modelName={requestedModelName}
	availableModels={availableModelNames}
/>

<DialogConfirmation
	bind:open={showDiscardDialog}
	title="Discard conversation?"
	description="This conversation only has a system message and no chat history yet. Leaving will discard it."
	confirmText="Discard"
	cancelText="Cancel"
	variant="destructive"
	icon={Trash2}
	onConfirm={handleDiscardConfirm}
	onCancel={handleDiscardCancel}
/>
