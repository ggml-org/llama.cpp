<script lang="ts">
	import { page } from '$app/state';
	import { APP_NAME, URL_PARAMS } from '$lib/constants';
	import { chatStore, conversationsStore, modelsStore } from '$lib/stores';
	import { onMount } from 'svelte';

	let qParam = $derived(page.url.searchParams.get(URL_PARAMS.QUERY));
	let modelParam = $derived(page.url.searchParams.get(URL_PARAMS.MODEL));

	onMount(async () => {
		if (!conversationsStore.isInitialized) {
			await conversationsStore.initialize();
		}

		conversationsStore.clearActiveConversation();
		chatStore.clearUIState();

		await modelsStore.fetch();

		// a prompt/model deep-link opens a new chat (tab or plain view, per the
		// Conversation tabs setting); otherwise `#/` is a plain new-chat view
		if (qParam !== null || modelParam !== null) {
			await conversationsStore.openNewChat();
		}

		await modelsStore.ensureFirstModelSelected();
	});
</script>

<svelte:head>
	<title>{APP_NAME}</title>
</svelte:head>
