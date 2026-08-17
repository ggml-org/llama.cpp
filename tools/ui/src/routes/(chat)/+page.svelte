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

		// a prompt/model deep-link routes to a new-chat tab, which sends it;
		// otherwise `#/` is a plain new-chat view with no tabs
		if (qParam !== null || modelParam !== null) {
			await conversationsStore.openNewChatTab();
		}

		await modelsStore.ensureFirstModelSelected();
	});
</script>

<svelte:head>
	<title>{APP_NAME}</title>
</svelte:head>
