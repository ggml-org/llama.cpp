<script lang="ts">
	import { page } from '$app/state';
	import { ChatScreen, ChatTabs } from '$lib/components/app';
	import { NEW_CHAT_TAB_ID } from '$lib/constants';
	import { settingsStore, tabsStore } from '$lib/stores';

	let { children } = $props();

	// the new-chat screen is the bare `#/` route (no conversation id)
	let showCenteredEmpty = $derived(!page.params.id);

	let showTabs = $derived(
		Boolean(settingsStore.config.conversationTabs) &&
			(page.params.id || tabsStore.openTabs.some((id) => id !== NEW_CHAT_TAB_ID))
	);

	$effect(() => {
		const id = page.params.id ?? (page.route.id === '/(chat)' ? NEW_CHAT_TAB_ID : undefined);

		if (id && settingsStore.config.conversationTabs) {
			tabsStore.syncWithRoute(id);
		}
	});
</script>

<div class={showTabs ? 'md:[--chat-tabs-offset:1.25rem]' : ''}>
	{#if showTabs}
		<ChatTabs />
	{/if}

	<ChatScreen {showCenteredEmpty} />
</div>

{@render children?.()}
