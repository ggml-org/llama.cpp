<script lang="ts">
	import { page } from '$app/state';
	import { ChatScreen, ChatTabs } from '$lib/components/app';
	import { conversationsStore, tabsStore } from '$lib/stores';

	let { children } = $props();

	// the new-chat screen is a temporary conversation (unsaved) or the bare `#/` route
	let showCenteredEmpty = $derived(
		page.params.id ? conversationsStore.isTemporaryConversation(page.params.id) : true
	);

	// tabs appear only on chat-id routes; the bare `#/` new-chat view has none
	let showTabs = $derived(!!page.params.id);

	// any navigation to a conversation or new-chat tab opens a tab for it
	$effect(() => {
		const id = page.params.id;

		if (id) {
			tabsStore.syncWithRoute(id);
		}
	});
</script>

<div class={showTabs ? 'md:[--chat-tabs-height:2.5rem]' : ''}>
	{#if showTabs}
		<ChatTabs />
	{/if}

	<ChatScreen {showCenteredEmpty} />
</div>

{@render children?.()}
