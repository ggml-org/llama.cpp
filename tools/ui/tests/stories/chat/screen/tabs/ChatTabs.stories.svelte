<script lang="ts" module>
	import { tabConversations } from '../fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatTabs from '$lib/components/app/chat/ChatTabs/ChatTabs.svelte';
	import { NEW_CHAT_TAB_ID } from '$lib/constants';
	import { chatStore, conversationsStore, tabsStore } from '$lib/stores';

	const { Story } = defineMeta({
		component: ChatTabs,
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Components/ChatScreen/ChatTabs/ChatTabs'
	});

	const chatRoute = (id: string) => ({
		sveltekit_experimental: {
			state: { page: { params: { id }, route: { id: '/(chat)/chat/[id]' } } }
		}
	});

	const newChatRoute = {
		sveltekit_experimental: { state: { page: { params: {}, route: { id: '/(chat)' } } } }
	};

	function seedTabs(openTabs: string[], loadingIds: string[] = []) {
		conversationsStore.conversations = tabConversations;
		tabsStore.openTabs = openTabs;

		for (const conv of tabConversations) {
			chatStore.setChatLoading(conv.id, loadingIds.includes(conv.id));
		}
	}
</script>

<Story
	asChild
	beforeEach={() => seedTabs(['conv-1', 'conv-2', 'conv-3'])}
	name="Default"
	parameters={chatRoute('conv-2')}
>
	<div class="w-[56rem]" style="--sidebar-width: 0px">
		<ChatTabs />
	</div>
</Story>

<!-- a new-chat tab is open, so the New chat button is hidden -->
<Story
	asChild
	beforeEach={() => seedTabs(['conv-1', 'conv-2', NEW_CHAT_TAB_ID])}
	name="NewChatTabActive"
	parameters={newChatRoute}
>
	<div class="w-[56rem]" style="--sidebar-width: 0px">
		<ChatTabs />
	</div>
</Story>

<Story
	asChild
	beforeEach={() => seedTabs(['conv-1', 'conv-2', 'conv-3', 'conv-5'], ['conv-1', 'conv-3'])}
	name="WithLoadingTabs"
	parameters={chatRoute('conv-1')}
>
	<div class="w-[56rem]" style="--sidebar-width: 0px">
		<ChatTabs />
	</div>
</Story>

<!-- more tabs than fit: edge fades show and the active tab scrolls into view -->
<Story
	asChild
	beforeEach={() =>
		seedTabs(
			tabConversations.map((c) => c.id),
			['conv-6']
		)}
	name="ManyTabsOverflow"
	parameters={chatRoute('conv-9')}
>
	<div class="w-[40rem]" style="--sidebar-width: 0px">
		<ChatTabs />
	</div>
</Story>

<!-- conversation id not in the store: falls back to the unnamed label -->
<Story
	asChild
	beforeEach={() => seedTabs(['conv-1', 'conv-missing'])}
	name="UnnamedChat"
	parameters={chatRoute('conv-missing')}
>
	<div class="w-[56rem]" style="--sidebar-width: 0px">
		<ChatTabs />
	</div>
</Story>
