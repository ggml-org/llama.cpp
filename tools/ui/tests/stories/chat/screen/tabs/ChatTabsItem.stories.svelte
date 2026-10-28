<script lang="ts" module>
	import { SquarePen } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import ChatTabsItem from '$lib/components/app/chat/ChatTabs/ChatTabsItem.svelte';
	import { NEW_CHAT_LABEL, NEW_CHAT_TAB_ID } from '$lib/constants';

	const { Story } = defineMeta({
		component: ChatTabsItem,
		parameters: {
			layout: 'centered'
		},
		title: 'Components/ChatScreen/ChatTabs/ChatTabsItem'
	});

	const tab = { href: '#/chat/conv-1', id: 'conv-1', name: 'Rust borrow checker' };

	const handlers = {
		onActivate: (id: string) => console.log('activate', id),
		onClose: (id: string) => console.log('close', id),
		onStop: (id: string) => console.log('stop', id)
	};
</script>

<!-- idle: content is dimmed until hovered -->
<Story args={{ ...handlers, tab }} name="Default" />

<Story args={{ ...handlers, isActive: true, tab }} name="Active" />

<!-- generation running in a background tab; hover the spinner to get the stop button -->
<Story args={{ ...handlers, isLoading: true, tab }} name="Loading" />

<Story args={{ ...handlers, isActive: true, isLoading: true, tab }} name="ActiveLoading" />

<Story
	args={{
		...handlers,
		isActive: true,
		tab: { href: '#/', icon: SquarePen, id: NEW_CHAT_TAB_ID, name: NEW_CHAT_LABEL }
	}}
	name="NewChat"
/>

<Story
	args={{
		...handlers,
		isActive: true,
		tab: {
			...tab,
			name: 'A very long conversation title that has to be truncated inside the tab'
		}
	}}
	name="LongName"
/>
