<script lang="ts" module>
	import StateGallery from '../../../helpers/StateGallery.svelte';
	import StateGalleryItem from '../../../helpers/StateGalleryItem.svelte';
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

<Story asChild name="States">
	<StateGallery columns={3}>
		<StateGalleryItem label="Idle" note="dimmed until hovered">
			<ChatTabsItem {...handlers} {tab} />
		</StateGalleryItem>

		<StateGalleryItem label="Active">
			<ChatTabsItem {...handlers} isActive {tab} />
		</StateGalleryItem>

		<StateGalleryItem label="New chat">
			<ChatTabsItem
				{...handlers}
				isActive
				tab={{ href: '#/', icon: SquarePen, id: NEW_CHAT_TAB_ID, name: NEW_CHAT_LABEL }}
			/>
		</StateGalleryItem>

		<StateGalleryItem label="Loading" note="hover the spinner to stop">
			<ChatTabsItem {...handlers} isLoading {tab} />
		</StateGalleryItem>

		<StateGalleryItem label="Active, loading">
			<ChatTabsItem {...handlers} isActive isLoading {tab} />
		</StateGalleryItem>

		<StateGalleryItem label="Long name" note="truncated">
			<ChatTabsItem
				{...handlers}
				isActive
				tab={{
					...tab,
					name: 'A very long conversation title that has to be truncated inside the tab'
				}}
			/>
		</StateGalleryItem>
	</StateGallery>
</Story>
