<script module lang="ts">
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import SidebarNavigationConversationItem from '$lib/components/app/navigation/SidebarNavigation/SidebarNavigationConversationItem.svelte';
	import { expect } from 'storybook/test';

	const mockForkedConversation: DatabaseConversation = {
		id: 'conv-2',
		name: 'Forked Conversation',
		lastModified: Date.now(),
		currNode: 'msg-2',
		forkedFromConversationId: 'conv-1'
	};

	const { Story } = defineMeta({
		title: 'Components/SidebarNavigationConversationItem/Accessibility',
		component: SidebarNavigationConversationItem,
		parameters: {
			layout: 'centered'
		},
		tags: ['!dev']
	});
</script>

<Story
	name="ForkIconSingleTabStop"
	args={{ conversation: mockForkedConversation, depth: 1 }}
	play={async ({ canvas, userEvent }) => {
		const row = await canvas.findByRole('button', { name: /Forked Conversation/ });
		const forkIcon = await canvas.findByRole('link');

		row.focus();
		await userEvent.tab();

		await expect(forkIcon).toHaveFocus();
	}}
/>
