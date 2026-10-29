<script lang="ts" module>
	import { MessagesSquare } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { Button } from '$lib/components/ui/button';
	import * as Empty from '$lib/components/ui/empty';

	const { Story } = defineMeta({
		component: Empty.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Display/Empty'
	});
</script>

<Story asChild name="Default">
	<Empty.Root class="w-96 border">
		<Empty.Header>
			<Empty.Media variant="icon">
				<MessagesSquare />
			</Empty.Media>

			<Empty.Title>No conversations yet</Empty.Title>

			<Empty.Description>Your chats are saved in this browser and show up here.</Empty.Description>
		</Empty.Header>

		<Empty.Content>
			<Button size="sm">New chat</Button>
		</Empty.Content>
	</Empty.Root>
</Story>
