<script lang="ts" module>
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Empty from '@ggml-org/llama-ui/ui/empty';
	import { MessagesSquare } from '@lucide/svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	const { Story } = defineMeta({
		component: Empty.Root,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama UI/Components/Display/Empty'
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
