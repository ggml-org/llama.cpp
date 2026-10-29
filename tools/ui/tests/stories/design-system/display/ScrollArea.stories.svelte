<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { Separator } from '$lib/components/ui/separator';

	const { Story } = defineMeta({
		component: ScrollArea,
		parameters: {
			layout: 'centered'
		},
		title: 'Design system/Display/ScrollArea'
	});

	const chats = Array.from({ length: 30 }, (_, i) => `Conversation ${i + 1}`);
</script>

<Story asChild name="Default">
	<ScrollArea class="h-72 w-56 rounded-md border">
		<div class="p-4">
			<h4 class="mb-4 text-sm font-medium">Conversations</h4>

			{#each chats as chat (chat)}
				<div class="text-sm">{chat}</div>

				<Separator class="my-2" />
			{/each}
		</div>
	</ScrollArea>
</Story>
