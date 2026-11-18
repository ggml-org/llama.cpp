<script lang="ts" module>
	import { seedServerHealth, servers } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import McpServerCard from '#lib/components/app/mcp/McpServerCard/McpServerCard.svelte';
	import StateGallery from '$llama-ui-stories/helpers/StateGallery.svelte';
	import StateGalleryItem from '$llama-ui-stories/helpers/StateGalleryItem.svelte';

	const { Story } = defineMeta({
		component: McpServerCard,
		parameters: {
			layout: 'centered'
		},
		title: 'Llama Chat/MCP/McpServerCard'
	});

	const handlers = {
		onDelete: () => console.log('delete'),
		onToggle: (enabled: boolean) => console.log('toggle', enabled),
		onUpdate: (updates: object) => console.log('update', updates)
	};
</script>

<Story asChild beforeEach={seedServerHealth} name="Default">
	<StateGallery columns={2} minItemWidth="24rem">
		<StateGalleryItem label="Connected">
			<McpServerCard {...handlers} enabled server={servers.connected} />
		</StateGalleryItem>

		<StateGalleryItem label="Connecting">
			<McpServerCard {...handlers} enabled server={servers.connecting} />
		</StateGalleryItem>

		<StateGalleryItem label="Error">
			<McpServerCard {...handlers} enabled server={servers.error} />
		</StateGalleryItem>

		<StateGalleryItem label="Disabled">
			<McpServerCard {...handlers} enabled={false} server={servers.disabled} />
		</StateGalleryItem>
	</StateGallery>
</Story>
