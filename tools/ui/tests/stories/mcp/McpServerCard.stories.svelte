<script lang="ts" module>
	import StateGallery from '../helpers/StateGallery.svelte';
	import StateGalleryItem from '../helpers/StateGalleryItem.svelte';
	import { seedServerHealth, servers } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import McpServerCard from '$lib/components/app/mcp/McpServerCard/McpServerCard.svelte';

	const { Story } = defineMeta({
		component: McpServerCard,
		parameters: {
			layout: 'centered'
		},
		title: 'MCP/McpServerCard'
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
