<script lang="ts" module>
	import { capabilities } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import McpCapabilitiesBadges from '$lib/components/app/mcp/McpCapabilitiesBadges.svelte';

	const { Story } = defineMeta({
		component: McpCapabilitiesBadges,
		parameters: {
			layout: 'centered'
		},
		title: 'MCP/McpCapabilitiesBadges'
	});
</script>

<Story args={{ capabilities, onBrowseResources: () => console.log('browse') }} name="Default" />
