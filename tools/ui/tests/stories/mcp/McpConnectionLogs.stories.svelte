<script lang="ts" module>
	import { connectionLogs } from './fixtures';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import McpConnectionLogs from '$lib/components/app/mcp/McpConnectionLogs.svelte';

	const { Story } = defineMeta({
		component: McpConnectionLogs,
		parameters: {
			layout: 'centered'
		},
		title: 'MCP/McpConnectionLogs'
	});
</script>

<Story
	args={{ class: 'w-[32rem]', connectionTimeMs: 301, defaultExpanded: true, logs: connectionLogs }}
	name="Default"
/>
