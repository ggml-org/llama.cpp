<script lang="ts">
	import { Settings } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { McpLogo } from '$lib/components/app';
	// import { getChatSettingsDialogContext, getMcpServersDialogContext } from '$lib/contexts';
	import { page } from '$app/state';

	// const chatSettingsDialog = getChatSettingsDialogContext();
	// const mcpServersDialog = getMcpServersDialogContext();

	let isMcpActive = $derived(page.route.id === '/settings/mcp');
	let isSettingsActive = $derived(page.route.id === '/settings/chat');
</script>

<div class="space-y-1 pt-0">
	<Button
		class="w-full justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100 {isMcpActive
			? 'bg-accent text-accent-foreground'
			: ''}"
		href="#/settings/mcp"
		variant="ghost"
	>
		<div class="flex items-center gap-2">
			<McpLogo class="h-4 w-4" />

			MCP Servers
		</div>
	</Button>

	<Button
		class="w-full justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100 {isSettingsActive
			? 'bg-accent text-accent-foreground'
			: ''}"
		href="#/settings/chat"
		variant="ghost"
	>
		<div class="flex items-center gap-2">
			<Settings class="h-4 w-4" />

			Settings
		</div>
	</Button>
</div>
