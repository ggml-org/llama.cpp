<script lang="ts">
	import { SettingsMcpServers } from '$lib/components/app/settings';
	import { APP_NAME } from '$lib/constants';
</script>

<svelte:head>
	<title>MCP Servers - {APP_NAME}</title>
</svelte:head>

<!-- a phone's MCP servers: the page the sidebar opens there, where a desktop or tablet opens the
     MCP servers dialog -->
<div
	class="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 pt-(--page-top) pb-16 max-md:pb-safe-24 md:px-8 md:pt-6"
>
	<div class="flex min-w-0 flex-col gap-1">
		<h1 class="text-2xl font-semibold tracking-tight">MCP Servers</h1>

		<p class="text-sm text-muted-foreground">
			Connect the servers whose tools, prompts and resources your chats can use.
		</p>
	</div>

	<SettingsMcpServers />
</div>
