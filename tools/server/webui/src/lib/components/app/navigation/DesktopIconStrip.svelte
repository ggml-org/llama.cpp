<script lang="ts">
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { ActionIcon } from '$lib/components/app/actions';
	import { McpLogo } from '$lib/components/app/mcp';
	import { Database, Settings, Search, SquarePen } from '@lucide/svelte';

	interface Props {
		sidebarOpen: boolean;
		onSearchClick: () => void;
	}

	let { sidebarOpen, onSearchClick }: Props = $props();

	let isMcpActive = $derived(page.route.id === '/settings/mcp');
	let isImportExportActive = $derived(page.route.id === '/settings/import-export');
	let isSettingsActive = $derived(page.route.id === '/settings/chat');
</script>

<!-- Spacer to reserve space for icon strip in the flex layout -->
<div
	class="hidden shrink-0 transition-[width] duration-200 ease-linear md:block {sidebarOpen
		? 'w-0'
		: 'w-[calc(var(--sidebar-width-icon)+1.5rem)]'}"
></div>
<!-- Desktop: icon strip, fixed position so it stays stationary and only fades -->
<aside
	class="fixed top-0 bottom-0 left-0 z-10 hidden w-[calc(var(--sidebar-width-icon)+1.5rem)] flex-col items-center justify-between py-3 transition-opacity duration-200 ease-linear md:flex {sidebarOpen
		? 'pointer-events-none opacity-0'
		: 'opacity-100'}"
>
	<div class="mt-12 flex flex-col items-center gap-1">
		<ActionIcon
			icon={SquarePen}
			tooltip="New Chat"
			size="lg"
			iconSize="h-4 w-4"
			class="h-9 w-9 rounded-full hover:bg-accent!"
			onclick={() => goto('?new_chat=true#/')}
		/>

		<ActionIcon
			icon={Search}
			tooltip="Search"
			size="lg"
			iconSize="h-4 w-4"
			class="h-9 w-9 rounded-full hover:bg-accent!"
			onclick={onSearchClick}
		/>
	</div>
	<div class="flex flex-col items-center gap-1">
		<ActionIcon
			icon={McpLogo}
			tooltip="MCP Servers"
			size="lg"
			iconSize="h-4 w-4"
			class="h-9 w-9 rounded-full hover:bg-accent! {isMcpActive
				? 'bg-accent text-accent-foreground'
				: ''}"
			onclick={() => goto('#/settings/mcp')}
		/>
		<ActionIcon
			icon={Database}
			tooltip="Import / Export"
			size="lg"
			iconSize="h-4 w-4"
			class="h-9 w-9 rounded-full hover:bg-accent! {isImportExportActive
				? 'bg-accent text-accent-foreground'
				: ''}"
			onclick={() => goto('#/settings/import-export')}
		/>
		<ActionIcon
			icon={Settings}
			tooltip="Settings"
			size="lg"
			iconSize="h-4 w-4"
			class="h-9 w-9 rounded-full hover:bg-accent! {isSettingsActive
				? 'bg-accent text-accent-foreground'
				: ''}"
			onclick={() => goto('#/settings/chat')}
		/>
	</div>
</aside>
