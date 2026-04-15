<script lang="ts">
	import { Database, Search, Settings, SquarePen } from '@lucide/svelte';
	import { KeyboardShortcutInfo } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { McpLogo, SearchInput } from '$lib/components/app';
	import { page } from '$app/state';

	interface Props {
		handleMobileSidebarItemClick: () => void;
		isSearchModeActive: boolean;
		searchQuery: string;
		isCancelAlwaysVisible?: boolean;
	}

	let {
		handleMobileSidebarItemClick,
		isSearchModeActive = $bindable(),
		searchQuery = $bindable(),
		isCancelAlwaysVisible = false
	}: Props = $props();

	let searchInputRef = $state<HTMLInputElement | null>(null);
	let isMcpActive = $derived(page.route.id === '/settings/mcp');
	let isSettingsActive = $derived(!!page.route.id?.startsWith('/settings/chat'));
	let isImportExportActive = $derived(page.route.id === '/settings/import-export');

	function handleSearchModeDeactivate() {
		isSearchModeActive = false;
		searchQuery = '';
	}

	export function activateSearch() {
		isSearchModeActive = true;
		// Focus after Svelte renders the input
		queueMicrotask(() => searchInputRef?.focus());
	}
</script>

<div class="my-1 space-y-1">
	{#if isSearchModeActive}
		<SearchInput
			bind:value={searchQuery}
			bind:ref={searchInputRef}
			onClose={handleSearchModeDeactivate}
			onKeyDown={(e) => e.key === 'Escape' && handleSearchModeDeactivate()}
			placeholder="Search conversations..."
			{isCancelAlwaysVisible}
		/>
	{:else}
		<Button
			class="w-full justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100"
			href="?new_chat=true#/"
			onclick={handleMobileSidebarItemClick}
			variant="ghost"
		>
			<div class="flex items-center gap-2">
				<SquarePen class="h-4 w-4" />

				New chat
			</div>

			<KeyboardShortcutInfo keys={['shift', 'cmd', 'o']} />
		</Button>

		<Button
			class="w-full justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100"
			onclick={activateSearch}
			variant="ghost"
		>
			<div class="flex items-center gap-2">
				<Search class="h-4 w-4" />

				Search
			</div>

			<KeyboardShortcutInfo keys={['cmd', 'k']} />
		</Button>
		<Button
			class="w-full justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100 {isMcpActive
				? 'bg-accent text-accent-foreground'
				: ''}"
			href="#/settings/mcp"
			onclick={handleMobileSidebarItemClick}
			variant="ghost"
		>
			<div class="flex items-center gap-2">
				<McpLogo class="h-4 w-4" />

				MCP Servers
			</div>
		</Button>

		<Button
			class="w-full justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100 {isImportExportActive
				? 'bg-accent text-accent-foreground'
				: ''}"
			href="#/settings/import-export"
			onclick={handleMobileSidebarItemClick}
			variant="ghost"
		>
			<div class="flex items-center gap-2">
				<Database class="h-4 w-4" />

				Import / Export
			</div>
		</Button>

		<Button
			class="w-full justify-between px-2 backdrop-blur-none! hover:[&>kbd]:opacity-100 {isSettingsActive
				? 'bg-accent text-accent-foreground'
				: ''}"
			href="#/settings/chat/general"
			onclick={handleMobileSidebarItemClick}
			variant="ghost"
		>
			<div class="flex items-center gap-2">
				<Settings class="h-4 w-4" />

				Settings
			</div>
		</Button>
	{/if}
</div>
