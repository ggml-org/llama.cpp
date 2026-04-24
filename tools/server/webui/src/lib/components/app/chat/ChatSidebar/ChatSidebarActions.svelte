<script lang="ts">
	import { Search, SquarePen, X } from '@lucide/svelte';
	import { KeyboardShortcutInfo } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { McpLogo } from '$lib/components/app';
	import { SETTINGS_SECTION_TITLES } from '$lib/constants';
	import { getChatSettingsDialogContext } from '$lib/contexts';

	interface Props {
		handleMobileSidebarItemClick: () => void;
		isSearchModeActive: boolean;
		searchQuery: string;
	}

	let {
		handleMobileSidebarItemClick,
		isSearchModeActive = $bindable(),
		searchQuery = $bindable()
	}: Props = $props();

	let searchInput: HTMLInputElement | null = $state(null);

	const chatSettingsDialog = getChatSettingsDialogContext();

	function handleSearchModeDeactivate() {
		isSearchModeActive = false;
		searchQuery = '';
	}

	$effect(() => {
		if (isSearchModeActive) {
			searchInput?.focus();
		}
	});
</script>

<div class="my-1 space-y-1">
	{#if isSearchModeActive}
		<div class="relative">
			<Search class="absolute top-2.5 left-2 h-4 w-4 text-muted-foreground" />

			<Input
				bind:ref={searchInput}
				bind:value={searchQuery}
				onkeydown={(e) => e.key === 'Escape' && handleSearchModeDeactivate()}
				placeholder="Search conversations..."
				class="pl-8"
			/>

			<X
				class="cursor-pointertext-muted-foreground absolute top-2.5 right-2 h-4 w-4"
				onclick={handleSearchModeDeactivate}
			/>
		</div>
	{:else}
		<Button
			class="w-full justify-between backdrop-blur-none! hover:[&>kbd]:opacity-100"
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
			class="w-full justify-between backdrop-blur-none! hover:[&>kbd]:opacity-100"
			onclick={() => {
				isSearchModeActive = true;
			}}
			variant="ghost"
		>
			<div class="flex items-center gap-2">
				<Search class="h-4 w-4" />

				Search
			</div>

			<KeyboardShortcutInfo keys={['cmd', 'k']} />
		</Button>

		<Button
			class="w-full justify-between backdrop-blur-none! hover:[&>kbd]:opacity-100"
			onclick={() => {
				chatSettingsDialog.open(SETTINGS_SECTION_TITLES.MCP);
			}}
			variant="ghost"
		>
			<div class="flex items-center gap-2">
				<McpLogo class="h-4 w-4" />

				MCP Servers
			</div>
		</Button>
	{/if}
</div>
