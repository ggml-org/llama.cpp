<script lang="ts">
	import { Search, SquarePen, X, Download, Upload } from '@lucide/svelte';
	import { KeyboardShortcutInfo } from '$lib/components/app';
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { exportAllConversations, importConversations } from '$lib/stores/chat.svelte';

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

<div class="space-y-0.5">
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
			class="w-full justify-between hover:[&>kbd]:opacity-100"
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
			class="w-full justify-between hover:[&>kbd]:opacity-100"
			onclick={() => {
				isSearchModeActive = true;
			}}
			variant="ghost"
		>
			<div class="flex items-center gap-2">
				<Search class="h-4 w-4" />
				Search conversations
			</div>

			<KeyboardShortcutInfo keys={['cmd', 'k']} />
		</Button>

		<Button
			class="w-full justify-start text-sm"
			onclick={() => {
				importConversations().catch((err) => {
					console.error('Import failed:', err);
					// Optional: show toast or dialog
				});
			}}
			variant="ghost"
		>
			<div class="flex items-center gap-2">
				<Upload class="h-4 w-4" />
				Import conversations
			</div>
		</Button>

		<Button
			class="w-full justify-start text-sm"
			onclick={() => {
				exportAllConversations();
			}}
			variant="ghost"
		>
			<div class="flex items-center gap-2">
				<Download class="h-4 w-4" />
				Export all conversations
			</div>
		</Button>
	{/if}
</div>
