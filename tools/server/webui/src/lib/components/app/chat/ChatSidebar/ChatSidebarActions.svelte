<script lang="ts">
	import { Button } from '$lib/components/ui/button';
	import { Input } from '$lib/components/ui/input';
	import { Search, SquarePen, X } from '@lucide/svelte';

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

	$effect(() => {
		if (isSearchModeActive) {
			searchInput?.focus();
		}
	});

	function handleSearchModeDeactivate() {
		isSearchModeActive = false;
		searchQuery = '';
	}
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
			class="w-full justify-start gap-2"
			href="/?new_chat=true"
			onclick={handleMobileSidebarItemClick}
			variant="ghost"
		>
			<SquarePen class="h-4 w-4" />

			New chat
		</Button>

		<Button
			class="w-full justify-start gap-2"
			onclick={() => {
				isSearchModeActive = true;
			}}
			variant="ghost"
		>
			<Search class="h-4 w-4" />

			Search conversations
		</Button>
	{/if}
</div>
