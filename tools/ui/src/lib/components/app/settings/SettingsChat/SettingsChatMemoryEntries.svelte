<script lang="ts">
	import { X } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { MEMORY_EMPTY_INDEX, MEMORY_NAME_SEPARATOR } from '$lib/constants';
	import { MemoryService, memoryStamp } from '$lib/services/memory.service';
	import type { DatabaseMemoryEntry } from '$lib/types';
	import { onMount } from 'svelte';
	import { toast } from 'svelte-sonner';

	let entries = $state<DatabaseMemoryEntry[]>([]);

	let grouped = $derived(
		[...new Set(entries.map(groupOf))].map((group) => ({
			group,
			items: entries.filter((entry) => groupOf(entry) === group)
		}))
	);

	function groupOf(entry: DatabaseMemoryEntry): string {
		return entry.name.split(MEMORY_NAME_SEPARATOR)[0];
	}

	async function refresh() {
		entries = await MemoryService.listEntries();
	}

	async function handleDelete(name: string) {
		try {
			await MemoryService.deleteEntry(name);
			await refresh();
		} catch (err) {
			console.error('Failed to delete memory entry:', err);
			toast.error('Failed to delete memory entry');
		}
	}

	onMount(() => {
		void refresh();
	});
</script>

<div class="mt-8 min-w-0 space-y-8 border-t border-border/30 pt-6">
	{#if entries.length === 0}
		<p class="text-sm text-muted-foreground">{MEMORY_EMPTY_INDEX}</p>
	{:else}
		{#each grouped as { group, items } (group)}
			<div class="grid min-w-0 gap-1">
				<h4 class="mt-0 mb-2 text-sm font-medium capitalize">{group}</h4>

				<ul class="min-w-0">
					{#each items as entry (entry.name)}
						<li
							class="grid grid-cols-[minmax(0,2fr)_minmax(0,3fr)_auto] items-center gap-3 border-b border-border/30 py-1.5 text-sm sm:grid-cols-[minmax(0,2fr)_minmax(0,3fr)_auto_auto]"
						>
							<span class="truncate font-medium">{entry.name}</span>

							<span class="truncate text-muted-foreground">{entry.description}</span>

							<span class="hidden text-xs whitespace-nowrap text-muted-foreground sm:block">
								{memoryStamp(entry.updated)}
							</span>

							<Button
								variant="ghost"
								size="icon"
								class="h-6 w-6 text-muted-foreground hover:text-destructive"
								aria-label={`Delete ${entry.name}`}
								onclick={() => handleDelete(entry.name)}
							>
								<X class="h-4 w-4" />
							</Button>
						</li>
					{/each}
				</ul>
			</div>
		{/each}
	{/if}
</div>
