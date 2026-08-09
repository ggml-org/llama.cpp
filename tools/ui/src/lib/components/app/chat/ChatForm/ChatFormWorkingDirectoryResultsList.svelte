<script lang="ts">
	import { Folder } from '@lucide/svelte';
	import { cn } from '$lib/components/ui/utils';
	import { highlightMatch } from '$lib/utils';
	import { fly } from 'svelte/transition';

	// Fly-in transition for the results list.
	const FLY_Y_PX = -4;
	const FLY_DURATION_MS = 100;

	interface Props {
		results: string[];
		hoveredIndex: number;
		isSearching: boolean;
		error: string | null;
		rawQuery: string;
		container?: HTMLDivElement | null;
		onCommit?: (path: string) => void;
		onHover?: (index: number) => void;
	}

	let {
		container = $bindable(null),
		error,
		hoveredIndex,
		isSearching,
		onCommit,
		onHover,
		rawQuery,
		results
	}: Props = $props();
</script>

<div
	bind:this={container}
	class="max-h-48 overflow-y-auto py-2"
	transition:fly={{ duration: FLY_DURATION_MS, y: FLY_Y_PX }}
>
	{#if isSearching && results.length === 0}
		<div class="px-2 py-1.5 text-sm text-muted-foreground">Searching...</div>
	{:else if error}
		<div class="px-2 py-1.5 text-sm text-destructive">{error}</div>
	{:else if results.length === 0}
		<div class="px-2 py-1.5 text-sm text-muted-foreground">No matching folders</div>
	{:else}
		{#each results as path, index (path)}
			<button
				type="button"
				data-result-index={index}
				data-highlighted={index === hoveredIndex ? '' : undefined}
				class={cn(
					'relative flex w-full cursor-pointer items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-hidden select-none data-highlighted:bg-accent data-highlighted:text-accent-foreground'
				)}
				onclick={() => onCommit?.(path)}
				onmouseenter={() => onHover?.(index)}
			>
				<Folder class="size-4 shrink-0 text-muted-foreground" />
				<span class="min-w-0 flex-1 truncate font-mono text-left">
					{#each highlightMatch(path, rawQuery.trim()) as seg, segIndex (segIndex)}
						{#if seg.match}
							<mark class="rounded bg-yellow-200/60 px-0.5 text-foreground dark:bg-yellow-500/30"
								>{seg.text}</mark
							>
						{:else}
							{seg.text}
						{/if}
					{/each}
				</span>
			</button>
		{/each}
	{/if}
</div>
