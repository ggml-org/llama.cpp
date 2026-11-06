<!--
	Several versions of one piece in a carousel: the shown one, and under it a button for each (its id) with the shown
	one's note beside them, and arrows that step through them, wrapping round. The controls sit below, so the piece
	lines up with its neighbours in a gallery; one version shows at a time, so a download with it takes that one.
-->
<script generics="T extends { id: string; note?: string }" lang="ts">
	import { ChevronLeft, ChevronRight } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet<[T]>;
		items: T[];
		/** what the versions are of, for assistive tech */
		label: string;
	}

	let { children, items, label }: Props = $props();

	let index = $state(0);
	const item = $derived(items[Math.min(index, items.length - 1)]);

	const step = (by: number) => (index = (index + by + items.length) % items.length);
</script>

<div aria-label={label} aria-roledescription="carousel" class="grid gap-2" role="group">
	{#key item.id}{@render children(item)}{/key}

	<div class="flex min-w-0 items-center gap-1">
		{#each items as version, i (version.id)}
			<Button
				aria-current={i === index}
				aria-label="Version {version.id}"
				class="h-6 min-w-6 px-1.5 font-mono text-xs"
				onclick={() => (index = i)}
				size="sm"
				variant={i === index ? 'secondary' : 'ghost'}>{version.id}</Button
			>
		{/each}

		{#if item.note}<span class="ml-1 truncate text-xs text-muted-foreground">{item.note}</span>{/if}

		<span class="ml-auto flex shrink-0 items-center gap-1">
			<Button
				aria-label="Previous version"
				class="size-6"
				onclick={() => step(-1)}
				size="icon"
				variant="ghost"><ChevronLeft class="size-4" /></Button
			>

			<span class="font-mono text-xs text-muted-foreground tabular-nums"
				>{index + 1} / {items.length}</span
			>

			<Button
				aria-label="Next version"
				class="size-6"
				onclick={() => step(1)}
				size="icon"
				variant="ghost"><ChevronRight class="size-4" /></Button
			>
		</span>
	</div>
</div>
