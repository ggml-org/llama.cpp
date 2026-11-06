<!--
	Several versions of one piece side by side in a row that scrolls sideways, snapping into place, with its id and note
	under each. Each is `itemWidth` wide (the row's width by default, and never wider), so a wide row shows several at
	once and a narrow one shows one at a time; the gap is a StateGallery's, so they line up with the columns below.
-->
<script generics="T extends { id: string; note?: string }" lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet<[T]>;
		/** each version's width */
		itemWidth?: string;
		items: T[];
		/** what the versions are of, for assistive tech */
		label: string;
	}

	let { children, items, itemWidth = '100%', label }: Props = $props();
</script>

<div
	aria-label={label}
	class="flex snap-x snap-mandatory gap-6 overflow-x-auto overscroll-x-contain pb-2"
	role="group"
>
	{#each items as item, i (item.id)}
		<div style:width="min({itemWidth}, 100%)" class="grid shrink-0 snap-start gap-1">
			{@render children(item)}

			<p class="m-0 text-xs text-muted-foreground">
				<span class="font-mono text-foreground">{item.id}</span>

				{item.note}<span class="ml-2 font-mono">{i + 1} / {items.length}</span>
			</p>
		</div>
	{/each}
</div>
