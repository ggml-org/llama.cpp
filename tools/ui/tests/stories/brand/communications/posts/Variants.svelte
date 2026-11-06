<!--
	Several versions of one piece side by side in a row that scrolls sideways, each as wide as the row and snapping into
	place, with its id and note under it. The first one fills the row, so the piece lines up with its neighbours in a
	gallery.
-->
<script generics="T extends { id: string; note?: string }" lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet<[T]>;
		items: T[];
		/** what the versions are of, for assistive tech */
		label: string;
	}

	let { children, items, label }: Props = $props();
</script>

<div
	aria-label={label}
	class="flex snap-x snap-mandatory gap-4 overflow-x-auto overscroll-x-contain pb-2"
	role="group"
>
	{#each items as item, i (item.id)}
		<div class="grid w-full shrink-0 snap-start gap-1">
			{@render children(item)}

			<p class="m-0 text-xs text-muted-foreground">
				<span class="font-mono text-foreground">{item.id}</span>

				{item.note}<span class="ml-2 font-mono">{i + 1} / {items.length}</span>
			</p>
		</div>
	{/each}
</div>
