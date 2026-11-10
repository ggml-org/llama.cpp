<!-- One labelled cell of a StateGallery. `span` lets a wide state take several columns, `full` the whole row. -->
<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		children: Snippet;
		/** the gallery's whole width, past its columns when they don't fill it */
		full?: boolean;
		label: string;
		note?: string;
		span?: number;
	}

	let { children, full = false, label, note, span = 1 }: Props = $props();
</script>

<figure
	style:grid-column={full ? '1 / -1' : `span ${span}`}
	style:width={full ? '100cqw' : undefined}
	class="m-0 flex min-w-0 flex-col gap-2"
>
	<figcaption class="flex items-baseline gap-2 text-[11px] font-medium tracking-wide uppercase">
		<span class="text-muted-foreground">{label}</span>

		{#if note}
			<span class="truncate font-normal tracking-normal text-muted-foreground/70 normal-case"
				>{note}</span
			>
		{/if}
	</figcaption>

	<div class="min-w-0">
		{@render children()}
	</div>
</figure>
