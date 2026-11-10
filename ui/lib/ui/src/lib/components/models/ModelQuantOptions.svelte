<!--
	The panel a model's quant rows sit in: rounded, on the raised muted surface, one hairline between rows. The shell of
	ui/apps/chat's ModelsDiscoverDetailsDownloadOptions, with the rows (ModelQuantRow) as children and an optional footer
	under them, where the app puts its draft-model pick.
-->
<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		/** The rows, one ModelQuantRow per bit depth. */
		children: Snippet;
		/** Under the rows, behind a hairline: an option that applies to every chip above. */
		footer?: Snippet;
		class?: string;
	}

	let { children, class: className = '', footer }: Props = $props();
</script>

<section
	class="@container rounded-3xl border border-border/30 bg-muted/60 shadow-sm dark:border-border/20 {className}"
>
	<div class="flex w-full flex-col divide-y divide-border/50 px-4 pb-1 dark:divide-border/35">
		{@render children()}
	</div>

	{#if footer}
		<div
			class="flex flex-wrap items-center justify-start gap-x-1.5 gap-y-1 border-t border-border/50 px-4 py-2.5 text-xs text-muted-foreground dark:border-border/35"
		>
			{@render footer()}
		</div>
	{/if}
</section>
