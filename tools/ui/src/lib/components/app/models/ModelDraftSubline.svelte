<script lang="ts">
	import { Check } from '@lucide/svelte';
	import { MODEL_BADGE_CLASS, MODEL_VARIANT_BADGE_CLASS } from '$lib/constants';
	import { ModelRowDownloadState } from '$lib/enums';

	interface Props {
		/**
		 * Left inset of the tree connector, tuned per surface so the elbow hangs
		 * under the main line's avatar and "Draft model" aligns with its id.
		 */
		connectorClass?: string;
		/** Draft sidecar kind, e.g. dflash. */
		kind: string;
		/** Quant of the draft, when its tag carries one. */
		quant?: string | null;
		/** Live state of the draft download. */
		state: ModelRowDownloadState;
		/** Rounded download percent, when the feed reports progress. */
		percent?: number | null;
	}

	let { connectorClass = 'ml-6', kind, percent = null, quant = null, state }: Props = $props();
</script>

<!-- the compact draft line lives inside its main item, under the id line; the
     main model's controls pause, resume and cancel its quant-matched drafts, so
     the line only reports its state -->
<div class="flex items-center gap-2 py-1 pr-2 text-sm">
	<span
		aria-hidden="true"
		class="h-2.5 w-3.5 shrink-0 self-center rounded-bl-sm border-b border-l border-border/70 {connectorClass}"
	></span>

	<span class="shrink-0 text-xs text-muted-foreground">Draft model</span>

	<span class={MODEL_VARIANT_BADGE_CLASS}>{kind}</span>

	{#if quant}
		<span class={MODEL_BADGE_CLASS}>{quant}</span>
	{/if}

	{#if state === ModelRowDownloadState.DOWNLOADED}
		<Check class="ml-auto size-icon-md shrink-0 text-green-500" />
	{:else}
		<span class="ml-auto flex shrink-0 items-center">
			{#if percent !== null}
				<span class="font-mono text-xs tabular-nums text-muted-foreground">{percent}%</span>
			{:else if state === ModelRowDownloadState.PAUSED}
				<span class="text-xs text-muted-foreground">Paused</span>
			{/if}
		</span>
	{/if}
</div>
