<!--
	A headline block: an optional Mono label over the headline (at headline or display size), the lede under it and
	use cases as chips. The text half of most layouts.
-->
<script lang="ts">
	import PostChip from './PostChip.svelte';

	interface Props {
		chips?: string[];
		headline: string;
		label?: string;
		lede?: string;
		/** the headline's place in the type scale */
		level?: 'display' | 'headline';
	}

	let { chips = [], headline, label, lede, level = 'headline' }: Props = $props();
</script>

<div class="block">
	{#if label}<p class="t-label">{label}</p>{/if}

	<h2 class="t-{level}">{headline}</h2>

	{#if lede}<p class="t-lede">{lede}</p>{/if}

	{#if chips.length}
		<div class="chips">
			{#each chips as chip (chip)}<PostChip>{chip}</PostChip>{/each}
		</div>
	{/if}
</div>

<style>
	.block {
		display: grid;
		gap: calc(var(--u) * 0.6);
		justify-items: start;
	}
	.t-label {
		margin-bottom: calc(var(--u) * 0.1);
	}
	.chips {
		display: flex;
		flex-wrap: wrap;
		gap: calc(var(--u) * 0.28);
		margin-top: calc(var(--u) * 0.1);
	}
</style>
