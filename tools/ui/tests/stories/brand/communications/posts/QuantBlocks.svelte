<!--
	Quantised blocks: a model's weights as a grid of squares in a few levels of the ink, as a 2-bit or 4-bit quant
	keeps them, and one in cinder. A motif for posts about quants, sizes and fitting a model in memory.
-->
<script lang="ts">
	interface Props {
		columns?: number;
		/** the levels a block can take: 4 for a 2-bit quant, 16 for a 4-bit one */
		levels?: number;
		rows?: number;
		seed?: number;
	}

	let { columns = 8, levels = 4, rows = 8, seed = 7 }: Props = $props();

	const blocks = $derived.by(() => {
		let x = seed;

		return Array.from({ length: columns * rows }, () => {
			x = (x * 1103515245 + 12345) % 2147483648;

			return Math.floor((x / 2147483648) * levels) / (levels - 1);
		});
	});
	const accent = $derived(Math.floor(blocks.length * 0.62));
</script>

<div style:--c={columns} aria-hidden="true" class="blocks">
	{#each blocks as b, i (i)}
		<i class:accent={i === accent} style:opacity={i === accent ? 1 : 0.12 + b * 0.88}></i>
	{/each}
</div>

<style>
	.blocks {
		display: grid;
		grid-template-columns: repeat(var(--c), 1fr);
		gap: calc(var(--u) * 0.14);
		width: 100%;
	}
	.blocks i {
		aspect-ratio: 1;
		border-radius: calc(var(--u) * 0.1);
		background: var(--p-ink);
	}
	.blocks .accent {
		background: var(--art-cinder);
	}
</style>
