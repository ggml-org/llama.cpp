<!--
	A stat card: a figure, its label and a line on what it means, on the ground's card. In a row they compare; the one
	that matters most is the accent.
-->
<script lang="ts">
	import Figure from './Figure.svelte';

	interface Props {
		accent?: boolean;
		label: string;
		note?: string;
		/** the figure's size in units; the padding follows it */
		size?: number;
		value: string;
	}

	let { accent = false, label, note, size = 2.4, value }: Props = $props();
</script>

<article style:--size={size} class="card">
	<Figure {accent} {label} {size} {value} />

	{#if note}<p class="t-body note">{note}</p>{/if}
</article>

<style>
	.card {
		display: grid;
		align-content: start;
		gap: calc(var(--u) * 0.6);
		padding: calc(var(--u) * 0.35 * var(--size));
		border: 2px solid var(--p-line);
		border-radius: calc(var(--u) * 0.55);
		background: var(--p-card);
	}
	.note {
		color: var(--p-muted);
	}
</style>
