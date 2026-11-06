<!--
	A figure: a big number with a Mono label under it. In cinder when it is the piece's accent; otherwise in ink.
-->
<script lang="ts">
	interface Props {
		accent?: boolean;
		label: string;
		/** the figure's size in units */
		size?: number;
		value: string;
	}

	let { accent = false, label, size = 5, value }: Props = $props();
</script>

<div style:--s="calc(var(--u) * {size})" class="figure">
	<p class:accent class="t-figure value">{value}</p>

	<p class="t-label">{label}</p>
</div>

<style>
	.figure {
		display: grid;
		gap: calc(var(--s) * 0.14);
		justify-items: start;
	}
	.value {
		font-size: var(--s);
	}
	.accent {
		color: var(--art-cinder);
	}
</style>
