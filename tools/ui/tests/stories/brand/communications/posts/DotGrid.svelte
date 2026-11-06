<!--
	A dot grid on the unit: the ground's lines as dots, one unit apart, and one dot in cinder
	unless the accent is elsewhere. A quiet field behind a
	word or a figure; it shows the grid the piece is built on.
-->
<script lang="ts">
	interface Props {
		/** the cinder dot, in units from the top left; none when the piece's accent is elsewhere */
		at?: [number, number] | null;
	}

	let { at = [3, 2] }: Props = $props();
</script>

<div aria-hidden="true" class="dots">
	{#if at}<i style:left="calc(var(--u) * {at[0]})" style:top="calc(var(--u) * {at[1]})" class="dot"
		></i>{/if}
</div>

<style>
	.dots {
		position: relative;
		width: 100%;
		height: 100%;
		min-height: calc(var(--u) * 4);
		background: radial-gradient(circle, var(--p-muted) calc(var(--u) * 0.06), transparent 0) 0 0 /
			var(--u) var(--u);
		opacity: 0.9;
	}
	.dot {
		position: absolute;
		width: calc(var(--u) * 0.36);
		height: calc(var(--u) * 0.36);
		border-radius: 50%;
		background: var(--art-cinder);
		translate: calc(var(--u) * 0.32) calc(var(--u) * 0.32);
	}
</style>
