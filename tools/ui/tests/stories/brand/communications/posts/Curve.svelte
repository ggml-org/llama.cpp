<!--
	A curve: a figure over time or size, such as tokens per second, as a line in ink with a faint fill and its last
	point in cinder, labelled in Mono. A motif for posts about speed and benchmarks; the values are illustrative.
-->
<script lang="ts">
	interface Props {
		label?: string;
		/** the points, left to right, 0 to 1 */
		points?: number[];
	}

	let {
		label = '142 tok/s',
		points = [0.08, 0.14, 0.13, 0.27, 0.35, 0.33, 0.52, 0.61, 0.7, 0.86]
	}: Props = $props();

	const w = 600;
	const h = 320;
	const xy = $derived(
		points.map((p, i) => [(i / (points.length - 1)) * w, h - p * (h - 40)] as const)
	);
	const line = $derived(xy.map(([x, y], i) => `${i ? 'L' : 'M'}${x} ${y}`).join(' '));
	const last = $derived(xy[xy.length - 1]);
</script>

<svg
	aria-hidden="true"
	class="curve"
	preserveAspectRatio="xMidYMid meet"
	viewBox="-20 -40 {w + 40} {h + 60}"
>
	{#each [0.25, 0.5, 0.75, 1] as r (r)}<line
			class="rule"
			x1="0"
			x2={w}
			y1={h * r}
			y2={h * r}
		/>{/each}

	<path class="fill" d="{line} L{w} {h} L0 {h} Z" />

	<path class="line" d={line} />

	<circle class="end" cx={last[0]} cy={last[1]} r="11" />

	<text class="value" text-anchor="end" x={last[0] - 22} y={last[1] - 18}>{label}</text>
</svg>

<style>
	.curve {
		display: block;
		width: 100%;
		height: 100%;
		overflow: visible;
	}
	.rule {
		stroke: var(--p-line);
		stroke-width: 2;
	}
	.fill {
		fill: var(--p-inset);
	}
	.line {
		fill: none;
		stroke: var(--p-ink);
		stroke-width: 5;
		stroke-linejoin: round;
	}
	.end {
		fill: var(--art-cinder);
	}
	.value {
		fill: var(--p-ink);
		font: 500 30px var(--mono);
	}
</style>
