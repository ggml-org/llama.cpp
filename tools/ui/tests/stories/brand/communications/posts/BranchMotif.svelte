<!--
	A token tree: a model choosing between options, the chosen path in cinder with its probability in Mono, the
	rest in the post's ink. A motif for posts about decisions, sampling and probabilities.
-->
<script lang="ts">
	interface Props {
		/** the chosen option's probability and the runner-up's */
		values?: [string, string];
	}

	let { values = ['0.91', '0.06'] }: Props = $props();

	// the root, two forks, and the leaves (x, y); the first leaf of the first fork is the one chosen
	const root = [40, 250];
	const forks = [
		[220, 130],
		[220, 365]
	];
	const leaves = [
		[
			[400, 40],
			[400, 195]
		],
		[
			[400, 285],
			[400, 365],
			[400, 445]
		]
	];
	const curve = ([x1, y1]: number[], [x2, y2]: number[]) => {
		const mx = (x1 + x2) / 2;

		return `M${x1} ${y1} C${mx} ${y1} ${mx} ${y2} ${x2} ${y2}`;
	};
</script>

<svg aria-hidden="true" class="motif" viewBox="0 0 560 490">
	{#each forks as fork, f (f)}
		<path class:chosen={f === 0} d={curve(root, fork)} />

		{#each leaves[f] as leaf, l (l)}
			<path class:chosen={f === 0 && l === 0} d={curve(fork, leaf)} />
		{/each}
	{/each}

	{#each leaves as group, f (f)}
		{#each group as [x, y], l (l)}<circle
				class:chosen={f === 0 && l === 0}
				cx={x}
				cy={y}
				r="30"
			/>{/each}
	{/each}

	{#each forks as [x, y], f (f)}<circle class:chosen={f === 0} cx={x} cy={y} r="30" />{/each}

	<circle class="chosen" cx={root[0]} cy={root[1]} r="34" />

	<text class="chosen" x="452" y="40">{values[0]}</text>

	<text x="452" y="195">{values[1]}</text>
</svg>

<style>
	.motif {
		display: block;
		width: 100%;
		height: 100%;
		overflow: visible;
	}
	path {
		fill: none;
		stroke: var(--p-ink);
		stroke-width: 26;
		stroke-linecap: round;
	}
	circle {
		fill: var(--p-ink);
	}
	text {
		fill: var(--p-muted);
		font: 500 36px var(--mono);
		dominant-baseline: central;
	}
	.chosen {
		fill: var(--art-cinder);
		stroke: var(--art-cinder);
	}
	path.chosen {
		fill: none;
	}
	text.chosen {
		stroke: none;
	}
</style>
