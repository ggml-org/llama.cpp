<!-- One rule that shapes the icon, drawn over a faint icon: the two blades, the 60° edges, the shared diagonal,
     the split, the steps of 120 and the soft corners. The caption is the children. -->
<script lang="ts" module>
	export const RULES = ['blades', 'angle', 'diagonal', 'split', 'steps', 'corners'] as const;
	export type RuleName = (typeof RULES)[number];
</script>

<script lang="ts">
	import { lower, STEPS } from '../blueprint';
	import { corners, edges, PATHS, SIZE, SPLIT } from '../logo';
	import Figure from './Figure.svelte';
	import type { Snippet } from 'svelte';

	let { children, rule, title }: { rule: RuleName; title?: string; children?: Snippet } = $props();

	const titles: Record<RuleName, string> = {
		angle: '60° everywhere',
		blades: 'Two blades',
		corners: 'Soft outer corners',
		diagonal: 'One diagonal',
		split: 'The split',
		steps: 'Steps of 120'
	};
</script>

{#snippet icon(fill = 'var(--l-faint)')}
	{#each PATHS as d, i (i)}<path {d} {fill} />{/each}
{/snippet}

<Figure {children} title={title ?? titles[rule]}>
	{#snippet drawing()}
		{#if rule === 'blades'}
			<path d={PATHS[1]} fill="var(--l-shape)" />

			<path d={PATHS[0]} fill="var(--l-shape-2)" />

			<text class="note on-shape" x="150" y="250">upper blade</text>

			<text class="note on-shape" x="300" y="520">lower blade</text>
		{:else if rule === 'angle'}
			{@render icon()}

			{#each [[0, SPLIT], [240, SPLIT], [120, SIZE]] as [x, y], i (i)}
				<path class="arc" d="M{x + 80} {y} A80 80 0 0 0 {x + 40} {y - 69.3}" />

				<text class="note" x={x + 92} y={y - 26}>60°</text>
			{/each}

			{#each [edges.upperLeft, edges.shared, edges.lowerRight] as [a, b], i (i)}
				<line class="hi" x1={a[0]} x2={b[0]} y1={a[1]} y2={b[1]} />
			{/each}
		{:else if rule === 'diagonal'}
			{@render icon()}

			<line class="hi" x1="466.154" x2="120" y1="0" y2="600" />

			<circle class="joint" cx="240" cy={SPLIT} r="12" />

			<text class="note" x="270" y={SPLIT + 10}>240, {SPLIT}</text>
		{:else if rule === 'split'}
			{@render icon()}

			<line class="hi" x1="-40" x2="640" y1={SPLIT} y2={SPLIT} />

			<line class="dim" x1="-30" x2="-30" y1="0" y2={SPLIT} />

			<line class="dim" x1="-30" x2="-30" y1={SPLIT} y2={SIZE} />

			<text class="note" x="-20" y={SPLIT / 2}>{SPLIT}</text>

			<text class="note" x="-20" y={SPLIT + lower / 2 + 10}>{lower}</text>
		{:else if rule === 'steps'}
			{@render icon()}

			{#each STEPS as x (x)}
				<line class="hi" x1={x} x2={x} y1="-40" y2="616" />

				<text class="note" text-anchor="middle" {x} y="652">{x}</text>
			{/each}
		{:else}
			{@render icon()}

			{#each corners as c, i (i)}
				<circle class="hi" cx={c.centre[0]} cy={c.centre[1]} r={c.r} />

				<line
					class="dim"
					x1={c.sharp[0]}
					x2={c.tangentFlat[0]}
					y1={c.sharp[1]}
					y2={c.tangentFlat[1]}
				/>

				<circle class="joint" cx={c.sharp[0]} cy={c.sharp[1]} r="8" />
			{/each}
		{/if}
	{/snippet}
</Figure>
