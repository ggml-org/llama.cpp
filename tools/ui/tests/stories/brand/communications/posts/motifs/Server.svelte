<!--
	A server: four rack units between two rails, each with a row of drive bays, a vent and its status lights, one light
	in cinder for the one serving. A motif for posts about llama-server, sharing and the largest models.
-->
<script lang="ts">
	import './motif.css';
	import { type Box, device, pen, shadow } from './box';
	import Solid from './Solid.svelte';

	const view = device(100, 230);
	const units: Box[] = [0, 1, 2, 3].map((i) => ({
		max: [1.25, 0.06 + i * 0.44 + 0.38, 0.85],
		min: [-1.25, 0.06 + i * 0.44, -0.85]
	}));
	const rails: Box[] = [-1.36, 1.25].map((x) => ({
		max: [x + 0.11, 1.86, -0.78],
		min: [x, 0, -0.9]
	}));
	const pens = units.map((u) => pen(u, view));
	const bays = [0.14, 0.42, 0.7, 0.98, 1.26];
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	<polygon class="floor" points={shadow(units[0], view)} />

	{#each units as unit, i (i)}
		{@const d = pens[i]}
		<Solid box={unit} {view} />

		{#each bays as u (u)}
			<polygon class="detail" points={d.rect('front', u, 0.08, 0.24, 0.22)} />

			<polygon class="hole" points={d.circle('front', u + 0.2, 0.13, 0.012, 8)} />
		{/each}

		{#each [1.62, 1.68, 1.74, 1.8, 1.86, 1.92, 1.98] as u (u)}
			<polyline style:opacity="0.6" class="detail" points={d.line('front', u, 0.09, u, 0.29)} />
		{/each}

		<polygon class="hole" points={d.circle('front', 2.16, 0.19, 0.028, 12)} />

		<polygon
			class={i === 2 ? 'accent' : 'hole'}
			points={d.circle('front', 2.28, 0.19, 0.028, 12)}
		/>
	{/each}

	{#each rails as rail, i (i)}
		<Solid box={rail} {view} />
	{/each}
</svg>
