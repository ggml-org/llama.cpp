<!--
	A GPU: a graphics card standing on its edge, two fans with curved blades on its shroud, heatsink fins along the
	top, the PCIe fingers under it and the outputs on its bracket, its memory in Mono and the activity light in cinder.
	A motif for posts about hardware, offload and speed.
-->
<script lang="ts">
	import './motif.css';
	import { type Box, device, pen, shadow } from './box';
	import Solid from './Solid.svelte';

	const view = device(118, 150);
	const card: Box = { max: [1.3, 0.62, 0.2], min: [-1.3, -0.5, -0.2] };
	const d = pen(card, view);

	const fans = [0.68, 1.9];
	const blade = (cu: number) =>
		Array.from({ length: 9 }, (_, k) =>
			d.path(
				'front',
				Array.from({ length: 8 }, (_, i) => {
					const t = i / 7;
					const r = 0.12 + t * 0.3;
					const a = (k / 9) * Math.PI * 2 + t * 0.9;

					return [cu + r * Math.cos(a), 0.56 + r * Math.sin(a)] as [number, number];
				})
			)
		);
	const fins = Array.from({ length: 30 }, (_, i) => 0.12 + i * 0.08);
	const fingers = Array.from({ length: 16 }, (_, i) => 0.42 + i * 0.06);
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	<polygon class="floor" points={shadow(card, view)} />

	{#each fingers as u (u)}
		<polygon class="hole" points={d.rect('front', u, -0.12, 0.035, 0.12)} />
	{/each}

	<Solid box={card} {view} />

	<polygon class="detail" points={d.rect('front', 0.08, 0.08, 2.44, 0.96)} />

	{#each fans as cu (cu)}
		<polygon class="face" points={d.circle('front', cu, 0.56, 0.44)} />

		<polygon style:stroke-width="2.2" class="detail" points={d.circle('front', cu, 0.56, 0.44)} />

		{#each blade(cu) as b, i (i)}<polyline class="detail" points={b} />{/each}

		<polygon class="solid" points={d.circle('front', cu, 0.56, 0.11)} />
	{/each}

	{#each fins as u (u)}
		<polyline style:opacity="0.6" class="detail" points={d.line('top', u, 0.06, u, 0.34)} />
	{/each}

	{#each [0.2, 0.42, 0.64] as v (v)}
		<polygon class="detail" points={d.rect('right', 0.1, v, 0.2, 0.12)} />
	{/each}

	<polygon class="accent" points={d.circle('front', 2.36, 0.94, 0.045)} />

	<text text-anchor="end" x="372" y="285">24 GB</text>
</svg>
