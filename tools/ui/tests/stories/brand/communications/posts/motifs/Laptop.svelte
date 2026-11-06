<!--
	A laptop, open: a keyboard and trackpad on the base, and on the screen a conversation, the model's reply being
	typed with a cinder cursor. A motif for posts about running models on the machine people already have.
-->
<script lang="ts">
	import './motif.css';
	import { type Box, device, pen, shadow } from './box';
	import { type Point3, project, pts } from './project';
	import Solid from './Solid.svelte';

	const view = device(98, 190);
	const base: Box = { max: [1.25, 0.09, 0.85], min: [-1.25, 0, -0.85] };
	const b = pen(base, view);

	// the lid, hinged at the back of the base and leaning back a little; (u, t) across and up its face
	const lean = 0.22;
	const lid = (u: number, t: number) =>
		project([-1.2 + u, 0.09 + t * Math.cos(lean), 0.82 + t * Math.sin(lean)] as Point3, view);
	const quad = (u: number, t: number, w: number, h: number) =>
		pts([lid(u, t), lid(u + w, t), lid(u + w, t + h), lid(u, t + h)]);

	const keys = Array.from({ length: 4 * 12 }, (_, i) => [
		0.2 + (i % 12) * 0.176,
		0.92 + Math.floor(i / 12) * 0.15
	]);
	const lines = [
		[0.22, 1.12, 0.9],
		[0.22, 0.94, 1.5],
		[0.22, 0.76, 1.15],
		[0.22, 0.58, 0.55]
	];
</script>

<svg aria-hidden="true" class="motif-svg" viewBox="0 0 400 300">
	<polygon class="floor" points={shadow(base, view)} />

	<polygon class="face" points={quad(0, 0, 2.4, 1.6)} />

	<polygon style:fill-opacity="0.05" class="shade" points={quad(0, 0, 2.4, 1.6)} />

	<polygon style:opacity="0.7" class="detail" points={quad(0.1, 0.12, 2.2, 1.38)} />

	<polygon class="soft" points={quad(1.3, 1.24, 0.85, 0.14)} />

	{#each lines as [u, t, w] (t)}
		<polyline style:stroke-width="3" class="detail" points={pts([lid(u, t), lid(u + w, t)])} />
	{/each}

	<polygon class="accent" points={quad(0.79, 0.53, 0.035, 0.12)} />

	<Solid box={base} {view} />

	{#each keys as [u, v] (u + '-' + v)}
		<polygon style:opacity="0.55" class="detail" points={b.rect('top', u, v, 0.13, 0.1)} />
	{/each}

	<polygon style:opacity="0.7" class="detail" points={b.rect('top', 0.88, 0.16, 0.74, 0.56)} />
</svg>
