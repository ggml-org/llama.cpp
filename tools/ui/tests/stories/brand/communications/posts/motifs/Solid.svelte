<!-- A box drawn solid (in a flat version, its front alone): its three visible faces filled with the ground, shaded by which way they face, edged in ink. -->
<script lang="ts">
	import { type Box, faces } from './box';
	import type { View } from './project';

	let { box, flat = false, view }: { box: Box; flat?: boolean; view: View } = $props();
</script>

{#each faces(box, view).filter((f) => !flat || f.name === 'front') as f (f.name)}
	<polygon class="face" points={f.points} />

	<polygon style:fill-opacity={f.shade} class="shade" points={f.points} />
{/each}
