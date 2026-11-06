<!-- A box drawn solid: its three visible faces filled with the ground, shaded by which way they face, edged in ink. -->
<script lang="ts">
	import { type Box, faces } from './box';
	import type { View } from './project';

	let { box, view }: { box: Box; view: View } = $props();
</script>

{#each faces(box, view) as f (f.name)}
	<polygon class="face" points={f.points} />

	<polygon style:fill-opacity={f.shade} class="shade" points={f.points} />
{/each}
