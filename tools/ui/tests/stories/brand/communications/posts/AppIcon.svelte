<!--
	The app icon, for when Llama is a thing in a piece (a node in a diagram, the app in a row of apps): the macOS
	icon's tile as it ships (app/apple/macos-1024x1024), its corners at 22.5% of its side and the icon at 67% of it,
	lifted and nudged left to its optical centre, in the ground's solid ink with the icon in the ground's colour. Never
	a square drawn round the icon. `size` is its side in units; give it a unit of clear space all round.
-->
<script lang="ts">
	import { PATHS } from '../../logo-showcase/lib/logo';

	interface Props {
		/** the tile's side in units */
		size?: number;
	}

	let { size = 2.4 }: Props = $props();
</script>

<svg
	style:--s="calc(var(--u) * {size})"
	aria-label="Llama"
	class="app-icon"
	role="img"
	viewBox="0 0 824 824"
>
	<rect fill="var(--p-solid)" height="824" rx="185.4" width="824" />

	<g fill="var(--p-solid-ink)" transform="translate(129.599 121.293) scale(0.92288)">
		{#each PATHS as d (d)}<path {d} />{/each}
	</g>
</svg>

<style>
	.app-icon {
		display: block;
		flex: none;
		width: var(--s);
		height: var(--s);
	}
</style>
