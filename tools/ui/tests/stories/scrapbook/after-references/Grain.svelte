<!--
	Film grain over an artwork field: fine noise, blended so it lightens the dark and darkens the light, as the
	references' soft fields have. It fills its positioned parent and takes no pointer.
-->
<script lang="ts">
	interface Props {
		/** how strong the grain is, 0 to 1 */
		amount?: number;
	}

	let { amount = 0.35 }: Props = $props();

	const id = `grain-${Math.random().toString(36).slice(2, 8)}`;
</script>

<svg style:opacity={amount} aria-hidden="true" class="grain">
	<filter {id}>
		<feTurbulence baseFrequency="0.85" numOctaves="2" stitchTiles="stitch" type="fractalNoise" />

		<feColorMatrix type="saturate" values="0" />
	</filter>

	<rect filter="url(#{id})" height="100%" width="100%" />
</svg>

<style>
	.grain {
		position: absolute;
		inset: 0;
		width: 100%;
		height: 100%;
		mix-blend-mode: overlay;
		pointer-events: none;
	}
</style>
