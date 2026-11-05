<script lang="ts" module>
	import KeyVisuals from './llama-mono/KeyVisuals.svelte';
	import MonoOverview from './llama-mono/MonoOverview.svelte';
	import OverviewFrame from './llama-sans/overview/OverviewFrame.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Llama Mono, the family's monospace: the overview (code, the look-alikes kept apart, weights, characters) and
	// its key visuals
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Typography/Llama Mono'
	});
</script>

<Story asChild name="Overview">
	<OverviewFrame>
		<div class="mono-page">
			<MonoOverview />
		</div>
	</OverviewFrame>
</Story>

<Story asChild name="Key visuals">
	<OverviewFrame>
		<div class="mono-page posters-only">
			<KeyVisuals />
		</div>
	</OverviewFrame>
</Story>

<style>
	/* Llama Mono is the code face on these pages; the rest stays on the overview's fonts */
	.mono-page {
		--code: 'Llama Mono', ui-monospace, monospace;
		/* Geist Mono's coding ligatures (-> => -- !==) are one cell wide, so they would pull the columns out of line;
		   off until the font draws them across their cells (the tt join is calt, and stays) */
		display: contents;
	}
	.mono-page :global(*) {
		font-feature-settings: 'liga' 0;
	}
	.posters-only {
		display: block;
		padding-top: clamp(32px, 6vw, 72px);
	}
</style>
