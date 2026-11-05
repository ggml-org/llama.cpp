<script lang="ts" module>
	import KeyVisuals from './llama-mono/KeyVisuals.svelte';
	import MonoOverview from './llama-mono/MonoOverview.svelte';
	import Hero from './llama-sans/overview/Hero.svelte';
	import OverviewFrame from './llama-sans/overview/OverviewFrame.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Llama Mono, the family's monospace, on its own: the overview (the same as the Typography overview's Mono tab) and
	// its key visuals
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Typography/Llama Mono'
	});
</script>

<Story asChild name="Overview">
	<OverviewFrame>
		<Hero
			face="var(--code)"
			kicker="For code and numbers"
			tagline={['Every character, the same cell.', 'Llama Sans’s letters, set for code.']}
			title={['Llama', 'Mono']}
		/>

		<MonoOverview />
	</OverviewFrame>
</Story>

<Story asChild name="Key visuals">
	<OverviewFrame>
		<div class="posters-only">
			<KeyVisuals />
		</div>
	</OverviewFrame>
</Story>

<style>
	.posters-only {
		padding-top: clamp(32px, 6vw, 72px);
	}
</style>
