<script lang="ts" module>
	import ShowcaseProse from './llama-sans/lib/components/ShowcaseProse.svelte';
	import ShowcaseUI from './llama-sans/lib/components/ShowcaseUI.svelte';
	import { specimen } from './llama-sans/lib/specimen';
	import SpecimenFrame from './llama-sans/SpecimenFrame.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Llama Sans at its UI optical size (opsz 16), from the brand specimen: in controls and in running text. The weights
	// are in the Typography overview and the tester is Brand/Typography/Tester
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Typography/Llama Sans/UI'
	});

	const { glyphCount } = specimen;
</script>

<Story asChild name="Interface">
	<SpecimenFrame variant="UI">
		<ShowcaseUI {glyphCount} guides />
	</SpecimenFrame>
</Story>

<Story asChild name="Running text">
	<SpecimenFrame variant="UI">
		<ShowcaseProse />
	</SpecimenFrame>
</Story>
