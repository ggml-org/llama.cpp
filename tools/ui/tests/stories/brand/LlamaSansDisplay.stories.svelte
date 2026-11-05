<script lang="ts" module>
	import ShowcaseDisplay from './llama-sans/lib/components/ShowcaseDisplay.svelte';
	import SpecimenFrame from './llama-sans/SpecimenFrame.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Llama Sans at its Display optical size (opsz 40), from the brand specimen: headlines and large type at work
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Typography/Llama Sans/Display'
	});
</script>

<Story asChild name="Display">
	<SpecimenFrame variant="Display">
		<ShowcaseDisplay />
	</SpecimenFrame>
</Story>
