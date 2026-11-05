<script lang="ts" module>
	import ShowcaseUI from './llama-sans/lib/components/ShowcaseUI.svelte';
	import { specimen } from './llama-sans/lib/specimen';
	import SpecimenFrame from './llama-sans/SpecimenFrame.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Llama Sans at its UI optical size (opsz 16), from the brand specimen: in controls and small interface text
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Typography/Llama Sans/User Interface'
	});

	const { glyphCount } = specimen;
</script>

<Story asChild name="User Interface">
	<SpecimenFrame variant="UI">
		<ShowcaseUI {glyphCount} guides />
	</SpecimenFrame>
</Story>
