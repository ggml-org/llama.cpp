<script lang="ts" module>
	import ShowcaseProse from './llama-sans/lib/components/ShowcaseProse.svelte';
	import SpecimenFrame from './llama-sans/SpecimenFrame.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Llama Sans at its UI optical size (opsz 16), from the brand specimen: in running text, at line-height 1.5
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Typography/Llama Sans/Running text'
	});
</script>

<Story asChild name="Running text">
	<SpecimenFrame variant="UI">
		<ShowcaseProse />
	</SpecimenFrame>
</Story>
