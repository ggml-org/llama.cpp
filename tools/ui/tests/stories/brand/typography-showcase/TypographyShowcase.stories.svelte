<script lang="ts" module>
	import SpecimenFrame from './SpecimenFrame.svelte';
	import TypographyShowcase from './TypographyShowcase.svx';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the Llama Sans specimen, as on the brand site: the masthead, live controls, the versions at work, the weights,
	// the glyph grid and the design notes
	const { Story } = defineMeta({
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Typography/Showcase'
	});
</script>

<Story asChild name="Page">
	<SpecimenFrame>
		<TypographyShowcase />
	</SpecimenFrame>
</Story>
