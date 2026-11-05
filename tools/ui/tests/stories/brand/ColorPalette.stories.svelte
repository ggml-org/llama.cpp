<script lang="ts" module>
	import BrandSwatches from './colors/BrandSwatches.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the colours and gradients the brand's designs use, as swatches; the UI's tokens are in UI/Foundations/Colors
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Colors/Palette'
	});
</script>

<Story asChild name="Palette">
	<BrandSwatches />
</Story>
