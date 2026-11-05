<script lang="ts" module>
	import AssetGrid from './AssetGrid.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// The brand's logo files, from ggml-org/llama.brand (logo/), to download as SVG, WebP or PNG
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Introduction/Assets'
	});
</script>

<Story asChild name="Assets">
	<AssetGrid />
</Story>
