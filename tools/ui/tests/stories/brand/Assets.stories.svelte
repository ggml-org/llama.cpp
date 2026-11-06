<script lang="ts" module>
	import AssetGrid from './assets/AssetGrid.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// The brand's files, from ggml-org/llama.brand (logo/), to download as SVG, WebP or PNG: the logo files and the
	// social profile pictures and banners
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Logo/Assets'
	});
</script>

<Story asChild name="Logo">
	<AssetGrid section="logo" />
</Story>

<Story asChild name="Social profiles">
	<AssetGrid section="social" />
</Story>
