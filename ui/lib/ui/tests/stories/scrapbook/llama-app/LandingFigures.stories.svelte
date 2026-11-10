<script lang="ts" module>
	import Page from './figures/Page.svelte';
	import Frames from './Frames.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// llama.app's landing figures redrawn in the design system's figure language (Philosophy's figures, Architecture's
	// diagrams), in two directions: A as Philosophy draws, neutral with cinder as the one accent; B as Architecture ›
	// Stack draws, apps in cinder, Llama in the skies and the engine and model files in grey. Each at a desktop width and
	// a phone's
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Scrapbook/UI/llama.app/Landing figures'
	});
</script>

<!-- A: Philosophy's figures, neutral, the cinder dot as the one accent -->
<Story asChild name="A Figure">
	<Frames><Page /></Frames>
</Story>

<!-- B: Architecture's roles, apps in cinder, Llama in the skies, the engine and files in grey -->
<Story asChild name="B Architecture">
	<Frames><Page arch /></Frames>
</Story>
