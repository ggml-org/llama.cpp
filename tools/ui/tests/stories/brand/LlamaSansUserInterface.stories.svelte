<script lang="ts" module>
	import UserInterface from './typography/UserInterface.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Llama Sans in the user interface: the design system's own components with llama.cpp's content, at the sizes that
	// put it on its UI optical size (opsz 16)
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Typography/Llama Sans/User Interface'
	});
</script>

<Story asChild name="User Interface">
	<UserInterface />
</Story>
