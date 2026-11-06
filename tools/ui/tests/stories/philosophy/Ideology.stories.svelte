<script lang="ts" module>
	import Editorial from './Editorial.svelte';
	import Ideology from './Ideology.svx';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// What we stand for: why llama.cpp, Llama UI and the apps exist
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Philosophy/Ideology'
	});
</script>

<Story asChild name="Ideology">
	<Editorial>
		<Ideology />
	</Editorial>
</Story>
