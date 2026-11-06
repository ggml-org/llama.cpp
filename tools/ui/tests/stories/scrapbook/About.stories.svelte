<script lang="ts" module>
	import Editorial from '../philosophy/Editorial.svelte';
	import About from './About.svx';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the Scrapbook's front page: what goes in it and how a proposal is filed
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Scrapbook/About'
	});
</script>

<Story asChild name="About">
	<Editorial>
		<About />
	</Editorial>
</Story>
