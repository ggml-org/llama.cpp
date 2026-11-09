<script lang="ts" module>
	import ShowcaseFrame from './logo-showcase/ShowcaseFrame.svelte';
	import Lockup from './logo-usage/Lockup.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the two lockups, Llama and llama.cpp: how they are built, typed, sized, the grounds they go on, what they never
	// do, the files, and the lockups in context
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Logo/Lockup'
	});
</script>

<Story asChild name="Lockup">
	<ShowcaseFrame><Lockup /></ShowcaseFrame>
</Story>
