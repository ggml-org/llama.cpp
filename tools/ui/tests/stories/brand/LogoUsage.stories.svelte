<script lang="ts" module>
	import ShowcaseFrame from './logo-showcase/ShowcaseFrame.svelte';
	import IconUsage from './logo-usage/IconUsage.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the icon in use, one case per story: the font's glyphs, buttons, a download badge, running text, the web app's icons
	// with the social avatars, with their proportions (the lockups are Brand/Logo/Lockup, the splash screens
	// UI/Foundations/Brand)
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Brand/Logo/Usage'
	});
</script>

<Story asChild name="Glyphs">
	<ShowcaseFrame><IconUsage section="glyphs" /></ShowcaseFrame>
</Story>

<Story asChild name="Buttons">
	<ShowcaseFrame><IconUsage section="buttons" /></ShowcaseFrame>
</Story>

<Story asChild name="Download badge">
	<ShowcaseFrame><IconUsage section="badge" /></ShowcaseFrame>
</Story>

<Story asChild name="Running text">
	<ShowcaseFrame><IconUsage section="text" /></ShowcaseFrame>
</Story>

<Story asChild name="App icons and avatars">
	<ShowcaseFrame><IconUsage section="icons" /></ShowcaseFrame>
</Story>
