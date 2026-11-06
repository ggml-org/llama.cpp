<script lang="ts" module>
	import ShowcaseFrame from '../../brand/logo-showcase/ShowcaseFrame.svelte';
	import IconUsage from '../../brand/logo-usage/IconUsage.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the brand where the web app shows it: the splash screens, the icon on the page colour while the app opens
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'UI/Foundations/Brand'
	});
</script>

<Story asChild name="Splash screens">
	<ShowcaseFrame><IconUsage section="splash" /></ShowcaseFrame>
</Story>
