<script lang="ts" module>
	import LogoShowcase from './LogoShowcase.svx';
	import ShowcaseFrame from './ShowcaseFrame.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the icon's construction, as on the brand site: the origin animation into the blueprint, then its rules
	const { Story } = defineMeta({
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Brand/Logo/Overview'
	});
</script>

<Story asChild name="Overview">
	<ShowcaseFrame>
		<LogoShowcase />
	</ShowcaseFrame>
</Story>
