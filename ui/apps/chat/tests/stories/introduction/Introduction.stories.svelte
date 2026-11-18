<script lang="ts" module>
	import Introduction from './Introduction.svx';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import Editorial from '$llama-ui-stories/editorial/Editorial.svelte';

	// the Storybook's front page: what it is for, how we think, how the products fit together and how it is laid out
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Introduction'
	});
</script>

<Story asChild name="Introduction">
	<Editorial>
		<Introduction />
	</Editorial>
</Story>
