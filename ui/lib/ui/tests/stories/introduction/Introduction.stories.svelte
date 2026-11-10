<script lang="ts" module>
	import Editorial from '../philosophy/Editorial.svelte';
	import Introduction from './Introduction.svx';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// the Storybook's front page: what the package is and how it keeps in step with ui/apps/chat
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
