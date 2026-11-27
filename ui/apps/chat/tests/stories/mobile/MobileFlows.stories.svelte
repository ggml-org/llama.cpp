<script lang="ts" module>
	import MobileFlows from './MobileFlows.svx';
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import Editorial from '$llama-ui-stories/editorial/Editorial.svelte';

	// how Llama Chat works on a phone: the sidebar's keys, the three sheets and the pages each pushes
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Llama Chat/Mobile/Flows'
	});
</script>

<Story asChild name="Flows">
	<Editorial>
		<MobileFlows />
	</Editorial>
</Story>
