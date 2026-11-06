<script lang="ts" module>
	import Swag from './communications/swag/Swag.svelte';
	import { defineMeta } from '@storybook/addon-svelte-csf';

	// Communications' merchandise: the prints the Swag Creator previews, the quantized L among them, flat on each fabric
	const { Story } = defineMeta({
		parameters: {
			controls: { disable: true },
			layout: 'fullscreen'
		},
		title: 'Communications/Swag'
	});
</script>

<Story asChild name="Swag">
	<Swag />
</Story>
