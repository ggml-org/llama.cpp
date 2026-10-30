<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import DialogSettingsChat from '$lib/components/app/dialogs/DialogSettingsChat.svelte';

	// the whole settings dialog; each section has its own story under Settings/Sections
	const { Story } = defineMeta({
		component: DialogSettingsChat,
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Settings/DialogSettingsChat'
	});
</script>

<Story
	args={{ initialSection: 'general', open: true }}
	globals={{ viewport: { isRotated: false, value: 'desktop' } }}
	name="Desktop"
/>

<!-- full screen, with the sections in a scrolling row under the title -->
<Story
	args={{ initialSection: 'general', open: true }}
	globals={{ viewport: { isRotated: false, value: 'mobile2' } }}
	name="Mobile"
/>
