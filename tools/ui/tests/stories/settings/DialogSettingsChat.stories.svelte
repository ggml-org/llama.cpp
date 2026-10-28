<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import DialogSettingsChat from '$lib/components/app/dialogs/DialogSettingsChat.svelte';

	const { Story } = defineMeta({
		component: DialogSettingsChat,
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Settings/DialogSettingsChat'
	});
</script>

<Story args={{ initialSection: 'general', open: true }} name="Default" />
