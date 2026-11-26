<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import DialogSettingsChat from '#lib/components/app/dialogs/DialogSettingsChat.svelte';
	import { SETTINGS_SECTIONS } from '#lib/constants/index.ts';

	// one story per settings section, each opening the dialog on that section
	const { Story } = defineMeta({
		component: DialogSettingsChat,
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Llama Chat/Settings/Sections'
	});
</script>

<Story args={{ initialSection: SETTINGS_SECTIONS.GENERAL.slug, open: true }} name="General" />

<Story args={{ initialSection: SETTINGS_SECTIONS.CHAT.slug, open: true }} name="Chat" />

<Story args={{ initialSection: SETTINGS_SECTIONS.MODEL.slug, open: true }} name="Model" />

<Story args={{ initialSection: SETTINGS_SECTIONS.AGENT.slug, open: true }} name="Agent" />

<Story args={{ initialSection: SETTINGS_SECTIONS.VOICE.slug, open: true }} name="Voice" />

<Story args={{ initialSection: SETTINGS_SECTIONS.DATA.slug, open: true }} name="Data" />
