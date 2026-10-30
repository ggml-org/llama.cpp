<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import DialogSettingsChat from '$lib/components/app/dialogs/DialogSettingsChat.svelte';
	import { SETTINGS_SECTIONS } from '$lib/constants';

	// one story per settings section, each opening the dialog on that section
	const { Story } = defineMeta({
		component: DialogSettingsChat,
		parameters: {
			layout: 'fullscreen'
		},
		title: 'Settings/Sections'
	});
</script>

<Story args={{ initialSection: SETTINGS_SECTIONS.GENERAL.slug, open: true }} name="General" />

<Story args={{ initialSection: SETTINGS_SECTIONS.DISPLAY.slug, open: true }} name="Display" />

<Story args={{ initialSection: SETTINGS_SECTIONS.TOOLS.slug, open: true }} name="Tools" />

<Story args={{ initialSection: SETTINGS_SECTIONS.AGENTIC.slug, open: true }} name="Agentic" />

<Story
	args={{ initialSection: SETTINGS_SECTIONS.IMPORT_EXPORT.slug, open: true }}
	name="Import/Export"
/>

<Story
	args={{ initialSection: SETTINGS_SECTIONS.SAMPLING_PENALTIES.slug, open: true }}
	name="Sampling & Penalties"
/>

<Story args={{ initialSection: SETTINGS_SECTIONS.DEVELOPER.slug, open: true }} name="Developer" />
