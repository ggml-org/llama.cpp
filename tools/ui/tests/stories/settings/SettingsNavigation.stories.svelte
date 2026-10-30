<script lang="ts" module>
	import { defineMeta } from '@storybook/addon-svelte-csf';
	import {
		SettingsChatDesktopSidebar,
		SettingsChatMobileHeader
	} from '$lib/components/app/settings';
	import { SETTINGS_CHAT_SECTIONS } from '$lib/constants';
	import type { SettingsSection, SettingsSectionTitle } from '$lib/types';

	// the settings section switcher: a sidebar list from md up, a scrolling row below it
	const { Story } = defineMeta({
		title: 'Settings/Navigation'
	});
</script>

<script lang="ts">
	let activeTitle: SettingsSectionTitle = $state(SETTINGS_CHAT_SECTIONS[0].title);

	const isActive = (section: SettingsSection) => section.title === activeTitle;
	const onSectionChange = (title: SettingsSectionTitle) => (activeTitle = title);
</script>

<Story asChild globals={{ viewport: { isRotated: false, value: 'desktop' } }} name="Desktop">
	<SettingsChatDesktopSidebar {isActive} {onSectionChange} sections={SETTINGS_CHAT_SECTIONS} />
</Story>

<Story
	asChild
	globals={{ viewport: { isRotated: false, value: 'mobile2' } }}
	name="Mobile"
	parameters={{ layout: 'fullscreen' }}
>
	<SettingsChatMobileHeader {isActive} {onSectionChange} sections={SETTINGS_CHAT_SECTIONS} />
</Story>
