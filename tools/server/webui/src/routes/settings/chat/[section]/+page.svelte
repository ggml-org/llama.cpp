<script lang="ts">
	import { page } from '$app/state';
	import { ChatSettingsFields, ChatSettingsToolsTab } from '$lib/components/app';
	import { getChatSettingsConfigContext } from '$lib/contexts';
	import { SETTINGS_SECTION_TITLES, SETTINGS_CHAT_SECTIONS } from '$lib/constants';

	const { localConfig, handleConfigChange, handleThemeChange } = getChatSettingsConfigContext();

	let currentSection = $derived(
		SETTINGS_CHAT_SECTIONS.find((s) => s.slug === page.params.section) ?? SETTINGS_CHAT_SECTIONS[0]
	);
</script>

{#if currentSection.title === SETTINGS_SECTION_TITLES.TOOLS}
	<ChatSettingsToolsTab />
{:else if currentSection.fields}
	<div class="space-y-6">
		<ChatSettingsFields
			fields={currentSection.fields}
			{localConfig}
			onConfigChange={handleConfigChange}
			onThemeChange={handleThemeChange}
		/>
	</div>
{/if}
