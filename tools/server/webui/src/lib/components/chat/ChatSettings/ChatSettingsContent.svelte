<script lang="ts">
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import ChatSettingsSection from './ChatSettingsSection.svelte';
	import ChatSettingsField from './ChatSettingsField.svelte';
	import type { SettingSection, SettingsConfig } from '$lib/types/settings';

	interface Props {
		currentSection: SettingSection;
		config: SettingsConfig;
		defaultConfig: SettingsConfig;
		onConfigChange: (key: string, value: any) => void;
	}

	let { currentSection, config, defaultConfig, onConfigChange }: Props = $props();

	function handleFieldChange(key: string, value: any) {
		onConfigChange(key, value);
	}
</script>

<div class="flex flex-1 flex-col">
	<ScrollArea class="flex-1 p-6">
		<div class="space-y-6">
			<ChatSettingsSection title={currentSection.title} icon={currentSection.icon}>
				<!-- Section Fields -->
				{#each currentSection.fields as field}
					<ChatSettingsField
						{field}
						value={config[field.key]}
						defaultValue={defaultConfig[field.key]}
						onChange={(value) => handleFieldChange(field.key, value)}
					/>
				{/each}
			</ChatSettingsSection>

			<!-- Footer Note -->
			<div class="mt-8 border-t pt-6">
				<p class="text-muted-foreground text-xs">
					Settings are saved in browser's localStorage
				</p>
			</div>
		</div>
	</ScrollArea>
</div>
