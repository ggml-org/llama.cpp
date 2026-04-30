<script lang="ts">
	import {
		ChatSettingsFooter,
		ChatSettingsFields,
		ChatSettingsToolsTab
	} from '$lib/components/app';
	import {
		SettingsChatDesktopSidebar,
		SettingsChatMobileHeader
	} from '$lib/components/app/settings';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import {
		NUMERIC_FIELDS,
		POSITIVE_INTEGER_FIELDS,
		SETTINGS_CHAT_SECTIONS,
		SETTINGS_SECTION_TITLES,
		type SettingsSection
	} from '$lib/constants';
	import { setMode } from 'mode-watcher';
	import { ColorMode } from '$lib/enums/ui';
	import { fade } from 'svelte/transition';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import { setChatSettingsConfigContext } from '$lib/contexts';
	import { settingsReferrer } from '$lib/stores/settings-referrer.svelte';

	interface Props {
		initialSection?: string;
		getSectionHref?: (section: SettingsSection) => string;
	}

	let { initialSection, getSectionHref }: Props = $props();

	let activeSlug = $derived(
		initialSection ?? (page.params as Record<string, string | undefined>).section ?? 'general'
	);
	let currentSection = $derived(
		SETTINGS_CHAT_SECTIONS.find((section) => section.slug === activeSlug) ||
			SETTINGS_CHAT_SECTIONS[0]
	);
	let localConfig: SettingsConfigType = $state({ ...config() });

	let mobileHeader: { updateCarousel: () => void } | undefined;

	function handleThemeChange(newTheme: string) {
		localConfig.theme = newTheme;
		setMode(newTheme as ColorMode);
	}

	function handleConfigChange(key: string, value: string | boolean) {
		localConfig[key] = value;
	}

	function handleReset() {
		localConfig = { ...config() };
		setMode(localConfig.theme as ColorMode);
		mobileHeader?.updateCarousel();
	}

	function handleSave() {
		if (localConfig.custom && typeof localConfig.custom === 'string' && localConfig.custom.trim()) {
			try {
				JSON.parse(localConfig.custom);
			} catch (error) {
				alert('Invalid JSON in custom parameters. Please check the format and try again.');
				console.error(error);
				return;
			}
		}

		const processedConfig = { ...localConfig };

		for (const field of NUMERIC_FIELDS) {
			if (processedConfig[field] !== undefined && processedConfig[field] !== '') {
				const numValue = Number(processedConfig[field]);
				if (!isNaN(numValue)) {
					if ((POSITIVE_INTEGER_FIELDS as readonly string[]).includes(field)) {
						processedConfig[field] = Math.max(1, Math.round(numValue));
					} else {
						processedConfig[field] = numValue;
					}
				} else {
					alert(`Invalid numeric value for ${field}. Please enter a valid number.`);
					return;
				}
			}
		}

		settingsStore.updateMultipleConfig(processedConfig);
		goto(settingsReferrer.url);
	}

	export function reset() {
		localConfig = { ...config() };
	}

	setChatSettingsConfigContext({
		get localConfig() {
			return localConfig;
		},
		handleConfigChange,
		handleThemeChange
	});
</script>

<div
	class="mx-auto flex h-full max-h-[100dvh] w-full flex-col overflow-y-auto md:pl-8"
	in:fade={{ duration: 150 }}
>
	<div class="flex flex-1 flex-col gap-4 md:flex-row">
		<SettingsChatDesktopSidebar
			sections={SETTINGS_CHAT_SECTIONS}
			isActive={(section: SettingsSection) => section.slug === activeSlug}
			getHref={getSectionHref ?? ((section: SettingsSection) => `#/settings/chat/${section.slug}`)}
		/>

		<SettingsChatMobileHeader
			sections={SETTINGS_CHAT_SECTIONS}
			isActive={(section: SettingsSection) => section.slug === activeSlug}
			getHref={getSectionHref ?? ((section: SettingsSection) => `#/settings/chat/${section.slug}`)}
			bind:this={mobileHeader}
		/>

		<div class="mx-auto max-w-3xl flex-1">
			<div class="space-y-6 p-4 md:p-6 md:pt-28">
				<div class="grid">
					<div class="mb-6 flex items-center gap-2 border-b border-border/30 pb-6 md:flex">
						<currentSection.icon class="h-5 w-5" />
						<h3 class="text-lg font-semibold">{currentSection.title}</h3>
					</div>

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
				</div>

				<div class="mt-8 border-t border-border/30 pt-6">
					<p class="text-xs text-muted-foreground">Settings are saved in browser's localStorage</p>
				</div>
			</div>

			<ChatSettingsFooter onReset={handleReset} onSave={handleSave} />
		</div>
	</div>
</div>
