<script lang="ts">
	import { RefreshCw } from '@lucide/svelte';
	import { goto } from '$app/navigation';
	import { page } from '$app/state';
	import {
		SettingsChatDesktopSidebar,
		SettingsChatFields,
		SettingsChatImportExportTab,
		SettingsChatMobileHeader,
		SettingsChatToolsTab,
		SettingsFooter
	} from '$lib/components/app/settings';
	import { Button } from '$lib/components/ui/button';
	import {
		NUMERIC_FIELDS,
		POSITIVE_INTEGER_FIELDS,
		SETTINGS_CHAT_SECTIONS,
		SETTINGS_SECTION_TITLES
	} from '$lib/constants';
	import { setChatSettingsConfigContext } from '$lib/contexts';
	import { ColorMode } from '$lib/enums/ui.enums';
	import { RouterService } from '$lib/services/router.service';
	import { modelsStore, serverStore, settingsReferrer, settingsStore } from '$lib/stores';
	import type { SettingsSection } from '$lib/types';
	import { setMode } from 'mode-watcher';
	import { fade } from 'svelte/transition';
	interface Props {
		initialSection?: string;
		getSectionHref?: (section: SettingsSection) => string;
	}

	let { getSectionHref, initialSection }: Props = $props();

	let activeSlug = $derived(
		initialSection ?? (page.params as Record<string, string | undefined>).section ?? 'general'
	);

	let currentSection = $derived(
		SETTINGS_CHAT_SECTIONS.find((section) => section.slug === activeSlug) ||
			SETTINGS_CHAT_SECTIONS[0]
	);

	let localConfig: SettingsConfigType = $state({ ...settingsStore.config });

	let mobileHeader: { updateCarousel: () => void } | undefined;

	let fetchInitiated = false;

	$effect(() => {
		if (serverStore.isRouterMode && currentSection.fields && !fetchInitiated) {
			fetchInitiated = true;

			void modelsStore
				.fetch()
				.then(() => modelsStore.fetchRouterModels())
				.then(() => modelsStore.fetchModalitiesForLoadedModels())
				.then(() => modelsStore.ensureFirstModelSelected());
		}
	});

	function handleThemeChange(newTheme: string) {
		localConfig.theme = newTheme;
		setMode(newTheme as ColorMode);
	}

	function handleConfigChange(key: string, value: string | boolean) {
		localConfig[key] = value;
	}

	function handleReset() {
		localConfig = { ...settingsStore.config };
		setMode(localConfig.theme as ColorMode);
		mobileHeader?.updateCarousel();
	}

	function handleSave() {
		if (
			localConfig.customJson &&
			typeof localConfig.customJson === 'string' &&
			localConfig.customJson.trim()
		) {
			try {
				JSON.parse(localConfig.customJson);
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
						const entryByMinMax = SETTINGS_CHAT_SECTIONS.flatMap(
							(section) => section.fields ?? []
						).find((entry) => entry.key === field);
						const lo = entryByMinMax?.min ?? 1;
						const hi = entryByMinMax?.max ?? Number.POSITIVE_INFINITY;

						processedConfig[field] = Math.max(lo, Math.min(hi, Math.round(numValue)));
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
		localConfig = { ...settingsStore.config };
	}

	setChatSettingsConfigContext({
		handleConfigChange,
		handleThemeChange,
		get localConfig() {
			return localConfig;
		}
	});
</script>

<div class="mx-auto flex h-full w-full flex-col md:pl-8" in:fade={{ duration: 150 }}>
	<div class="flex flex-1 flex-col gap-4 md:flex-row">
		<SettingsChatDesktopSidebar
			sections={SETTINGS_CHAT_SECTIONS}
			isActive={(section: SettingsSection) => section.slug === activeSlug}
			getHref={getSectionHref ??
				((section: SettingsSection) => RouterService.settings(section.slug))}
		/>

		<SettingsChatMobileHeader
			sections={SETTINGS_CHAT_SECTIONS}
			isActive={(section: SettingsSection) => section.slug === activeSlug}
			getHref={getSectionHref ??
				((section: SettingsSection) => RouterService.settings(section.slug))}
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
						<SettingsChatToolsTab />
					{:else if currentSection.title === SETTINGS_SECTION_TITLES.IMPORT_EXPORT}
						<SettingsChatImportExportTab />
					{:else if currentSection.fields}
						<div class="space-y-6">
							<SettingsChatFields
								fields={currentSection.fields}
								{localConfig}
								onConfigChange={handleConfigChange}
								onThemeChange={handleThemeChange}
							/>

							{#if currentSection.title === SETTINGS_SECTION_TITLES.GENERAL}
								<div class="flex justify-end">
									<Button variant="outline" onclick={() => window.location.reload()}>
										<RefreshCw class="h-3 w-3" />
										Reload app
									</Button>
								</div>
							{/if}
						</div>
					{/if}
				</div>

				<div class="mt-8 border-t border-border/30 pt-6">
					<p class="text-xs text-muted-foreground">Settings are saved in browser's localStorage</p>
				</div>
			</div>

			<SettingsFooter onReset={handleReset} onSave={handleSave} />
		</div>
	</div>
</div>
