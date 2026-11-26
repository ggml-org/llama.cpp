<script lang="ts">
	import { Button } from '@ggml-org/llama-ui/ui/button';
	import * as Collapsible from '@ggml-org/llama-ui/ui/collapsible';
	import { ChevronDown, ChevronRight, RefreshCw } from '@lucide/svelte';
	import {
		SettingsChatDesktopSidebar,
		SettingsChatFields,
		SettingsChatImportExportTab,
		SettingsChatMobileHeader,
		SettingsChatToolsTab,
		SettingsChatVoiceTab,
		SettingsFooter,
		SettingsGroup,
		SettingsResetButton
	} from '#lib/components/app/settings/index.ts';
	import {
		NUMERIC_FIELDS,
		POSITIVE_INTEGER_FIELDS,
		SETTINGS_KEYS,
		SETTINGS_SECTION_SLUGS,
		settingsChatSections
	} from '#lib/constants/index.ts';
	import { ColorMode } from '#lib/enums/ui.enums.ts';
	import { agenticMode } from '#lib/stores/agentic-mode.svelte.ts';
	import { deviceStore, modelsStore, serverStore, settingsStore } from '#lib/stores/index.ts';
	import type { SettingsSection, SettingsSectionTitle } from '#lib/types/index.ts';
	import { groupSettingsFields, isApiKeyFieldShown, isSettingShownOn } from '#lib/utils/index.ts';
	import { page } from '$app/state';
	import { setMode } from 'mode-watcher';
	import { tick } from 'svelte';
	import { SvelteSet } from 'svelte/reactivity';
	import { fade } from 'svelte/transition';
	interface Props {
		initialSection?: string;
		/** Where the phone's section tabs pin, see SettingsChatMobileHeader */
		mobileHeaderClass?: string;
		onSectionChange?: (section: SettingsSectionTitle) => void;
		onClose?: () => void;
	}

	let { initialSection, mobileHeaderClass, onClose, onSectionChange }: Props = $props();

	const sections = $derived(settingsChatSections(agenticMode.enabled));

	let activeSlug = $derived(initialSection ?? SETTINGS_SECTION_SLUGS.GENERAL);

	let contentEl = $state<HTMLDivElement>();

	// a new section starts at its top, whether the dialog, the drawer or the page scrolls
	function scrollToTop() {
		let el: HTMLElement | null = contentEl ?? null;

		while (el) {
			const { overflowY } = getComputedStyle(el);

			if ((overflowY === 'auto' || overflowY === 'scroll') && el.scrollHeight > el.clientHeight)
				break;

			el = el.parentElement;
		}

		(el ?? document.scrollingElement)?.scrollTo({ top: 0 });
	}

	function handleSectionChange(section: SettingsSectionTitle) {
		const found = sections.find((s) => s.title === section);

		if (found && found.slug !== activeSlug) {
			activeSlug = found.slug;
			void tick().then(scrollToTop);
		}

		onSectionChange?.(section);
	}

	let localConfig: SettingsConfigType = $state({ ...settingsStore.config });

	let currentSection = $derived(
		sections.find((section) => section.slug === activeSlug) || sections[0]
	);

	let showApiKey = $derived(
		isApiKeyFieldShown({
			draftKey: localConfig[SETTINGS_KEYS.API_KEY],
			status: page.status,
			storedKey: settingsStore.config[SETTINGS_KEYS.API_KEY]
		})
	);

	let sectionFields = $derived(
		(currentSection.fields ?? []).filter(
			(field) =>
				(field.key !== SETTINGS_KEYS.API_KEY || showApiKey) &&
				isSettingShownOn(field, deviceStore.isMobile)
		)
	);
	let basicFields = $derived(sectionFields.filter((field) => !field.advanced));
	let fieldGroups = $derived(groupSettingsFields(basicFields));
	let advancedGroups = $derived(
		groupSettingsFields(sectionFields.filter((field) => field.advanced))
	);

	// sections whose Advanced block is open, kept while the settings stay open
	let advancedOpen = new SvelteSet<string>();

	function toggleAdvanced(slug: string) {
		if (advancedOpen.has(slug)) advancedOpen.delete(slug);
		else advancedOpen.add(slug);
	}

	// inputs hand back strings for numeric fields, so compare stringified values
	let hasChanges = $derived.by(() => {
		const saved = settingsStore.config;
		const keys = new Set([...Object.keys(saved), ...Object.keys(localConfig)]);

		for (const key of keys) {
			if (String(localConfig[key] ?? '') !== String(saved[key] ?? '')) {
				return true;
			}
		}

		return false;
	});

	let mobileHeader: { updateCarousel: () => void } | undefined;

	let fetchInitiated = false;

	$effect(() => {
		if (serverStore.isRouterMode && currentSection.fields?.length && !fetchInitiated) {
			fetchInitiated = true;

			void modelsStore
				.fetch()
				.then(() => modelsStore.fetchRouterModels())
				.then(() => modelsStore.props.fetchModalitiesForLoadedModels())
				.then(() => modelsStore.ensureFirstModelSelected());
		}
	});

	function handleThemeChange(newTheme: string) {
		localConfig.theme = newTheme;
		setMode(newTheme as ColorMode);
	}

	function handleConfigChange(key: string, value: string | boolean) {
		localConfig[key] = value;

		// discover needs the hub metadata reads: checking it turns the hub on,
		// unchecking it turns the hub back off, so the checkbox never lies
		if (key === SETTINGS_KEYS.ENABLE_DISCOVER_MODELS) {
			localConfig[SETTINGS_KEYS.USE_HUGGING_FACE_HUB] = value === true;

			// it opens from the Models page, so checking it shows Models too; unchecking it
			// leaves Models as it is, since Models is useful without it
			if (value === true) localConfig[SETTINGS_KEYS.SHOW_MODELS_MANAGER] = true;
		}
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
						const entryByMinMax = sections
							.flatMap((section) => section.fields ?? [])
							.find((entry) => entry.key === field);
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
		onClose?.();
	}

	export function reset() {
		localConfig = { ...settingsStore.config };
	}
</script>

{#snippet groups(list: ReturnType<typeof groupSettingsFields>)}
	{#each list as group (group.title)}
		<SettingsGroup title={group.title}>
			<div class="space-y-6">
				<SettingsChatFields
					fields={group.fields}
					{localConfig}
					onConfigChange={handleConfigChange}
					onThemeChange={handleThemeChange}
				/>
			</div>
		</SettingsGroup>
	{/each}
{/snippet}

<div in:fade={{ duration: 150 }} class="mx-auto flex size-full flex-col">
	<div class="flex flex-1 flex-col md:flex-row md:gap-4">
		<SettingsChatDesktopSidebar
			isActive={(section: SettingsSection) => section.slug === activeSlug}
			onSectionChange={handleSectionChange}
			{sections}
		/>

		<SettingsChatMobileHeader
			bind:this={mobileHeader}
			class={mobileHeaderClass}
			isActive={(section: SettingsSection) => section.slug === activeSlug}
			onSectionChange={handleSectionChange}
			{sections}
		/>

		<div bind:this={contentEl} class="mx-auto max-w-2xl px-4 flex-1 md:mt-4">
			<div class="space-y-6 pt-3">
				<div class="grid">
					{#if currentSection.slug === SETTINGS_SECTION_SLUGS.DATA}
						<div class="space-y-12">
							<SettingsChatImportExportTab />

							<SettingsGroup title="App">
								<div class="flex flex-wrap gap-2">
									<SettingsResetButton onReset={handleReset} />

									<Button onclick={() => window.location.reload()} variant="outline">
										<RefreshCw class="size-icon-md" />
										Reload app
									</Button>
								</div>
							</SettingsGroup>
						</div>
					{:else}
						<div class="space-y-12">
							{#if currentSection.slug === SETTINGS_SECTION_SLUGS.VOICE}
								<SettingsChatVoiceTab
									fields={basicFields}
									{localConfig}
									onConfigChange={handleConfigChange}
								/>
							{:else}
								{#if currentSection.slug === SETTINGS_SECTION_SLUGS.AGENT}
									<SettingsGroup title="Tools">
										<SettingsChatToolsTab />
									</SettingsGroup>
								{/if}

								{@render groups(fieldGroups)}
							{/if}

							{#if advancedGroups.length > 0}
								{@const isOpen = advancedOpen.has(currentSection.slug)}

								<Collapsible.Root
									onOpenChange={() => toggleAdvanced(currentSection.slug)}
									open={isOpen}
								>
									<Collapsible.Trigger
										class="flex w-full cursor-pointer items-center gap-2 border-t border-border/30 pt-6 text-body font-semibold"
									>
										{#if isOpen}
											<ChevronDown class="size-icon-sm shrink-0" />
										{:else}
											<ChevronRight class="size-icon-sm shrink-0" />
										{/if}

										Advanced settings
									</Collapsible.Trigger>

									<Collapsible.Content>
										<div class="space-y-12 pt-8">
											{@render groups(advancedGroups)}
										</div>
									</Collapsible.Content>
								</Collapsible.Root>
							{/if}
						</div>
					{/if}
				</div>

				<div class="mt-8 border-t border-border/30 pt-6 pb-6">
					<p class="text-caption text-muted-foreground">
						Settings are saved in browser's localStorage
					</p>
				</div>
			</div>

			<SettingsFooter {hasChanges} onSave={handleSave} />
		</div>
	</div>
</div>
