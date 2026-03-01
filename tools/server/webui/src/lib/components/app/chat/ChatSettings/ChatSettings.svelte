<script lang="ts">
	import {
		Settings,
		Funnel,
		AlertTriangle,
		Code,
		Monitor,
		ChevronLeft,
		ChevronRight,
		Database
	} from '@lucide/svelte';
	import {
		ChatSettingsFooter,
		ChatSettingsImportExportTab,
		ChatSettingsFields
	} from '$lib/components/app';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import {
		SETTINGS_SECTION_TITLES,
		type SettingsSectionTitle
	} from '$lib/constants/settings-sections';
	import { setMode } from 'mode-watcher';
	import { ColorMode } from '$lib/enums/ui';
	import { SettingsFieldType } from '$lib/enums/settings';
	import type { Component } from 'svelte';
	import { NUMERIC_FIELDS, POSITIVE_INTEGER_FIELDS } from '$lib/constants/settings-fields';
	import { SETTINGS_COLOR_MODES_CONFIG } from '$lib/constants/settings-config';
	import { SETTINGS_KEYS } from '$lib/constants/settings-keys';

	interface Props {
		onSave?: () => void;
		initialSection?: SettingsSectionTitle;
	}

	let { onSave, initialSection }: Props = $props();

	const settingSections: Array<{
		fields: SettingsFieldConfig[];
		icon: Component;
		title: SettingsSectionTitle;
	}> = [
		{
			title: SETTINGS_SECTION_TITLES.GENERAL,
			icon: Settings,
			fields: [
				{
					key: SETTINGS_KEYS.THEME,
					label: 'Theme',
					type: SettingsFieldType.SELECT,
					options: SETTINGS_COLOR_MODES_CONFIG
				},
				{ key: SETTINGS_KEYS.API_KEY, label: 'API Key', type: SettingsFieldType.INPUT },
				{
					key: SETTINGS_KEYS.SYSTEM_MESSAGE,
					label: 'System Message',
					type: SettingsFieldType.TEXTAREA
				},
				{
					key: SETTINGS_KEYS.PASTE_LONG_TEXT_TO_FILE_LEN,
					label: 'Paste long text to file length',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.COPY_TEXT_ATTACHMENTS_AS_PLAIN_TEXT,
					label: 'Copy text attachments as plain text',
					type: SettingsFieldType.CHECKBOX
				},
				{
					key: SETTINGS_KEYS.ENABLE_CONTINUE_GENERATION,
					label: 'Enable "Continue" button',
					type: SettingsFieldType.CHECKBOX,
					isExperimental: true
				},
				{
					key: SETTINGS_KEYS.PDF_AS_IMAGE,
					label: 'Parse PDF as image',
					type: SettingsFieldType.CHECKBOX
				},
				{
					key: SETTINGS_KEYS.ASK_FOR_TITLE_CONFIRMATION,
					label: 'Ask for confirmation before changing conversation title',
					type: SettingsFieldType.CHECKBOX
				}
			]
		},
		{
			title: SETTINGS_SECTION_TITLES.DISPLAY,
			icon: Monitor,
			fields: [
				{
					key: SETTINGS_KEYS.SHOW_MESSAGE_STATS,
					label: 'Show message generation statistics',
					type: SettingsFieldType.CHECKBOX
				},
				{
					key: SETTINGS_KEYS.SHOW_THOUGHT_IN_PROGRESS,
					label: 'Show thought in progress',
					type: SettingsFieldType.CHECKBOX
				},
				{
					key: SETTINGS_KEYS.KEEP_STATS_VISIBLE,
					label: 'Keep stats visible after generation',
					type: SettingsFieldType.CHECKBOX
				},
				{
					key: SETTINGS_KEYS.AUTO_MIC_ON_EMPTY,
					label: 'Show microphone on empty input',
					type: SettingsFieldType.CHECKBOX,
					isExperimental: true
				},
				{
					key: SETTINGS_KEYS.RENDER_USER_CONTENT_AS_MARKDOWN,
					label: 'Render user content as Markdown',
					type: SettingsFieldType.CHECKBOX
				},
				{
					key: SETTINGS_KEYS.FULL_HEIGHT_CODE_BLOCKS,
					label: 'Use full height code blocks',
					type: SettingsFieldType.CHECKBOX
				},
				{
					key: SETTINGS_KEYS.DISABLE_AUTO_SCROLL,
					label: 'Disable automatic scroll',
					type: SettingsFieldType.CHECKBOX
				},
				{
					key: SETTINGS_KEYS.ALWAYS_SHOW_SIDEBAR_ON_DESKTOP,
					label: 'Always show sidebar on desktop',
					type: SettingsFieldType.CHECKBOX
				},
				{
					key: SETTINGS_KEYS.AUTO_SHOW_SIDEBAR_ON_NEW_CHAT,
					label: 'Auto-show sidebar on new chat',
					type: SettingsFieldType.CHECKBOX
				}
			]
		},
		{
			title: SETTINGS_SECTION_TITLES.SAMPLING,
			icon: Funnel,
			fields: [
				{
					key: SETTINGS_KEYS.TEMPERATURE,
					label: 'Temperature',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.DYNATEMP_RANGE,
					label: 'Dynamic temperature range',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.DYNATEMP_EXPONENT,
					label: 'Dynamic temperature exponent',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.TOP_K,
					label: 'Top K',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.TOP_P,
					label: 'Top P',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.MIN_P,
					label: 'Min P',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.XTC_PROBABILITY,
					label: 'XTC probability',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.XTC_THRESHOLD,
					label: 'XTC threshold',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.TYP_P,
					label: 'Typical P',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.MAX_TOKENS,
					label: 'Max tokens',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.SAMPLERS,
					label: 'Samplers',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.BACKEND_SAMPLING,
					label: 'Backend sampling',
					type: SettingsFieldType.CHECKBOX
				}
			]
		},
		{
			title: SETTINGS_SECTION_TITLES.PENALTIES,
			icon: AlertTriangle,
			fields: [
				{
					key: SETTINGS_KEYS.REPEAT_LAST_N,
					label: 'Repeat last N',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.REPEAT_PENALTY,
					label: 'Repeat penalty',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.PRESENCE_PENALTY,
					label: 'Presence penalty',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.FREQUENCY_PENALTY,
					label: 'Frequency penalty',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.DRY_MULTIPLIER,
					label: 'DRY multiplier',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.DRY_BASE,
					label: 'DRY base',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.DRY_ALLOWED_LENGTH,
					label: 'DRY allowed length',
					type: SettingsFieldType.INPUT
				},
				{
					key: SETTINGS_KEYS.DRY_PENALTY_LAST_N,
					label: 'DRY penalty last N',
					type: SettingsFieldType.INPUT
				}
			]
		},
		{
			title: SETTINGS_SECTION_TITLES.IMPORT_EXPORT,
			icon: Database,
			fields: []
		},
		{
			title: SETTINGS_SECTION_TITLES.DEVELOPER,
			icon: Code,
			fields: [
				{
					key: SETTINGS_KEYS.DISABLE_REASONING_PARSING,
					label: 'Disable reasoning content parsing',
					type: SettingsFieldType.CHECKBOX
				},
				{
					key: SETTINGS_KEYS.SHOW_RAW_OUTPUT_SWITCH,
					label: 'Enable raw output toggle',
					type: SettingsFieldType.CHECKBOX
				},
				{
					key: SETTINGS_KEYS.CUSTOM,
					label: 'Custom JSON',
					type: SettingsFieldType.TEXTAREA
				}
			]
		}
		// TODO: Experimental features section will be implemented after initial release
		// This includes Python interpreter (Pyodide integration) and other experimental features
		// {
		// 	title: 'Experimental',
		// 	icon: Beaker,
		// 	fields: [
		// 		{
		// 			key: 'pyInterpreterEnabled',
		// 			label: 'Enable Python interpreter',
		// 			type: 'checkbox'
		// 		}
		// 	]
		// }
	];

	let activeSection = $derived<SettingsSectionTitle>(
		initialSection ?? SETTINGS_SECTION_TITLES.GENERAL
	);
	let currentSection = $derived(
		settingSections.find((section) => section.title === activeSection) || settingSections[0]
	);
	let localConfig: SettingsConfigType = $state({ ...config() });

	let canScrollLeft = $state(false);
	let canScrollRight = $state(false);
	let scrollContainer: HTMLDivElement | undefined = $state();

	$effect(() => {
		if (initialSection) {
			activeSection = initialSection;
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
		localConfig = { ...config() };

		setMode(localConfig.theme as ColorMode);
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

		// Convert numeric strings to numbers for numeric fields
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
		onSave?.();
	}

	function scrollToCenter(element: HTMLElement) {
		if (!scrollContainer) return;

		const containerRect = scrollContainer.getBoundingClientRect();
		const elementRect = element.getBoundingClientRect();

		const elementCenter = elementRect.left + elementRect.width / 2;
		const containerCenter = containerRect.left + containerRect.width / 2;
		const scrollOffset = elementCenter - containerCenter;

		scrollContainer.scrollBy({ left: scrollOffset, behavior: 'smooth' });
	}

	function scrollLeft() {
		if (!scrollContainer) return;

		scrollContainer.scrollBy({ left: -250, behavior: 'smooth' });
	}

	function scrollRight() {
		if (!scrollContainer) return;

		scrollContainer.scrollBy({ left: 250, behavior: 'smooth' });
	}

	function updateScrollButtons() {
		if (!scrollContainer) return;

		const { scrollLeft, scrollWidth, clientWidth } = scrollContainer;
		canScrollLeft = scrollLeft > 0;
		canScrollRight = scrollLeft < scrollWidth - clientWidth - 1; // -1 for rounding
	}

	export function reset() {
		localConfig = { ...config() };

		setTimeout(updateScrollButtons, 100);
	}

	$effect(() => {
		if (scrollContainer) {
			updateScrollButtons();
		}
	});
</script>

<div class="flex h-full flex-col overflow-hidden md:flex-row">
	<!-- Desktop Sidebar -->
	<div class="hidden w-64 border-r border-border/30 p-6 md:block">
		<nav class="space-y-1 py-2">
			{#each settingSections as section (section.title)}
				<button
					class="flex w-full cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors hover:bg-accent {activeSection ===
					section.title
						? 'bg-accent text-accent-foreground'
						: 'text-muted-foreground'}"
					onclick={() => (activeSection = section.title)}
				>
					<section.icon class="h-4 w-4" />

					<span class="ml-2">{section.title}</span>
				</button>
			{/each}
		</nav>
	</div>

	<!-- Mobile Header with Horizontal Scrollable Menu -->
	<div class="flex flex-col pt-6 md:hidden">
		<div class="border-b border-border/30 py-4">
			<!-- Horizontal Scrollable Category Menu with Navigation -->
			<div class="relative flex items-center" style="scroll-padding: 1rem;">
				<button
					class="absolute left-2 z-10 flex h-6 w-6 items-center justify-center rounded-full bg-muted shadow-md backdrop-blur-sm transition-opacity hover:bg-accent {canScrollLeft
						? 'opacity-100'
						: 'pointer-events-none opacity-0'}"
					onclick={scrollLeft}
					aria-label="Scroll left"
				>
					<ChevronLeft class="h-4 w-4" />
				</button>

				<div
					class="scrollbar-hide overflow-x-auto py-2"
					bind:this={scrollContainer}
					onscroll={updateScrollButtons}
				>
					<div class="flex min-w-max gap-2">
						{#each settingSections as section (section.title)}
							<button
								class="flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm whitespace-nowrap transition-colors first:ml-4 last:mr-4 hover:bg-accent {activeSection ===
								section.title
									? 'bg-accent text-accent-foreground'
									: 'text-muted-foreground'}"
								onclick={(e: MouseEvent) => {
									activeSection = section.title;
									scrollToCenter(e.currentTarget as HTMLElement);
								}}
							>
								<section.icon class="h-4 w-4 flex-shrink-0" />
								<span>{section.title}</span>
							</button>
						{/each}
					</div>
				</div>

				<button
					class="absolute right-2 z-10 flex h-6 w-6 items-center justify-center rounded-full bg-muted shadow-md backdrop-blur-sm transition-opacity hover:bg-accent {canScrollRight
						? 'opacity-100'
						: 'pointer-events-none opacity-0'}"
					onclick={scrollRight}
					aria-label="Scroll right"
				>
					<ChevronRight class="h-4 w-4" />
				</button>
			</div>
		</div>
	</div>

	<ScrollArea class="max-h-[calc(100dvh-13.5rem)] flex-1 md:max-h-[calc(100vh-13.5rem)]">
		<div class="space-y-6 p-4 md:p-6">
			<div class="grid">
				<div class="mb-6 flex hidden items-center gap-2 border-b border-border/30 pb-6 md:flex">
					<currentSection.icon class="h-5 w-5" />

					<h3 class="text-lg font-semibold">{currentSection.title}</h3>
				</div>

				{#if currentSection.title === SETTINGS_SECTION_TITLES.IMPORT_EXPORT}
					<ChatSettingsImportExportTab />
				{:else}
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

			<div class="mt-8 border-t pt-6">
				<p class="text-xs text-muted-foreground">Settings are saved in browser's localStorage</p>
			</div>
		</div>
	</ScrollArea>
</div>

<ChatSettingsFooter onReset={handleReset} onSave={handleSave} />
