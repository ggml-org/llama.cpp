<script lang="ts">
	import {
		Settings,
		Funnel,
		AlertTriangle,
		Code,
		Monitor,
		Sun,
		Moon,
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
	import { setMode } from 'mode-watcher';
	import { t } from '$lib/i18n';
	import type { Component } from 'svelte';

	interface Props {
		onSave?: () => void;
	}

	let { onSave }: Props = $props();

	const settingSections: Array<{
		id: string;
		fields: SettingsFieldConfig[];
		icon: Component;
		titleKey: string;
	}> = [
		{
			id: 'general',
			titleKey: 'chat.settings.section.general',
			icon: Settings,
			fields: [
				{
					key: 'theme',
					label: 'chat.settings.field.theme',
					type: 'select',
					options: [
						{ value: 'system', label: 'chat.settings.option.theme.system', icon: Monitor },
						{ value: 'light', label: 'chat.settings.option.theme.light', icon: Sun },
						{ value: 'dark', label: 'chat.settings.option.theme.dark', icon: Moon }
					]
				},
				{ key: 'apiKey', label: 'chat.settings.field.api_key', type: 'input' },
				{
					key: 'systemMessage',
					label: 'chat.settings.field.system_message',
					type: 'textarea'
				},
				{
					key: 'pasteLongTextToFileLen',
					label: 'chat.settings.field.paste_long_text_to_file_length',
					type: 'input'
				},
				{
					key: 'copyTextAttachmentsAsPlainText',
					label: 'chat.settings.field.copy_text_attachments_as_plain_text',
					type: 'checkbox'
				},
				{
					key: 'enableContinueGeneration',
					label: 'chat.settings.field.enable_continue_generation',
					type: 'checkbox',
					isExperimental: true
				},
				{
					key: 'pdfAsImage',
					label: 'chat.settings.field.parse_pdf_as_image',
					type: 'checkbox'
				},
				{
					key: 'askForTitleConfirmation',
					label: 'chat.settings.field.ask_for_title_confirmation',
					type: 'checkbox'
				}
			]
		},
		{
			id: 'display',
			titleKey: 'chat.settings.section.display',
			icon: Monitor,
			fields: [
				{
					key: 'showMessageStats',
					label: 'chat.settings.field.show_message_stats',
					type: 'checkbox'
				},
				{
					key: 'showThoughtInProgress',
					label: 'chat.settings.field.show_thought_in_progress',
					type: 'checkbox'
				},
				{
					key: 'keepStatsVisible',
					label: 'chat.settings.field.keep_stats_visible',
					type: 'checkbox'
				},
				{
					key: 'autoMicOnEmpty',
					label: 'chat.settings.field.show_microphone_on_empty',
					type: 'checkbox',
					isExperimental: true
				},
				{
					key: 'renderUserContentAsMarkdown',
					label: 'chat.settings.field.render_user_content_as_markdown',
					type: 'checkbox'
				},
				{
					key: 'disableAutoScroll',
					label: 'chat.settings.field.disable_auto_scroll',
					type: 'checkbox'
				},
				{
					key: 'alwaysShowSidebarOnDesktop',
					label: 'chat.settings.field.always_show_sidebar_on_desktop',
					type: 'checkbox'
				},
				{
					key: 'autoShowSidebarOnNewChat',
					label: 'chat.settings.field.auto_show_sidebar_on_new_chat',
					type: 'checkbox'
				}
			]
		},
		{
			id: 'sampling',
			titleKey: 'chat.settings.section.sampling',
			icon: Funnel,
			fields: [
				{
					key: 'temperature',
					label: 'chat.settings.field.temperature',
					type: 'input'
				},
				{
					key: 'dynatemp_range',
					label: 'chat.settings.field.dynamic_temperature_range',
					type: 'input'
				},
				{
					key: 'dynatemp_exponent',
					label: 'chat.settings.field.dynamic_temperature_exponent',
					type: 'input'
				},
				{
					key: 'top_k',
					label: 'chat.settings.field.top_k',
					type: 'input'
				},
				{
					key: 'top_p',
					label: 'chat.settings.field.top_p',
					type: 'input'
				},
				{
					key: 'min_p',
					label: 'chat.settings.field.min_p',
					type: 'input'
				},
				{
					key: 'xtc_probability',
					label: 'chat.settings.field.xtc_probability',
					type: 'input'
				},
				{
					key: 'xtc_threshold',
					label: 'chat.settings.field.xtc_threshold',
					type: 'input'
				},
				{
					key: 'typ_p',
					label: 'chat.settings.field.typical_p',
					type: 'input'
				},
				{
					key: 'max_tokens',
					label: 'chat.settings.field.max_tokens',
					type: 'input'
				},
				{
					key: 'samplers',
					label: 'chat.settings.field.samplers',
					type: 'input'
				},
				{
					key: 'backend_sampling',
					label: 'chat.settings.field.backend_sampling',
					type: 'checkbox'
				}
			]
		},
		{
			id: 'penalties',
			titleKey: 'chat.settings.section.penalties',
			icon: AlertTriangle,
			fields: [
				{
					key: 'repeat_last_n',
					label: 'chat.settings.field.repeat_last_n',
					type: 'input'
				},
				{
					key: 'repeat_penalty',
					label: 'chat.settings.field.repeat_penalty',
					type: 'input'
				},
				{
					key: 'presence_penalty',
					label: 'chat.settings.field.presence_penalty',
					type: 'input'
				},
				{
					key: 'frequency_penalty',
					label: 'chat.settings.field.frequency_penalty',
					type: 'input'
				},
				{
					key: 'dry_multiplier',
					label: 'chat.settings.field.dry_multiplier',
					type: 'input'
				},
				{
					key: 'dry_base',
					label: 'chat.settings.field.dry_base',
					type: 'input'
				},
				{
					key: 'dry_allowed_length',
					label: 'chat.settings.field.dry_allowed_length',
					type: 'input'
				},
				{
					key: 'dry_penalty_last_n',
					label: 'chat.settings.field.dry_penalty_last_n',
					type: 'input'
				}
			]
		},
		{
			id: 'import_export',
			titleKey: 'chat.settings.section.import_export',
			icon: Database,
			fields: []
		},
		{
			id: 'developer',
			titleKey: 'chat.settings.section.developer',
			icon: Code,
			fields: [
				{
					key: 'showToolCalls',
					label: 'chat.settings.field.show_tool_call_labels',
					type: 'checkbox'
				},
				{
					key: 'disableReasoningFormat',
					label: 'chat.settings.field.show_raw_llm_output',
					type: 'checkbox'
				},
				{
					key: 'custom',
					label: 'chat.settings.field.custom_json',
					type: 'textarea'
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

	let activeSection = $state('general');
	let currentSection = $derived(
		settingSections.find((section) => section.id === activeSection) || settingSections[0]
	);
	let localConfig: SettingsConfigType = $state({ ...config() });

	let canScrollLeft = $state(false);
	let canScrollRight = $state(false);
	let scrollContainer: HTMLDivElement | undefined = $state();

	function handleThemeChange(newTheme: string) {
		localConfig.theme = newTheme;

		setMode(newTheme as 'light' | 'dark' | 'system');
	}

	function handleConfigChange(key: string, value: string | boolean) {
		localConfig[key] = value;
	}

	function handleReset() {
		localConfig = { ...config() };

		setMode(localConfig.theme as 'light' | 'dark' | 'system');
	}

	function handleSave() {
		if (localConfig.custom && typeof localConfig.custom === 'string' && localConfig.custom.trim()) {
			try {
				JSON.parse(localConfig.custom);
			} catch (error) {
				alert(t('chat.settings.error.invalid_custom_json'));
				console.error(error);
				return;
			}
		}

		// Convert numeric strings to numbers for numeric fields
		const processedConfig = { ...localConfig };
		const numericFields = [
			'temperature',
			'top_k',
			'top_p',
			'min_p',
			'max_tokens',
			'pasteLongTextToFileLen',
			'dynatemp_range',
			'dynatemp_exponent',
			'typ_p',
			'xtc_probability',
			'xtc_threshold',
			'repeat_last_n',
			'repeat_penalty',
			'presence_penalty',
			'frequency_penalty',
			'dry_multiplier',
			'dry_base',
			'dry_allowed_length',
			'dry_penalty_last_n'
		];

		for (const field of numericFields) {
			if (processedConfig[field] !== undefined && processedConfig[field] !== '') {
				const numValue = Number(processedConfig[field]);
				if (!isNaN(numValue)) {
					processedConfig[field] = numValue;
				} else {
					alert(t('chat.settings.error.invalid_numeric', { field }));
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
			{#each settingSections as section (section.id)}
				<button
					class="flex w-full cursor-pointer items-center gap-3 rounded-lg px-3 py-2 text-left text-sm transition-colors hover:bg-accent {activeSection ===
					section.id
						? 'bg-accent text-accent-foreground'
						: 'text-muted-foreground'}"
					onclick={() => (activeSection = section.id)}
				>
					<section.icon class="h-4 w-4" />

					<span class="ml-2">{t(section.titleKey)}</span>
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
					aria-label={t('chat.settings.scroll_left')}
				>
					<ChevronLeft class="h-4 w-4" />
				</button>

				<div
					class="scrollbar-hide overflow-x-auto py-2"
					bind:this={scrollContainer}
					onscroll={updateScrollButtons}
				>
					<div class="flex min-w-max gap-2">
						{#each settingSections as section (section.id)}
							<button
								class="flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm whitespace-nowrap transition-colors first:ml-4 last:mr-4 hover:bg-accent {activeSection ===
								section.id
									? 'bg-accent text-accent-foreground'
									: 'text-muted-foreground'}"
								onclick={(e: MouseEvent) => {
									activeSection = section.id;
									scrollToCenter(e.currentTarget as HTMLElement);
								}}
							>
								<section.icon class="h-4 w-4 flex-shrink-0" />
								<span>{t(section.titleKey)}</span>
							</button>
						{/each}
					</div>
				</div>

				<button
					class="absolute right-2 z-10 flex h-6 w-6 items-center justify-center rounded-full bg-muted shadow-md backdrop-blur-sm transition-opacity hover:bg-accent {canScrollRight
						? 'opacity-100'
						: 'pointer-events-none opacity-0'}"
					onclick={scrollRight}
					aria-label={t('chat.settings.scroll_right')}
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

					<h3 class="text-lg font-semibold">{t(currentSection.titleKey)}</h3>
				</div>

				{#if currentSection.id === 'import_export'}
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
				<p class="text-xs text-muted-foreground">{t('chat.settings.saved_notice')}</p>
			</div>
		</div>
	</ScrollArea>
</div>

<ChatSettingsFooter onReset={handleReset} onSave={handleSave} />
