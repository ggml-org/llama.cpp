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
	import type { Component } from 'svelte';

	interface Props {
		onSave?: () => void;
	}

	let { onSave }: Props = $props();

	const settingSections: Array<{
		fields: SettingsFieldConfig[];
		icon: Component;
		title: string;
	}> = [
		{
			title: 'General',
			icon: Settings,
			fields: [
				{
					key: 'theme',
					label: 'Theme',
					type: 'select',
					options: [
						{ value: 'system', label: 'System', icon: Monitor },
						{ value: 'light', label: 'Light', icon: Sun },
						{ value: 'dark', label: 'Dark', icon: Moon }
					]
				},
				{ key: 'apiKey', label: 'API Key', type: 'input' },
				{
					key: 'systemMessage',
					label: 'System Message',
					type: 'textarea'
				},
				{
					key: 'pasteLongTextToFileLen',
					label: 'Paste long text to file length',
					type: 'input'
				},
				{
					key: 'copyTextAttachmentsAsPlainText',
					label: 'Copy text attachments as plain text',
					type: 'checkbox'
				},
				{
					key: 'enableContinueGeneration',
					label: 'Enable "Continue" button',
					type: 'checkbox',
					isExperimental: true
				},
				{
					key: 'pdfAsImage',
					label: 'Parse PDF as image',
					type: 'checkbox'
				},
				{
					key: 'askForTitleConfirmation',
					label: 'Ask for confirmation before changing conversation title',
					type: 'checkbox'
				}
			]
		},
		{
			title: 'Display',
			icon: Monitor,
			fields: [
				{
					key: 'showMessageStats',
					label: 'Show message generation statistics',
					type: 'checkbox'
				},
				{
					key: 'showThoughtInProgress',
					label: 'Show thought in progress',
					type: 'checkbox'
				},
				{
					key: 'keepStatsVisible',
					label: 'Keep stats visible after generation',
					type: 'checkbox'
				},
				{
					key: 'autoMicOnEmpty',
					label: 'Show microphone on empty input',
					type: 'checkbox',
					isExperimental: true
				},
				{
					key: 'renderUserContentAsMarkdown',
					label: 'Render user content as Markdown',
					type: 'checkbox'
				},
				{
					key: 'disableAutoScroll',
					label: 'Disable automatic scroll',
					type: 'checkbox'
				},
				{
					key: 'alwaysShowSidebarOnDesktop',
					label: 'Always show sidebar on desktop',
					type: 'checkbox'
				},
				{
					key: 'autoShowSidebarOnNewChat',
					label: 'Auto-show sidebar on new chat',
					type: 'checkbox'
				}
			]
		},
		{
			title: 'Sampling',
			icon: Funnel,
			fields: [
				{
					key: 'temperature',
					label: 'Temperature',
					type: 'input'
				},
				{
					key: 'dynatemp_range',
					label: 'Dynamic temperature range',
					type: 'input'
				},
				{
					key: 'dynatemp_exponent',
					label: 'Dynamic temperature exponent',
					type: 'input'
				},
				{
					key: 'top_k',
					label: 'Top K',
					type: 'input'
				},
				{
					key: 'top_p',
					label: 'Top P',
					type: 'input'
				},
				{
					key: 'min_p',
					label: 'Min P',
					type: 'input'
				},
				{
					key: 'xtc_probability',
					label: 'XTC probability',
					type: 'input'
				},
				{
					key: 'xtc_threshold',
					label: 'XTC threshold',
					type: 'input'
				},
				{
					key: 'typ_p',
					label: 'Typical P',
					type: 'input'
				},
				{
					key: 'max_tokens',
					label: 'Max tokens',
					type: 'input'
				},
				{
					key: 'samplers',
					label: 'Samplers',
					type: 'input'
				},
				{
					key: 'backend_sampling',
					label: 'Backend sampling',
					type: 'checkbox'
				}
			]
		},
		{
			title: 'Penalties',
			icon: AlertTriangle,
			fields: [
				{
					key: 'repeat_last_n',
					label: 'Repeat last N',
					type: 'input'
				},
				{
					key: 'repeat_penalty',
					label: 'Repeat penalty',
					type: 'input'
				},
				{
					key: 'presence_penalty',
					label: 'Presence penalty',
					type: 'input'
				},
				{
					key: 'frequency_penalty',
					label: 'Frequency penalty',
					type: 'input'
				},
				{
					key: 'dry_multiplier',
					label: 'DRY multiplier',
					type: 'input'
				},
				{
					key: 'dry_base',
					label: 'DRY base',
					type: 'input'
				},
				{
					key: 'dry_allowed_length',
					label: 'DRY allowed length',
					type: 'input'
				},
				{
					key: 'dry_penalty_last_n',
					label: 'DRY penalty last N',
					type: 'input'
				}
			]
		},
		{
			title: 'Import/Export',
			icon: Database,
			fields: []
		},
		{
			title: 'Developer',
			icon: Code,
			fields: [
				{
					key: 'showToolCalls',
					label: 'Show tool call labels',
					type: 'checkbox'
				},
				{
					key: 'disableReasoningFormat',
					label: 'Show raw LLM output',
					type: 'checkbox'
				},
				{
					key: 'custom',
					label: 'Custom JSON',
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

	let activeSection = $state('General');
	let currentSection = $derived(
		settingSections.find((section) => section.title === activeSection) || settingSections[0]
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
				alert('Invalid JSON in custom parameters. Please check the format and try again.');
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

				{#if currentSection.title === 'Import/Export'}
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
