<script lang="ts">
	import { Settings, ChevronLeft, ChevronRight } from '@lucide/svelte';
	import {
		ChatSettingsFooter,
		ChatSettingsFields,
		ChatSettingsToolsTab
	} from '$lib/components/app';
	import { config, settingsStore } from '$lib/stores/settings.svelte';
	import {
		SETTINGS_SECTION_TITLES,
		type SettingsSectionTitle,
		NUMERIC_FIELDS,
		POSITIVE_INTEGER_FIELDS,
		SETTINGS_CHAT_SECTIONS
	} from '$lib/constants';
	import { setMode } from 'mode-watcher';
	import { ColorMode } from '$lib/enums/ui';
	import { fade } from 'svelte/transition';

	interface Props {
		class?: string;
		onSave?: () => void;
		initialSection?: SettingsSectionTitle;
	}

	let { class: className, onSave, initialSection }: Props = $props();

	const settingSections = SETTINGS_CHAT_SECTIONS;

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

<div class="flex h-full flex-col overflow-y-auto {className} w-full" in:fade={{ duration: 150 }}>
	<div class="flex flex-1 flex-col gap-4 md:flex-row">
		<!-- Desktop Sidebar -->
		<div class="sticky top-0 hidden w-64 flex-col self-start bg-background pt-8 pb-4 md:flex">
			<div class="flex items-center gap-2 pb-12">
				<Settings class="h-6 w-6" />
				<h1 class="text-2xl font-semibold">Settings</h1>
			</div>
			<nav class="space-y-1">
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
		<div class="sticky top-0 z-10 flex flex-col bg-background md:hidden">
			<div class="flex items-center gap-2 px-4 pt-4 pb-2 md:pt-6">
				<Settings class="h-5 w-5 md:h-6 md:w-6" />

				<h1 class="text-xl font-semibold md:text-2xl">Settings</h1>
			</div>

			<div class="border-b border-border/30 py-2">
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
