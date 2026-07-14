<script lang="ts">
	import { Settings, ChevronLeft, ChevronRight } from '@lucide/svelte';
	import { onMount, tick } from 'svelte';
	import type { SettingsSection, SettingsSectionTitle } from '$lib/constants';
	import { useScrollCarousel } from '$lib/hooks/use-scroll-carousel.svelte';

	interface Props {
		sections: SettingsSection[];
		isActive: (section: SettingsSection) => boolean;
		getHref?: (section: SettingsSection) => string;
		onSectionChange?: (section: SettingsSectionTitle) => void;
	}

	let { sections, isActive, getHref, onSectionChange }: Props = $props();

	const carousel = useScrollCarousel();

	onMount(async () => {
		await tick();
		if (carousel.scrollContainer) {
			const activeTab = carousel.scrollContainer.querySelector('[data-active="true"]');
			if (activeTab instanceof HTMLElement) {
				carousel.scrollToCenter(activeTab);
			}
		}
	});

	export function updateCarousel() {
		setTimeout(carousel.updateScrollButtons, 100);
	}
</script>

<div class="sticky top-0 z-10 flex flex-col bg-background md:hidden">
	<div class="flex items-center gap-2 px-4 pt-4 pb-2 md:pt-6">
		<Settings class="h-5 w-5 md:h-6 md:w-6" />

		<h1 class="text-xl font-semibold md:text-2xl">Settings</h1>
	</div>

	<div class="border-b border-border/30 py-2">
		<div class="relative flex items-center" style="scroll-padding: 1rem;">
			<button
				class="absolute left-2 z-10 flex h-6 w-6 items-center justify-center rounded-full bg-muted shadow-md backdrop-blur-sm transition-opacity hover:bg-accent {carousel.canScrollLeft
					? 'opacity-100'
					: 'pointer-events-none opacity-0'}"
				onclick={carousel.scrollLeft}
				aria-label="Scroll left"
			>
				<ChevronLeft class="h-4 w-4" />
			</button>

			<div
				class="scrollbar-hide overflow-x-auto py-2"
				bind:this={carousel.scrollContainer}
				onscroll={carousel.updateScrollButtons}
			>
				<div class="flex min-w-max gap-2">
					{#each sections as section (section.title)}
						{#if getHref}
							<a
								class="flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm whitespace-nowrap no-underline transition-colors first:ml-4 last:mr-4 hover:bg-accent {isActive(
									section
								)
									? 'bg-accent text-accent-foreground'
									: 'text-muted-foreground'}"
								data-active={isActive(section)}
								href={getHref(section)}
								onclick={(e: MouseEvent) => {
									carousel.scrollToCenter(e.currentTarget as HTMLElement);
								}}
							>
								<section.icon class="h-4 w-4 flex-shrink-0" />
								<span>{section.title}</span>
							</a>
						{:else}
							<button
								class="flex cursor-pointer items-center gap-2 rounded-lg px-3 py-2 text-sm whitespace-nowrap transition-colors first:ml-4 last:mr-4 hover:bg-accent {isActive(
									section
								)
									? 'bg-accent text-accent-foreground'
									: 'text-muted-foreground'}"
								data-active={isActive(section)}
								onclick={(e: MouseEvent) => {
									onSectionChange?.(section.title);
									carousel.scrollToCenter(e.currentTarget as HTMLElement);
								}}
							>
								<section.icon class="h-4 w-4 flex-shrink-0" />
								<span>{section.title}</span>
							</button>
						{/if}
					{/each}
				</div>
			</div>

			<button
				class="absolute right-2 z-10 flex h-6 w-6 items-center justify-center rounded-full bg-muted shadow-md backdrop-blur-sm transition-opacity hover:bg-accent {carousel.canScrollRight
					? 'opacity-100'
					: 'pointer-events-none opacity-0'}"
				onclick={carousel.scrollRight}
				aria-label="Scroll right"
			>
				<ChevronRight class="h-4 w-4" />
			</button>
		</div>
	</div>
</div>
